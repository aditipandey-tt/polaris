#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.op import CCLOpHandle

def get_num_devices(mesh_device):
    """Dynamically determine the number of chips in the mesh."""
    if hasattr(mesh_device, 'shape'):
        # For a 2x4 mesh, this returns 8
        return mesh_device.shape[0] * mesh_device.shape[1]
    return 8  # Fallback default

def all_reduce(tensor, mesh_device, cluster_axis=0, dim=3, **kwargs):
    """
    Official Polaris Implementation of All-Reduce.
    Integrates latency and traffic into opstats.csv via CCLOpHandle.
    """
    num_devices = get_num_devices(mesh_device)
    
    # 1. Performance Math (Traffic and Latency)
    nelems_full = tensor.nelems()
    in_bytes = nelems_full * 2
    traffic_bytes = in_bytes * (num_devices - 1)
    
    bw_gbps = 7.0  # Effective bandwidth for Wormhole
    latency_ms = (traffic_bytes / (1024**3) / bw_gbps) * 1000
    frequency_mhz = 1000 
    target_cycles = int(latency_ms * frequency_mhz)
    print(f"[DEBUG CCL] latency_ms={latency_ms:.6f}, target_cycles={target_cycles}")



    # Generic Name based on tensor name
    op_name = f"CCL_AllReduce_{getattr(tensor, 'name', 'output')}"

    # 2. Handover to Polaris Engine
    # Isse shape update aur perf_stats integration automatic ho jayega
    output = CCLOpHandle(
        name=op_name, 
        optype="all_reduce", 
        in_tensor=tensor, 
        num_devices=num_devices, 
        latency_ms=latency_ms
    )
    try:
        from ttsim.ops.op import SimOp
        op = SimOp({
            'name': op_name,
            'optype': 'all_reduce',
            'attrs': {'latency_ms': latency_ms, 'traffic_bytes': traffic_bytes}
        })
        
    # Set performance stats
        print(f"[DEBUG CCL] Setting mem_rd_cycles={target_cycles//2}, mem_wr_cycles={target_cycles//2}")
        nelems = tensor.nelems()
        target_cycles = int(latency_ms * frequency_mhz)
        op.perf_stats = {
            'inBytes': in_bytes,
            'outBytes': in_bytes,
            'inElems': nelems,        # <--- Mandatory key 1
            'outElems': nelems,
            'cycles': target_cycles,  # Convert to cycles
            'mem_rd_cycles': target_cycles // 2,
            'mem_wr_cycles': target_cycles // 2,
            'instrs': {'mov': int(traffic_bytes/4)},
            'inParamCount': 0,      # CCL has no parameters
            'inActCount': nelems,   # Input activations
            'outActCount': nelems,  # Output activations


        }
        
        # Register with device
        if hasattr(mesh_device, 'add_op'):
            mesh_device.add_op(op)
            print(f"[CCL] Registered: {op_name}")
    except Exception as e:
        print(f"[CCL] Warning: Could not register op: {e}")
    
    # 4. Return output
    return output

def all_gather(tensor, mesh_device, cluster_axis=0, dim=3, **kwargs):
    """Official Polaris Implementation of All-Gather."""
    num_devices = get_num_devices(mesh_device)
    
    in_bytes = tensor.nelems() * 2
    bw_gbps = 7.0
    latency_ms = (in_bytes * (num_devices - 1) / (1024**3) / bw_gbps) * 1000
    
    op_name = f"CCL_AllGather_{getattr(tensor, 'name', 'output')}"
    
    return CCLOpHandle(
        name=op_name, 
        optype="all_gather", 
        in_tensor=tensor, 
        num_devices=num_devices, 
        latency_ms=latency_ms
    )

# Aliases for compatibility
tt_all_reduce = all_reduce
tt_all_gather = all_gather
