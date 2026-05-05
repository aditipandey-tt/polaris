#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.op import CCLOpHandle
import re

def get_num_devices(mesh_device):
    """Dynamically determine the number of chips in the mesh."""
    if hasattr(mesh_device, 'shape'):
        return mesh_device.shape[0] * mesh_device.shape[1]
    import sys
    cmd_line = ' '.join(sys.argv)
    if '--filterarch n300' in cmd_line or 'n300' in cmd_line:
        return 2
    elif '--filterarch n150' in cmd_line or 'n150' in cmd_line:
        return 1
    elif '--filterarch n800' in cmd_line or 'n800' in cmd_line:
        return 8
    return 8 

def all_reduce(tensor, mesh_device, cluster_axis=0, dim=3, **kwargs):
    """Cleaned Dynamic All-Reduce for Llama3"""
    num_devices = get_num_devices(mesh_device)
    nelems_val = tensor.nelems() # Variable define kiya
    in_bytes = nelems_val * 2
    traffic_bytes = in_bytes * (num_devices - 1)
    latency_ms = (traffic_bytes / (1024**3) / 7.0) * 1000
    target_cycles = int(latency_ms * 1000)

    in_name = getattr(tensor, 'name', 'unknown_in')
    match = re.search(r'Op_(\d+)', in_name)
    if match:
        curr_id = int(match.group(1))
        target_out = in_name.replace(f"Op_{curr_id}", f"Op_{curr_id + 1}")
    else:
        target_out = in_name + ".reduce_out"

    output = CCLOpHandle(name=target_out.replace(".out", ""), optype="all_reduce", in_tensor=tensor, num_devices=num_devices, latency_ms=latency_ms)

    try:
        from ttsim.ops.tensor import SimTensor
        from ttsim.ops.op import SimOp

        if hasattr(mesh_device, 'tensors'): mesh_device.tensors[in_name] = tensor
        if hasattr(mesh_device, '_tracker'): mesh_device._tracker.tensors[in_name] = tensor

        op = SimOp({
            'name': f"CCL_Reduce_{target_out}",
            'optype': 'all_reduce',
            'inList': [in_name],
            'outList': [target_out],
            'attrs': {'latency_ms': latency_ms}
        })
        
        op.perf_stats = {
            'cycles': target_cycles, 
            'inBytes': in_bytes, 
            'outBytes': in_bytes, 
            'inElems': nelems_val, 
            'outElems': nelems_val, 
            'instrs': {'mov': int(nelems_val)},
            'inParamCount': 0,
            'inActCount': int(nelems_val),
            'outActCount': int(nelems_val),
            'mem_rd_cycles': target_cycles // 2,
            'mem_wr_cycles': target_cycles // 2
        }

        if hasattr(mesh_device, 'add_op'):
            mesh_device.add_op(op)
            out_obj = SimTensor({'name': target_out, 'shape': tensor.shape, 'dtype': tensor.dtype})
            if hasattr(mesh_device, 'tensors'): mesh_device.tensors[target_out] = out_obj
            if hasattr(mesh_device, '_tracker'): mesh_device._tracker.tensors[target_out] = out_obj

    except Exception as e:
        print(f"[CCL REDUCE ERROR] {e}")

    output.name = target_out
    return output

def all_gather(tensor, mesh_device, cluster_axis=0, dim=3, **kwargs):
    """Cleaned Dynamic All-Gather for Llama3"""
    num_devices = get_num_devices(mesh_device)
    nelems_val = tensor.nelems() # Variable define kiya
    in_bytes = nelems_val * 2
    traffic_bytes = in_bytes * (num_devices - 1)
    latency_ms = (traffic_bytes / (1024**3) / 7.0) * 1000
    target_cycles = int(latency_ms * 1000)
    in_name = getattr(tensor, 'name', 'unknown_in')
    match = re.search(r'Op_(\d+)', in_name)
    if match:
        curr_id = int(match.group(1))
        target_out = in_name.replace(f"Op_{curr_id}", f"Op_{curr_id + 1}")
    else:
        target_out = in_name + ".gather_out"

    output = CCLOpHandle(name=target_out.replace(".out", ""), optype="all_gather", in_tensor=tensor, num_devices=num_devices, latency_ms=latency_ms, dim=dim)

    try:
        from ttsim.ops.tensor import SimTensor
        from ttsim.ops.op import SimOp

        if hasattr(mesh_device, 'tensors'): mesh_device.tensors[in_name] = tensor
        if hasattr(mesh_device, '_tracker'): mesh_device._tracker.tensors[in_name] = tensor

        out_shape = list(tensor.shape)
        try: out_shape[dim] *= num_devices
        except: out_shape[-1] *= num_devices

        op = SimOp({
            'name': f"CCL_Gather_{target_out}",
            'optype': 'all_gather',
            'inList': [in_name],
            'outList': [target_out],
            'attrs': {'latency_ms': latency_ms}
        })
        
        op.perf_stats = {
            'cycles': target_cycles, 
            'inBytes': in_bytes, 
            'outBytes': in_bytes * num_devices, # Gathered size
            'inElems': nelems_val, 
            'outElems': nelems_val * num_devices, # Gathered size
            'instrs': {'mov': int(nelems_val * num_devices)},
            'inParamCount': 0,
            'inActCount': int(nelems_val),
            'outActCount': int(nelems_val * num_devices),
            'mem_rd_cycles': target_cycles // 2,
            'mem_wr_cycles': target_cycles // 2
        }

        if hasattr(mesh_device, 'add_op'):
            mesh_device.add_op(op)
            out_obj = SimTensor({'name': target_out, 'shape': out_shape, 'dtype': tensor.dtype})
            if hasattr(mesh_device, 'tensors'): mesh_device.tensors[target_out] = out_obj
            if hasattr(mesh_device, '_tracker'): mesh_device._tracker.tensors[target_out] = out_obj
            print(f"[DEBUG CCL] Linked {in_name} -> {target_out}")

    except Exception as e:
        print(f"[CCL GATHER ERROR] {e}")

    output.name = target_out
    return output

tt_all_reduce = all_reduce
tt_all_gather = all_gather
