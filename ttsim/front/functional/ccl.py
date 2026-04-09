#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

global_ccl_latency = 0.0
def get_num_devices(mesh_device):
    """Dynamically determine the number of chips in the mesh."""
    if hasattr(mesh_device, 'shape'):
        # For a 2x4 mesh, this returns 8
        return mesh_device.shape[0] * mesh_device.shape[1]
    return 8  # Fallback for Mixtral 8-chip default

def all_reduce(tensor, mesh_device, cluster_axis=0, dim=3, **kwargs):
    """
    Full Polaris Implementation of All-Reduce:
    1. Scales data by NumDevices (Sum mimic)
    2. Shards shape by NumDevices (Distributed state mimic)
    3. Bills hardware for Adds, Loads, and Stores (Perf accounting)
    """
    global global_ccl_latency
    num_devices = get_num_devices(mesh_device)
    # 1. Performance Accounting (The Rama Requirement)
    nelems_full = 1
    if hasattr(tensor, 'shape'):
        for d in tensor.shape:
            nelems_full *= d
    in_bytes = nelems_full * 2
    traffic_bytes = in_bytes * (num_devices - 1)
    # Account for the reduction math and data movement overhead
    bw_gbps = 7.0  # 7 GB/s effective bandwidth for Wormhole
    latency_ms = (traffic_bytes / (1024**3) / bw_gbps) * 1000
    global_ccl_latency += latency_ms
    tensor.perf_stats = {
        'op_type': 'all_reduce',
        'inElems': nelems_full,
        'outElems': nelems_full,
        'inBytes': in_bytes,
        'outBytes': in_bytes,
        'instrs': {
            'add': nelems_full * (num_devices - 1),
            'load': nelems_full * num_devices,
            'store': nelems_full * num_devices
        },
        'mesh_traffic_bytes': traffic_bytes,
        'network_transfer_kb': traffic_bytes / 1024,
        'latency_ms': latency_ms,
        'num_devices': num_devices
    }

    if hasattr(tensor, 'data') and tensor.data is not None:
        tensor.data = tensor.data * num_devices

    if hasattr(tensor, 'shape'):
        new_shape = list(tensor.shape)
        new_shape[dim] = new_shape[dim] // num_devices
        tensor.shape = new_shape
        print(f"[CCL] all_reduce: Mimicked {num_devices} devices. Sharding dim {dim}: {new_shape[dim]}")
        print("="*60)
        print(f"CCL LAYER BREAKDOWN - DIM {dim}")
        print(f"  - Input Data: {in_bytes / 1024:.2f} KB")
        print(f"  - Network Traffic: {traffic_bytes / 1024:.2f} KB")
        print(f"  - Project Latency: {latency_ms:.6f} ms")
        print(f"  - Hardware Instrs: {tensor.perf_stats['instrs']}")
        print("="*60 + "\n")
    return tensor
def all_gather(tensor, mesh_device, cluster_axis=0, dim=3, **kwargs):
    """Full Polaris Implementation of All-Gather."""
    num_devices = get_num_devices(mesh_device)
    nelems = 1
    if hasattr(tensor, 'shape'):
        for d in tensor.shape: nelems *= d
    in_bytes = nelems * 2
    
    # 1. Update Shape
    if hasattr(tensor, 'shape'):
        new_shape = list(tensor.shape)
        new_shape[dim] *= num_devices
        tensor.shape = new_shape
    
    # 2. Accounting (Standard Polaris Keys)
    bw_gbps = 7.0
    latency_ms = (in_bytes * (num_devices - 1) / (1024**3) / bw_gbps) * 1000
    
    tensor.perf_stats = {
        'op_type': 'all_gather', 
        'inBytes': in_bytes, 
        'outBytes': in_bytes * num_devices,
        'latency_ms': latency_ms,
        'instrs': {'add': 0, 'load': nelems, 'store': nelems * num_devices}
    }
    return tensor
tt_all_reduce = all_reduce
tt_all_gather = all_gather
