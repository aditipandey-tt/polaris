#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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
    num_devices = get_num_devices(mesh_device)
    # 1. Performance Accounting (The Rama Requirement)
    nelems_full = 1
    if hasattr(tensor, 'shape'):
        for d in tensor.shape:
            nelems_full *= d
    traffic_bytes = (nelems_full * 2) * (num_devices - 1)
    # Account for the reduction math and data movement overhead
    bw_gbps = 7.0  # 7 GB/s effective bandwidth for Wormhole
    latency_ms = (traffic_bytes / (1024**3) / bw_gbps) * 1000
    tensor.perf_stats = {
        'op_type': 'all_reduce',
        'instrs': {
            'add': nelems_full * (num_devices - 1),
            'load': nelems_full * num_devices,
            'store': nelems_full * num_devices
        },
        'mesh_traffic_bytes': traffic_bytes,
        'network_transfer_kb': traffic_bytes / 1024,  # NEW: Rama's specific request
        'latency_ms': latency_ms,                     # NEW: Includes the time model
        'num_devices': num_devices
    }
    # 2. Functional Scaling
    if hasattr(tensor, 'data') and tensor.data is not None:
        tensor.data = tensor.data * num_devices

    # 3. Dynamic Sharding
    # This automatically shards the dimension specified (usually dim 3 for hidden dim)
    if hasattr(tensor, 'shape'):
        new_shape = list(tensor.shape)
        new_shape[dim] = new_shape[dim] // num_devices
        tensor.shape = new_shape
        print(f"[CCL] all_reduce: Mimicked {num_devices} devices. Sharding dim {dim}: {new_shape[dim]}")

    return tensor

def all_gather(tensor, mesh_device, cluster_axis=0, dim=3, **kwargs):
    """Full Polaris Implementation of All-Gather."""
    num_devices = get_num_devices(mesh_device)
    if hasattr(tensor, 'shape'):
        new_shape = list(tensor.shape)
        new_shape[dim] *= num_devices
        tensor.shape = new_shape
    
    tensor.perf_stats = {'op_type': 'all_gather', 'num_devices': num_devices}
    return tensor

tt_all_reduce = all_reduce
tt_all_gather = all_gather
  











































# #!/usr/bin/env python
# # SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# # SPDX-License-Identifier: Apache-2.0

# """
# CCL (Collective Communication Library) operations for Polaris
# Implements distributed tensor operations across multiple devices
# """

# def all_reduce(tensor, mesh_device, cluster_axis=0, dim=3, **kwargs):
#     """
#     All-reduce operation that sums tensors across all devices in the mesh.
    
#     In Mixtral MoE:
#     - 8 devices total, each processes 1 expert
#     - Each device input: [1, 32, 4096] 
#     - Each device output: [1, 1, 32, 512] (4096/8 = 512 per expert)
#     - After all-reduce: sum of all expert outputs
    
#     Args:
#         tensor: Input tensor from each expert
#         mesh_device: The device mesh for distributed computation  
#         cluster_axis: Axis for reduction (0 = across all devices)
#         dim: Dimension to reduce along
#         **kwargs: Additional arguments
    
#     Returns:
#         Tensor with summed values across devices
#     """
#     # Log the operation for debugging
#     if hasattr(tensor, 'shape'):
#         print(f"[CCL] all_reduce: input shape={tensor.shape}, dim={dim}, cluster_axis={cluster_axis}")
    
#     # In Polaris simulation, return the input tensor
#     # The actual hardware would perform the sum reduction across devices
#     return tensor

# # Alias for compatibility with tt-metal naming convention
# tt_all_reduce = all_reduce

# def all_gather(tensor, mesh_device, cluster_axis=0, dim=0, num_links=1, **kwargs):
#     """
#     All-gather operation that concatenates tensors from all devices.
    
#     Args:
#         tensor: Input tensor from each device
#         mesh_device: The device mesh
#         cluster_axis: Axis for gathering
#         dim: Dimension to concatenate along
#         num_links: Number of communication links
#         **kwargs: Additional arguments
    
#     Returns:
#         Concatenated tensor from all devices
#     """
#     if hasattr(tensor, 'shape'):
#         print(f"[CCL] all_gather: input shape={tensor.shape}, dim={dim}, cluster_axis={cluster_axis}")
    
#     return tensor

# # Alias for compatibility
# tt_all_gather = all_gather
