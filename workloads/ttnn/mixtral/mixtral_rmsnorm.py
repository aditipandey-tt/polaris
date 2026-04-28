#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn

TILE = 32
SHARD_HEIGHT = TILE

class RMSNorm():
    def __init__(self, device=None,
        dim=None,
        args=None,
        eps=0.00001,
        state_dict=None,
        weight_cache_path=None,
        state_dict_prefix="",
        weight_dtype=ttnn.bfloat16,
        weight_key="ffn_norm",
        is_distributed=False,
        add_unit_offset=False,
        sharded_program_config=None,
        sharded_output_config=None,
        ccl_topology=None):
        
        self.device = device
        self.dim = dim
        self.args = args
        self.eps = eps
        self.state_dict = state_dict
        self.weight_cache_path = weight_cache_path
        self.state_dict_prefix = state_dict_prefix
        self.weight_dtype = weight_dtype
        self.weight_key = weight_key
        self.is_distributed = is_distributed
        self.add_unit_offset = add_unit_offset
        self.sharded_program_config = sharded_program_config
        self.sharded_output_config = sharded_output_config
        self.ccl_topology = ccl_topology
        self.compute_kernel_config_hifi2 = ttnn.MathFidelity.HiFi2
        self.num_devices = getattr(args, 'num_devices', 8) if args else 8
        self.weight = ttnn._rand(shape=(1, 1, 32, self.dim), device=device, dtype=self.weight_dtype)
        self.bias = None
    
    def __call__(self, x, mode="decode"):
        # Get input dimensions and convert to list
        x_shape = list(x.shape) if hasattr(x.shape, '__iter__') else x.shape
        input_dim = x_shape[-1]
        
        # Create a dummy weight for layer_norm (all ones)
        dummy_weight = ttnn.ones(
            shape=(1, 1, x_shape[2], x_shape[3]),
            device=self.device,
            dtype=self.weight_dtype,
            layout=ttnn.TILE_LAYOUT
        )
        
        # Compute RMS using layer_norm with dummy weight
        rms = ttnn.layer_norm(x, weight=dummy_weight, epsilon=self.eps, axis=-1)
        
        # Apply all_reduce if distributed
        if self.is_distributed or True:
            from ttsim.front.functional.ccl import all_reduce
            rms = all_reduce(rms, mesh_device=self.device, dim=3)
            rms = ttnn.reshape(rms, tuple(x_shape))
        
        # Normalize
        normalized = ttnn.div(x, rms)
        
        # Handle actual weight tensor for scaling
        if input_dim != self.dim:
            # Create weight tensor matching input dimension
            weight_tensor = ttnn.ones(
                shape=(x_shape[0], x_shape[1], x_shape[2], input_dim),
                device=self.device,
                dtype=self.weight_dtype,
                layout=ttnn.TILE_LAYOUT
            )
        else:
            # Use original weight, creating new tensor to match shape if needed
            weight_shape = list(self.weight.shape) if hasattr(self.weight.shape, '__iter__') else self.weight.shape
            
            # Create weight tensor matching input shape
            if weight_shape[2] != x_shape[2] or weight_shape[3] != x_shape[3]:
                weight_tensor = ttnn.ones(
                    shape=tuple(x_shape),
                    device=self.device,
                    dtype=self.weight_dtype,
                    layout=ttnn.TILE_LAYOUT
                )
            else:
                weight_tensor = self.weight
        
        # Apply weight scaling
        normalized = ttnn.multiply(normalized, weight_tensor)
        
        # Apply bias if exists
        if self.bias is not None:
            normalized = ttnn.add(normalized, self.bias)
        
        return normalized


def run_rms_ccl(wln, device, gcfg):
    class Args:
        def __init__(self):
            self.dim = 4096
            self.num_experts = 8
            self.num_devices = 8
    
    args = Args()
    
    layer = RMSNorm(
        device=device,
        dim=args.dim,
        args=args,
        is_distributed=True
    )
    
    inputs = ttnn._rand(shape=[1, 1, 32, 512], device=device, dtype=ttnn.bfloat16)
    return layer(inputs, mode="decode")
