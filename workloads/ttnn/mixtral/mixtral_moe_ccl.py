#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from ttsim.front.functional.ccl import all_reduce

class TtMoeLayer():
    def __init__(self, mesh_device, state_dict, experts, args, layer_num: int, dtype, tt_ccl):
        super().__init__()
        self.mesh_device = mesh_device
        self.experts = experts
        self.args = args
        self.dtype = dtype
        self.tile_size = args.tile_size
        assert self.tile_size == 32, "tile size must be 32"
        self.num_devices = self.args.num_devices
        assert self.num_devices == 8, "num devices must be 8 for Mixtral MoE"
        self.tt_ccl = tt_ccl
        
        # Initialize tensors
        self.gates_H8 = ttnn._rand(shape=[1, 1, 4096, 64], device=mesh_device, dtype=ttnn.bfloat16)
        self.top8_mask_11B_64 = ttnn.full(shape=[1, 1, 1, 64], fill_value=1.0, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.top2_mask_11BB = ttnn.full(shape=[1, 1, 1, 32], fill_value=1.0, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.reduce_mask = ttnn.zeros(shape=[1, 1, self.tile_size, self.tile_size * 8], device=mesh_device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    def forward(self, inputs, mode):
        """
        WITH CCL Version - Includes all-reduce operation
        """
        input_i_1SBH = inputs
        expert_i_HH = self.experts
        
        # Get logits for the experts
        gate_logits_1SB8 = ttnn.matmul(
            input_i_1SBH,
            self.gates_H8,
            memory_config=None,
            compute_kernel_config=None,
            core_grid=None,
            dtype=ttnn.bfloat16,
        )
        
        # Get weights for top-2 experts
        gate_logits_1SB8 = ttnn.add(gate_logits_1SB8, self.top8_mask_11B_64)
        
        k_val = 32
        k_tensor = ttnn.full(shape=[1], fill_value=k_val, device=self.mesh_device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT)
        
        # Simplified gating (avoiding unsupported ops like moe, topk)
        weights_1SB1 = ttnn.softmax(gate_logits_1SB8, dim=-1)
        weights_1SB1 = ttnn.sum(weights_1SB1, dim=3, keepdim=True)
        weights_1SB1 = ttnn.reshape(weights_1SB1, [1, 1, 32, 1]) 
        # MLP and masking
        weights = expert_i_HH(input_i_1SBH, mode=mode)
        results_11BH = ttnn.multiply(weights, weights_1SB1)
        original_shape = results_11BH.shape
        seq_len = results_11BH.shape[-2]
        
        if seq_len >= 2048 and mode == "decode":
            results_11BH = ttnn.reshape(results_11BH, [1, 1, seq_len, self.args.dim])
        
        # MoE CCL - All-reduce operation to sum expert outputs across devices
        output_1SBH = all_reduce(
            results_11BH,
            self.mesh_device,
            cluster_axis=0,
            dim=3,
            sharded=(mode == "decode"),
            dtype=self.dtype,
            tt_ccl=self.tt_ccl
        )

        return output_1SBH
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def run_mixtral_ccl(wln, device, gcfg):
    # 1. Setup args
    class Args:
        def __init__(self):
            self.tile_size = 32
            self.num_devices = 8
            self.dim = 4096
            self.num_experts = 8
    
    args = Args()
    experts = lambda x, mode: x  # Dummy experts
    
    # 2. Initialize layer
    layer = TtMoeLayer(
        mesh_device=device,
        state_dict={},
        experts=experts,
        args=args,
        layer_num=gcfg.get('layers', 0),
        dtype=ttnn.bfloat16,
        tt_ccl=True  # CCL enabled
    )
    
    # 3. Create input
    inputs = ttnn._rand(shape=[1, 1, 32, 4096], device=device, dtype=ttnn.bfloat16)
    
    # 4. Forward pass (triggers CCL)
    return layer.forward(inputs, mode="decode")
