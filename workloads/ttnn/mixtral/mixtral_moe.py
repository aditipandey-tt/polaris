#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from loguru import logger

class ExpertFFN:
    """Single Expert FFN: 4096 -> 14336 -> 4096"""
    def __init__(self, device, dim=4096, hidden_dim=14336, dtype=ttnn.bfloat16):
        self.w1 = ttnn._rand(shape=[1, 1, dim, hidden_dim], device=device, dtype=dtype)
        self.w2 = ttnn._rand(shape=[1, 1, hidden_dim, dim], device=device, dtype=dtype)
        self.w3 = ttnn._rand(shape=[1, 1, dim, hidden_dim], device=device, dtype=dtype)
        
    def forward(self, x):
        # Gate projection with w1
        h1 = ttnn.matmul(x, self.w1)  # [batch*seq, 1, 1, 4096] x [1, 1, 4096, 14336]
        
        # Since ttnn doesn't have silu, just use the value directly for simulation
        # In real implementation, this would be silu(h1)
        h1_activated = h1
        
        # Up projection with w3
        h3 = ttnn.matmul(x, self.w3)  # [batch*seq, 1, 1, 4096] x [1, 1, 4096, 14336]
        
        # Combine (element-wise multiply)
        h = ttnn.multiply(h1_activated, h3)
        
        # Down projection with w2
        out = ttnn.matmul(h, self.w2)  # [batch*seq, 1, 1, 14336] x [1, 1, 14336, 4096]
        
        return out

class TtMoeLayer():
    def __init__(self, mesh_device, state_dict, experts, args, layer_num: int, dtype, tt_ccl):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.dtype = dtype
        self.tile_size = args.tile_size
        assert self.tile_size == 32, "tile size must be 32"
        self.num_devices = self.args.num_devices
        assert self.num_devices == 8, "num devices must be 8 for Mixtral MoE"
        self.tt_ccl = tt_ccl
        
        # Store original experts parameter (not used in our implementation)
        self.original_experts = experts
        
        # MoE specific parameters
        self.num_experts = 8  # Mixtral has 8 experts
        self.top_k = 2  # Select top 2 experts
        
        # Gate weights: [4096, 8]
        self.gates = ttnn._rand(shape=[1, 1, args.dim, self.num_experts], 
                               device=mesh_device, dtype=ttnn.bfloat16)
        
        # Create 8 expert FFN networks
        self.experts = []
        for i in range(self.num_experts):
            expert = ExpertFFN(mesh_device, dim=args.dim, hidden_dim=14336, dtype=dtype)
            self.experts.append(expert)
            
        logger.info(f"[MoE] Initialized {self.num_experts} experts, each with FFN: 4096->14336->4096")
        
        # Legacy tensors from original implementation (kept for compatibility)
        self.gates_H8 = ttnn._rand(shape=[1, 1, 4096, 64], device=mesh_device, dtype=ttnn.bfloat16)
        self.top8_mask_11B_64 = ttnn.full(shape=[1, 1, 1, 64], fill_value=1.0, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.top2_mask_11BB = ttnn.full(shape=[1, 1, 1, 32], fill_value=1.0, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.reduce_mask = ttnn.zeros(shape=[1, 1, self.tile_size, self.tile_size * 8, 1], device=mesh_device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    def forward(self, inputs, mode):
        logger.info(f"[MoE] MoE layer forward called! mode={mode}, num_experts={self.num_experts}, top_k={self.top_k}")
        
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1] if len(inputs.shape) > 3 else 1
        
        # Reshape input to [batch*seq, 1, 1, dim]
        x = ttnn.reshape(inputs, (batch_size * seq_len, 1, 1, self.args.dim))
        
        # 1. Compute router/gate scores
        logger.debug(f"[MoE] Computing gate scores...")
        gate_logits = ttnn.matmul(x, self.gates)  # [batch*seq, 1, 1, 8]
        
        # 2. Select top-k experts
        gate_scores = ttnn.softmax(gate_logits, dim=-1)
        
        # For simulation, we'll just use experts 0 and 1 with equal weights
        selected_experts = [0, 1]
        
        # 3. Run selected experts
        logger.debug(f"[MoE] Running {len(selected_experts)} experts...")
        expert_outputs = []
        
        for i, expert_idx in enumerate(selected_experts):
            logger.debug(f"[MoE] Running expert {expert_idx}...")
            expert = self.experts[expert_idx]
            
            # Run expert FFN - this will create MatMul operations
            expert_out = expert.forward(x)
            
            # Weight by expert score (simplified - just multiply by 0.5 for equal weighting)
            weight_scalar = 0.5  # Equal weight for each of the 2 experts
            weight = ttnn.full(shape=[batch_size * seq_len, 1, 1, 1],
                              fill_value=weight_scalar, 
                              device=self.mesh_device,
                              dtype=self.dtype, 
                              layout=ttnn.TILE_LAYOUT)
            
            weighted_out = ttnn.multiply(expert_out, weight)
            expert_outputs.append(weighted_out)
        
        # 4. Combine expert outputs
        logger.debug(f"[MoE] Combining expert outputs...")
        combined = expert_outputs[0]
        for expert_out in expert_outputs[1:]:
            combined = ttnn.add(combined, expert_out)
        
        # 5. All-reduce across devices (for model parallelism)
        if self.num_devices > 1:
            logger.debug(f"[MoE CCL] Performing all_reduce across {self.num_devices} devices...")
            combined = ttnn.all_reduce(
                combined,
                mesh_device=self.mesh_device,
                cluster_axis=0,
                dim=3
            )
            logger.debug(f"[MoE CCL] After all_reduce - combined shape: {combined.shape}")
        
        # 6. Reshape back to original shape
        output = ttnn.reshape(combined, (batch_size, seq_len, 1, self.args.dim))
        
        logger.debug(f"[MoE] Final output shape: {output.shape}")
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
