#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from loguru import logger
import ttsim.front.ttnn as ttnn
from workloads.ttnn.mixtral.mixtral_mlp import TtMixtralMLP
# CRITICAL: Use the _ccl version of the layer
from workloads.ttnn.mixtral.mixtral_moe_ccl import TtMoeLayer
from workloads.ttnn.tt_transformers.model_config import ModelArgs

def test_mixtral_moe_inference(mesh_device, model_name, mode):
    iterations = 1
    dtype = ttnn.bfloat8_b
    model_args = ModelArgs(mesh_device, model_name=model_name)
    state_dict = model_args.load_state_dict()
    model_args.n_layers = 1
    layer_num = 0

    tt_model = TtMoeLayer(
        mesh_device=mesh_device,
        state_dict=state_dict,
        experts=TtMixtralMLP(
            mesh_device=mesh_device,
            state_dict=state_dict,
            args=model_args,
            layer_num=layer_num,
            dtypes={"w1": dtype, "w2": dtype, "w3": dtype},
        ),
        args=model_args,
        layer_num=layer_num,
        dtype=dtype,
        tt_ccl=None,
    )

    seqlen = 1
    batch = 32

    for i in range(iterations):
        logger.info(f"{mode} Generating token {i}")
        _input = ttnn._rand(shape=(seqlen, batch, model_args.dim), device=mesh_device, dtype=ttnn.bfloat16)

        logger.info(f"Starting TT Mixtral MOE")
        
        # Execute forward pass (this triggers your CCL logic)
        tt_out = tt_model(_input, mode)
        if hasattr(tt_out, 'perf_stats'):
            stats = tt_out.perf_stats  # <--- MAKE SURE THIS LINE IS HERE FIRST
            print("\n" + "="*60)
            print(f"POLARIS CCL PERFORMANCE REPORT - {mode.upper()}")
            print(f"Instruction Accounting: {stats.get('instrs')}")
            print(f"Network Transfer: {stats.get('network_transfer_kb', 0):.2f} KB") 
            print(f"Projected Wire Latency: {stats.get('latency_ms', 0):.6f} ms")
            print(f"Mesh Data Volume: {stats.get('mesh_traffic_bytes', 0) / 1024:.2f} KB")
            print(f"Mimicked Mesh Size: {stats.get('num_devices')} Devices")
            print("="*60 + "\n")
        logger.info(f"output shape is {tt_out.shape}")
        assert tt_out.shape == [1, seqlen, batch, model_args.dim // model_args.num_experts]
        logger.info(f"Passed! TT Mixtral MOE {mode}.")

if __name__ == "__main__":
    mesh_device = ttnn.open_device(device_id=0)
    test_mixtral_moe_inference(mesh_device=mesh_device, model_name="mixtral-8x7B", mode="prefill")
    test_mixtral_moe_inference(mesh_device=mesh_device, model_name="mixtral-8x7B", mode="decode")
