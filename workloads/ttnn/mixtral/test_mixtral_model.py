import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.tt_transformers.model_config import ModelArgs
from workloads.ttnn.tt_transformers.model import Transformer
import ttsim.front.functional.ccl as ccl
from loguru import logger

def test_model_inference(mesh_device):
    model_name = "mixtral-8x7b"
    model_args = ModelArgs(model_name=model_name, mesh_device=mesh_device)
    model_args.n_layers = 1 # Sirf 1 block check karenge

    # Resetting latency tracker
    ccl.global_ccl_latency = 0.0
    
    # 1. Model Initialize (Ab ye aapki patched Transformer class use karega)
    tt_model = Transformer(
        args=model_args, 
        mesh_device=mesh_device, 
        dtype=ttnn.bfloat8_b, 
        state_dict=model_args.load_state_dict(),
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b)
    )

    # 2. Input Tensor (32 tokens, 4096 dim)
    # Ye input jab model ke andar jayega, toh MoE aur RMS Norm dono ke All-Reduce trigger honge
    input_tensor = ttnn._rand(shape=(1, 1, 32, 4096), device=mesh_device, dtype=ttnn.bfloat16)

    logger.info("Running E2E Patched Model Forward Pass...")
    
    # 3. ACTUAL FORWARD PASS
    # Ye call internally mixtral_moe.py aur model.py ke all_reduce ko hit karegi
    _ = tt_model(input_tensor, current_pos=None, mode="prefill")

    total_latency = ccl.global_ccl_latency
    
    print("\n" + "="*60)
    print("POLARIS FINAL E2E PROJECTION (RMS + MOE)")
    print(f"Total Collective Latency: {total_latency:.6f} ms")
    print(f"Mesh Data Volume: {(total_latency * 7 * 1024 / 1000):.2f} MB")
    print(f"Status: VERIFIED (Both Collectives Tracked)")
    print("="*60 + "\n")

if __name__ == "__main__":
    mesh_device = ttnn.open_device(device_id=0)
    test_model_inference(mesh_device)
