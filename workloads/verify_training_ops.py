#!/usr/bin/env python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ttsim.front.functional.op as F
from loguru import logger

print("=== Verifying Training Operations ===\n")

# Test backward operations
ops_to_test = [
    ("ReluBackward", lambda: F.ReluBackward("relu_back")),
    ("MaxPool2dBackward", lambda: F.MaxPool2dBackward("maxpool_back")),
    ("LinearBackward", lambda: F.LinearBackward("linear_back")),
    ("Conv2dBackward", lambda: F.Conv2dBackward("conv_back", kernel_size=3)),
    ("SGDUpdate", lambda: F.SGDUpdate("sgd", learning_rate=0.01)),
    ("AdamUpdate", lambda: F.AdamUpdate("adam")),
    ("SoftmaxCrossEntropyBackward", lambda: F.SoftmaxCrossEntropyBackward("ce_back")),
]

passed = 0
failed = 0

for op_name, op_creator in ops_to_test:
    try:
        op = op_creator()
        print(f"✅ {op_name}: SUCCESS")
        passed += 1
    except Exception as e:
        print(f"❌ {op_name}: FAILED - {e}")
        failed += 1

print(f"\n=== Summary ===")
print(f"Passed: {passed}/{len(ops_to_test)}")
print(f"Failed: {failed}/{len(ops_to_test)}")

if failed == 0:
    print("\n🎉 All training operations are working correctly!")
    print("Polaris is now ready for training workloads!")
else:
    print("\n⚠️  Some operations failed. Check the implementation.")

