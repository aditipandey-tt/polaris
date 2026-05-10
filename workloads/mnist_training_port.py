#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import numpy as np
from loguru import logger

class MNISTTraining(SimNN.Module):
    """MNIST Training workload with forward and backward passes"""
    
    def __init__(self, name, cfg):  # Fixed: was **init**
        super().__init__()
        self.name = name
        self.bs = cfg['bs']
        self.lr = cfg.get('learning_rate', 0.01)
        # Layer dimensions
        self.conv1_out_channels = 32
        self.conv2_out_channels = 64
        self.fc1_out = 128
        self.fc2_out = 10
        
        # Forward pass layers
        self.conv1 = F.Conv2d(f"{name}.conv1", 1, self.conv1_out_channels, kernel_size=3)
        self.relu1 = F.Relu(f"{name}.relu1")
        self.conv2 = F.Conv2d(f"{name}.conv2", self.conv1_out_channels, self.conv2_out_channels, kernel_size=3)
        self.relu2 = F.Relu(f"{name}.relu2")
        self.pool = F.MaxPool2d(f"{name}.pool", 2)
        
        # Calculate flattened dimension after conv layers
        # Input: 28x28, after conv1: 26x26, after conv2: 24x24, after pool: 12x12
        self.flatten_dim = self.conv2_out_channels * 12 * 12  # Fixed: was _12_ 12
        
        self.reshape = F.ReshapeFixed(f"{name}.reshape", [self.bs, self.flatten_dim])
        self.fc1 = F.Linear(f"{name}.fc1", self.flatten_dim, self.fc1_out)
        self.relu3 = F.Relu(f"{name}.relu3")
        self.fc2 = F.Linear(f"{name}.fc2", self.fc1_out, self.fc2_out)
        
        # Backward pass operations (using the new ops we added!)
        self.fc2_backward = F.LinearBackward(f"{name}.fc2_backward")
        self.fc1_backward = F.LinearBackward(f"{name}.fc1_backward")
        self.relu3_backward = F.ReluBackward(f"{name}.relu3_backward")
        self.relu2_backward = F.ReluBackward(f"{name}.relu2_backward")
        self.relu1_backward = F.ReluBackward(f"{name}.relu1_backward")
        self.pool_backward = F.MaxPool2dBackward(f"{name}.pool_backward")
        self.conv2_backward = F.Conv2dBackward(f"{name}.conv2_backward", kernel_size=3)
        self.conv1_backward = F.Conv2dBackward(f"{name}.conv1_backward", kernel_size=3)
        
        # Optimizer operations
        self.sgd_fc2 = F.SGDUpdate(f"{name}.sgd_fc2", learning_rate=self.lr)
        self.sgd_fc1 = F.SGDUpdate(f"{name}.sgd_fc1", learning_rate=self.lr)
        self.sgd_conv2 = F.SGDUpdate(f"{name}.sgd_conv2", learning_rate=self.lr)
        self.sgd_conv1 = F.SGDUpdate(f"{name}.sgd_conv1", learning_rate=self.lr)
        
        self.loss_diff = F.Sub(f"{name}.loss_diff")  # Create loss operation as instance variable    
        self.grad_pool_reshape = F.ReshapeFixed(f"{name}.grad_pool_reshape", 
                                        [self.bs, self.conv2_out_channels, 12, 12])

    
        super().link_op2module()
        
    def create_input_tensors(self):
        """Create input tensors for forward and backward passes"""
        self.input_tensors = {
            'x': F._from_shape('x', [self.bs, 1, 28, 28], is_param=False, np_dtype=np.float32),
            'labels': F._from_shape('labels', [self.bs, self.fc2_out], is_param=False, np_dtype=np.float32),
        }
        
        # Create weight tensors (needed for backward pass)
    #    self.weights = {
     #       'conv1_w': F._from_shape('conv1_w', [self.conv1_out_channels, 1, 3, 3], is_param=True, np_dtype=np.float32),
     #       'conv2_w': F._from_shape('conv2_w', [self.conv2_out_channels, self.conv1_out_channels, 3, 3], is_param=True, np_dtype=np.float32),
      #      'fc1_w': F._from_shape('fc1_w', [self.flatten_dim, self.fc1_out], is_param=True, np_dtype=np.float32),
       #     'fc2_w': F._from_shape('fc2_w', [self.fc1_out, self.fc2_out], is_param=True, np_dtype=np.float32),
       # }
        
        logger.debug('Input tensors created: x={}, labels={}',
                    self.input_tensors['x'].shape, self.input_tensors['labels'].shape)
        return
        
    def __call__(self):  # Fixed: was **call**
        """Complete training step: forward + backward + optimizer"""
        x = self.input_tensors['x']
        labels = self.input_tensors['labels']
        
        # ========== FORWARD PASS ==========
        logger.debug("Executing forward pass...")
        
        # Conv block 1
        conv1_out = self.conv1(x)
        relu1_out = self.relu1(conv1_out)
        
        # Conv block 2
        conv2_out = self.conv2(relu1_out)
        relu2_out = self.relu2(conv2_out)
        pool_out = self.pool(relu2_out)
        
        # Flatten and FC layers
        flatten_out = self.reshape(pool_out)
        fc1_out = self.fc1(flatten_out)
        relu3_out = self.relu3(fc1_out)
        output = self.fc2(relu3_out)
        
        # Loss computation (simplified - just difference for now)
        loss_grad = self.loss_diff(output, labels)
                
        # ========== BACKWARD PASS ==========
        logger.debug("Executing backward pass...")
        
        # FC2 backward
        grad_fc2_in, grad_fc2_w, grad_fc2_b = self.fc2_backward(
            loss_grad, relu3_out, self._tensors[f'{self.name}.fc2.param']
        )        
        # ReLU3 backward
        grad_relu3 = self.relu3_backward(grad_fc2_in, fc1_out)
        
        # FC1 backward
        grad_fc1_in, grad_fc1_w, grad_fc1_b = self.fc1_backward(
            grad_relu3, flatten_out, self._tensors[f'{self.name}.fc1.param']
        )

        
        # Reshape gradient back for conv layers
        #grad_pool = F.ReshapeFixed(f"{self.name}.grad_pool_reshape",
         #                         [self.bs, self.conv2_out_channels, 12, 12])(grad_fc1_in)
        grad_pool = self.grad_pool_reshape(grad_fc1_in)
        # Pool backward
        grad_relu2 = self.pool_backward(grad_pool, relu2_out, pool_out)
        
        # ReLU2 backward
        grad_conv2 = self.relu2_backward(grad_relu2, conv2_out)
        
        # Conv2 backward
        grad_relu1, grad_conv2_w = self.conv2_backward(
            grad_conv2, relu1_out, self._tensors[f'{self.name}.conv2.param']
        )
        
        # ReLU1 backward
        grad_conv1 = self.relu1_backward(grad_relu1, conv1_out)
        
        # Conv1 backward
        grad_input, grad_conv1_w = self.conv1_backward(
            grad_conv1, x, self._tensors[f'{self.name}.conv1.param']
        )
        
        # ========== OPTIMIZER UPDATE ==========
        logger.debug("Executing optimizer updates...")
        
        # Update weights using SGD
        new_fc2_w = self.sgd_fc2(self._tensors[f'{self.name}.fc2.param'], grad_fc2_w)
        new_fc1_w = self.sgd_fc1(self._tensors[f'{self.name}.fc1.param'], grad_fc1_w)
        new_conv2_w = self.sgd_conv2(self._tensors[f'{self.name}.conv2.param'], grad_conv2_w)
        new_conv1_w = self.sgd_conv1(self._tensors[f'{self.name}.conv1.param'], grad_conv1_w)        
        logger.info("Training step completed - Forward + Backward + Optimizer Update")
        
        return output

    def analytical_param_count(self):
        """Calculate total number of parameters in the model"""
        total_params = 0
        # Count parameters from all layers that have '.param' tensors
        for name, tensor in self._tensors.items():
            if '.param' in name and hasattr(tensor, 'shape'):
                # Calculate number of elements in this parameter tensor
                param_count = 1
                for dim in tensor.shape:
                    param_count *= dim
                total_params += param_count
        return total_params
        
    def get_forward_graph(self):
        """Get the complete computation graph"""
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

# Test configuration
if __name__ == "__main__":  # Fixed: was **name**
    cfg = {
        'bs': 64,
        'learning_rate': 0.01,
    }
    
    model = MNISTTraining('mnist_training', cfg)
    model.create_input_tensors()
    
    # Run training step
    output = model()
    
    print("\n" + "="*60)
    print("MNIST TRAINING WITH FULL BACKWARD SUPPORT")
    print("="*60)
    print("\n✅ Successfully extended Polaris with training operations!")
    print("\nWhat we achieved:")
    print("  1. Added backward operations to ttsim/front/functional/op.py")
    print("  2. Created complete training DAG (Forward + Backward + Optimizer)")
    print("  3. Model architecture matches PyTorch MNIST example")
    print("\nOperations used:")
    print("  - Forward: Conv2d, ReLU, MaxPool2d, Linear")
    print("  - Backward: Conv2dBackward, ReluBackward, LinearBackward, etc.")
    print("  - Optimizer: SGDUpdate")
    print("\nOutput shape:", output.shape if hasattr(output, 'shape') else output)
    print("="*60)
