#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Training operations for backward pass and optimization
Added by Aditi Pandey for MNIST training support
"""

import numpy as np
from ttsim.ops.desc.registry import register_ops

def linear_backward_fwd(iTList, oTList, op, **kwargs):
    """
    Backward pass for Linear layer
    Inputs: [grad_output, input, weight]
    Outputs: [grad_input, grad_weight, grad_bias]
    """
    grad_output, input_tensor, weight = iTList[0], iTList[1], iTList[2]
    
    # Check if we're in shape inference mode
    if grad_output.data is None:
        # Shape inference mode
        oTList[0].data = None
        oTList[1].data = None
        oTList[2].data = None
    else:
        # Compute gradients
        # grad_input = grad_output @ weight
        grad_input = np.matmul(grad_output.data, weight.data)
        oTList[0].data = grad_input
        
        # grad_weight = grad_output.T @ input
        grad_weight = np.matmul(grad_output.data.T, input_tensor.data)
        oTList[1].data = grad_weight
        
        # grad_bias = sum(grad_output, axis=0)
        grad_bias = np.sum(grad_output.data, axis=0)
        oTList[2].data = grad_bias
    
    # Performance metrics
    batch_size = grad_output.shape[0]
    in_features = weight.shape[1]
    out_features = weight.shape[0]
    return {
        "instrs": {
            "mac": batch_size * in_features * out_features * 2,  # Two matmuls
            "add": batch_size * out_features  # For bias gradient
        }
    }

def conv2d_backward_fwd(iTList, oTList, op, **kwargs):
    """
    Backward pass for Conv2d layer
    Inputs: [grad_output, input, weight]
    Outputs: [grad_input, grad_weight]
    """
    # Simplified implementation
    oTList[0].data = None
    oTList[1].data = None
    
    # Approximate performance metrics
    grad_output = iTList[0]
    kernel_size = op.attrs.get('kernel_size', 3)
    
    instrs = {
        "mac": grad_output.nelems() * kernel_size * kernel_size * iTList[1].shape[1]
    }
    
    return instrs

def relu_backward_fwd(iTList, oTList, op, **kwargs):
    """
    Backward pass for ReLU
    Inputs: [grad_output, input]
    Output: [grad_input]
    """
    grad_output, input_tensor = iTList[0], iTList[1]
    
    if grad_output.data is None:
        oTList[0].data = None
    else:
        # grad_input = grad_output * (input > 0)
        mask = input_tensor.data > 0
        oTList[0].data = grad_output.data * mask
    
    instrs = {
        "compare": grad_output.nelems(),
        "mul": grad_output.nelems()
    }
    
    return instrs

def sgd_update_fwd(iTList, oTList, op, **kwargs):
    """
    SGD optimizer update
    Inputs: [weight, gradient]
    Output: [updated_weight]
    """
    weight, gradient = iTList[0], iTList[1]
    learning_rate = iTList[2] if len(iTList) > 2 else op.attrs.get('learning_rate', 0.01)
    
    if weight.data is None:
        oTList[0].data = None
    else:
        lr_value = learning_rate.data[0] if hasattr(learning_rate, 'data') else learning_rate
        # weight_new = weight - learning_rate * gradient
        oTList[0].data = weight.data - learning_rate * gradient.data
    
    instrs = {
        "mul": weight.nelems(),  # Scale gradient
        "sub": weight.nelems()   # Update weight
    }
    
    return instrs

# Shape inference functions

def linear_backward_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for LinearBackward"""
    grad_output, input_tensor, weight = iTList[0], iTList[1], iTList[2]
    
    # Set output shapes
    oTList[0].shape = list(input_tensor.shape)
    oTList[1].shape = list(weight.shape)
    oTList[2].shape = [weight.shape[0]]
    
    # Set dtype for output tensors
    oTList[0].dtype = grad_output.dtype
    oTList[1].dtype = grad_output.dtype
    oTList[2].dtype = grad_output.dtype
    
    batch_size = grad_output.shape[0]
    in_features = weight.shape[1]
    out_features = weight.shape[0]
    
    # Calculate element counts and memory usage
    in_elems = grad_output.nelems() + input_tensor.nelems() + weight.nelems()
    out_elems = oTList[0].nelems() + oTList[1].nelems() + oTList[2].nelems()
    in_bytes = grad_output.nbytes() + input_tensor.nbytes() + weight.nbytes()
    out_bytes = oTList[0].nbytes() + oTList[1].nbytes() + oTList[2].nbytes()
    
    op.perf_stats = {
        "instrs": {
            "mac": batch_size * in_features * out_features * 2,
            "add": batch_size * out_features
        },
        "inElems": in_elems,
        "outElems": out_elems,
        "inBytes": in_bytes,
        "outBytes": out_bytes
    }


def conv2d_backward_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for Conv2dBackward"""
    grad_output, input_tensor, weight = iTList[0], iTList[1], iTList[2]
    
    # Set output shapes
    oTList[0].shape = list(input_tensor.shape)
    oTList[1].shape = list(weight.shape)
    
    # Set dtype for output tensors
    oTList[0].dtype = grad_output.dtype
    oTList[1].dtype = grad_output.dtype
    
    kernel_size = weight.shape[2] * weight.shape[3]
    
    # Calculate element counts and memory usage
    in_elems = grad_output.nelems() + input_tensor.nelems() + weight.nelems()
    out_elems = oTList[0].nelems() + oTList[1].nelems()
    in_bytes = grad_output.nbytes() + input_tensor.nbytes() + weight.nbytes()
    out_bytes = oTList[0].nbytes() + oTList[1].nbytes()
    
    op.perf_stats = {
        "instrs": {
            "mac": grad_output.nelems() * kernel_size * input_tensor.shape[1]
        },
        "inElems": in_elems,
        "outElems": out_elems,
        "inBytes": in_bytes,
        "outBytes": out_bytes
    }

def relu_backward_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for ReluBackward"""
    oTList[0].shape = list(iTList[0].shape)
    
    # Set dtype
    oTList[0].dtype = iTList[0].dtype
    
    # Calculate element counts and memory usage
    in_elems = sum(t.nelems() for t in iTList)
    out_elems = oTList[0].nelems()
    in_bytes = sum(t.nbytes() for t in iTList)
    out_bytes = oTList[0].nbytes()
    
    op.perf_stats = {
        "instrs": {
            "cmp": iTList[0].nelems(),
            "mul": iTList[0].nelems()
        },
        "inElems": in_elems,
        "outElems": out_elems,
        "inBytes": in_bytes,
        "outBytes": out_bytes
    }
def sgd_update_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for SGDUpdate"""
    oTList[0].shape = list(iTList[0].shape)
    
    # Set dtype
    oTList[0].dtype = iTList[0].dtype
    
    # Calculate element counts and memory usage
    in_elems = sum(t.nelems() for t in iTList)
    out_elems = oTList[0].nelems()
    in_bytes = sum(t.nbytes() for t in iTList)
    out_bytes = oTList[0].nbytes()
    
    op.perf_stats = {
        "instrs": {
            "mul": iTList[0].nelems(),
            "sub": iTList[0].nelems()
        },
        "inElems": in_elems,
        "outElems": out_elems,
        "inBytes": in_bytes,
        "outBytes": out_bytes
    }

# Operation table for registration
_optbl = [
    # name, shape_inf_func, n_inputs, n_outputs, fwd_fn, has_fwd_in_mbytes, has_fwd_out_mbytes, has_bwd_in_mbytes, has_bwd_out_mbytes
    ["LinearBackward", "ARITY_3->3", "ttsim.training", "COMMON", 1, 1, 3, 3, 3, 3, 
     linear_backward_sinf, False, True, True, True, False],
    ["Conv2dBackward", "ARITY_3->2", "ttsim.training", "COMMON", 1, 1, 3, 3, 2, 2, 
     conv2d_backward_sinf, False, True, True, True, False],
    ["ReluBackward", "ARITY_2->1", "ttsim.training", "COMMON", 1, 1, 2, 2, 1, 1, 
     relu_backward_sinf, False, True, True, True, False],
    ["MaxPool2dBackward", "ARITY_3->1", "ttsim.training", "COMMON", 1, 1, 3, 3, 1, 1, 
     relu_backward_sinf, False, True, True, True, False],
    ["SoftmaxCrossEntropyBackward", "ARITY_2->1", "ttsim.training", "COMMON", 1, 1, 2, 2, 1, 1, 
     relu_backward_sinf, False, True, True, True, False],
    ["SGDUpdate", "ARITY_3->1", "ttsim.training", "COMMON", 1, 1, 3, 3, 1, 1,
     sgd_update_sinf, False, True, True, True, False],
]

def register_training_ops():
    """Register all training operations"""
    register_ops('training', _optbl)
