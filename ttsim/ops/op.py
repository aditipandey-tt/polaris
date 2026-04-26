#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Union

import numpy as np

import ttsim.utils.common as common

from .desc.registry import get_opdesc_registry
from .tensor import SimTensor


class SimOp:
    def __init__(self, cfg):
        self.name         = cfg['name']
        self.optype       = cfg['optype']
        self.attrs        = cfg.get('attrs', {})
        self.inList       = cfg.get('inList', [])
        self.outList      = cfg.get('outList', [])
        self.domain       = cfg.get('domain', "")
        self.docstr       = cfg.get('docstr', "")
        self.opclass_str  = 'None'
        # Sequence number assigned when operator is added to a WorkloadGraph (see WorkloadGraph.add_op).
        # This is used as a tie-breaker in lexicographical topological sort to ensure deterministic
        # operator ordering in CSV stats and other outputs, even when the graph has multiple valid orderings.
        self.seqno        = None
        #special counter for some workloads, e.g., Transformer Blocks
        # where we execute the op only once, but account for repeated
        # executions for the full workload
        self.repeat_count = 1

        #These fields are set via __call__ / get_perf_counts() when the op is executed
        # with input tensors dim/shape being well defined
        self.perf_stats: Union[dict, None]   = None

        #These fields are set via execution of op of a device...
        self.precision               = None
        self.removed_in_optimization = False
        self.fused_in_optimization   = False
        self.fused_with_op           = None
        self.uses_compute_pipe       = None
        self.compute_cycles          = None
        self.mem_rd_cycles           = None
        self.mem_wr_cycles           = None
        self.fused_op_cycles         = None
        self.exec_stats              = None
        self._kw_args_defaults       = {}

    def __str__(self):
        s  = f"SimOp({self.name}) optype={self.optype}, cls={self.opclass_str}, "
        s += f"prec={self.precision}, attrs={self.attrs}, domain={self.domain}, "
        s += f"rpt={self.repeat_count}, "
        s += f"removed={self.removed_in_optimization}, "
        s += f"fused={self.fused_in_optimization}, "
        s += f"fused_with_op={self.fused_with_op}, "
        s += f"uses_compute_pipe={self.uses_compute_pipe}, "
        s += f"inList={self.inList}, "
        s += f"outList={self.outList}"
        return s

    def check_known_args(self, args) -> None:
        common.check_known_args(str(type(self)), args=args,
                                default_args=self._kw_args_defaults)

    def get_effective_args(self, args: dict[str, Any]) -> dict[str, Any]:
        return common.get_kwargs_with_defaults(str(type(self)),
                                               args=args,
                                               default_args=self._kw_args_defaults)

    def get_perf_counts(self, inT, outT, **kwargs):
        if self.perf_stats is not None:
            return self.perf_stats

        opdesc = get_opdesc_registry().get_opdesc(self.optype)

        #check in/out counts
        in_range  = range(opdesc['min_input'], opdesc['max_input']+1)
        out_range = range(opdesc['min_output'], opdesc['max_output']+1)
        assert len(inT) in in_range,   f"#inputs for {self} operator should be in {in_range}, is {len(inT)}"
        assert len(outT) in out_range, f"#outputs for {self} operator should be in {out_range}, is {len(outT)}"

        #Do Shape Inference, Update perf_stats
        shape_inf_func = opdesc['shape_inf_func']
        shape_inf_func(inT, outT, self, **kwargs)

        return self.perf_stats

    def update_tensor_counts(self, inT, outT, **kwargs):
        in_param_count  = sum([x.nelems() for x in inT if x.is_param == True])
        in_act_count    = sum([x.nelems() for x in inT if x.is_param == False])
        out_act_count   = sum([x.nelems() for x in outT if x.is_param == False])
        out_param_count = sum([x.nelems() for x in outT if x.is_param == True])
        assert out_param_count == 0, "OP{self.name} has output param count > 0: {out_param_count}"
        if TYPE_CHECKING:
            assert self.perf_stats is not None
        self.perf_stats.update({
            'inParamCount': int(in_param_count),
            'inActCount'  : int(in_act_count),
            'outActCount' : int(out_act_count),
            })
        return

    def set_precision(self, prec):
        self.precision = prec

    def remove_in_optimization(self):
        self.removed_in_optimization = True

    def fuse_op(self, fused_with_op):
        self.fused_in_optimization = True
        self.fused_with_op         = fused_with_op

    def get_effective_precision(self, tensor: SimTensor) -> np.dtype:
        """
        Get the effective precision for a tensor based on the op's precision.
        If the op's precision is not set, return the tensor's dtype.
        """
        if self.precision is not None:
            return self.precision
        assert tensor.dtype is not None, f"Tensor {tensor.name} has no dtype set"
        return tensor.dtype

def get_tensor_broadcast_shape(shape1, shape2):
    """Determine broadcasted shape for element-wise operations"""
    s1 = shape1[::-1]
    s2 = shape2[::-1]
    max_len = max(len(s1), len(s2))
    s1_list = list(s1)
    s2_list = list(s2)
    s1_list.extend([1] * (max_len - len(s1_list)))
    s2_list.extend([1] * (max_len - len(s2_list)))

    result = []
    for d1, d2 in zip(s1_list, s2_list):
        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            raise ValueError(f"Shapes {shape1} and {shape2} not broadcast-compatible")
    return result[::-1]





def CCLOpHandle(name, optype, in_tensor, num_devices, latency_ms, dim=3):
    from .tensor import SimTensor
    import copy

    # 1. Generic Shape Detection
    out_shape = copy.deepcopy(in_tensor.shape)
    op_lower = optype.lower()
    
    # Bytes per element (Default 2 for bfloat16, generic handle)
    bpe = 2 
    if hasattr(in_tensor, 'dtype') and '32' in str(in_tensor.dtype): bpe = 4
    # Universal dimension handling - SINGLE CHECK
    if dim is None:
        dim = -1  # Use last dimension
    
    # Handle negative indexing (Python style)
    if dim < 0:
        dim = len(out_shape) + dim
    
    # Final validation - SINGLE CHECK
    if dim >= len(out_shape) or dim < 0:
        print(f"[WARNING CCL] Invalid dim {dim} for shape {out_shape}, using last dimension")
        dim = len(out_shape) - 1
    
    # 2. Universal Sharding Logic (Purane CCL logic ke hisaab se)
    if op_lower == 'all_gather':
        out_shape[dim] *= num_devices
    elif op_lower == 'reduce_scatter':
        # Agar tumhare workload ko sharded output chahiye (mixtral logic)
        if out_shape[dim] % num_devices == 0:
            out_shape[dim] //= num_devices
    elif op_lower == 'all_reduce':
        pass
    out_tensor = SimTensor({
        'name': f"{name}.out",
        'shape': out_shape,
        'dtype': in_tensor.dtype,
        'op_out': [name]
    })
    if hasattr(in_tensor, 'device'):
        out_tensor.device = in_tensor.device

    # 3. Perf Accounting (Rama's Stats)
    freq_mhz = 1000.0
    ideal_cycles = int(latency_ms * freq_mhz)
    ideal_cycles = int(latency_ms * freq_mhz)
    print(f"[CCLOpHandle] name={name}, latency_ms={latency_ms:.6f}, ideal_cycles={ideal_cycles}")
    print(f"[CCLOpHandle] mem_rd={ideal_cycles//2}, mem_wr={ideal_cycles//2}")


    out_tensor.perf_stats = {
        'op_type': optype,
        'inActCount': int(in_tensor.nelems()),
        'outActCount': int(out_tensor.nelems()),
        'inBytes': int(in_tensor.nelems() * bpe),
        'outBytes': int(out_tensor.nelems() * bpe),
        'ideal_cycles': ideal_cycles,
        'compute_cycles': 0,
        'mem_rd_cycles': ideal_cycles // 2,
        'mem_wr_cycles': ideal_cycles // 2,
        'instrs': {
            'add': int(in_tensor.nelems() * (num_devices - 1)) if 'reduce' in op_lower else 0,
            'load': int(in_tensor.nelems()), 
            'store': int(out_tensor.nelems())
        }
    }
    
    return out_tensor
 
