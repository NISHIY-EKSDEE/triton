from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


triton_fused_argmax_argmin_0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import reduction
from torch._inductor.utils import instance_descriptor

@reduction(size_hints=[1, 524288],
              reduction_hint=ReductionHint.INNER,
              filename=__file__,
              meta={'signature': {0: '*fp32', 1: '*i64', 2: '*i64', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16
=(0, 1, 2, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    _tmp1_index = tl.zeros([XBLOCK, RBLOCK], tl.int64)
    _tmp2 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("inf")
    _tmp2_index = tl.zeros([XBLOCK, RBLOCK], tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last')
        _tmp1_index = tl.where(xmask & rmask & (_tmp1 < tmp0),  rindex, _tmp1_index)
        _tmp1 = tl.where(xmask & rmask & (_tmp1 < tmp0), tmp0, _tmp1)
        _tmp2_index = tl.where(xmask & rmask & (_tmp2 > tmp0),  rindex, _tmp2_index)
        _tmp2 = tl.where(xmask & rmask & (_tmp2 > tmp0), tmp0, _tmp2)
    _tmp1_index_reduce = tl.argmax(_tmp1, 1)[:, None].to(tl.int32)
    _tmp1_index_mask = tl.arange(0, RBLOCK)[None, :] == _tmp1_index_reduce
    tmp1 = tl.sum(tl.where(_tmp1_index_mask, _tmp1_index, 0), 1)[:, None]
    tl.store(out_ptr0 + 0 + tl.zeros([XBLOCK, 1], tl.int32), tmp1, None)
    _tmp2_index_reduce = tl.argmin(_tmp2, 1)[:, None].to(tl.int32)
    _tmp2_index_mask = tl.arange(0, RBLOCK)[None, :] == _tmp2_index_reduce
    tmp2 = tl.sum(tl.where(_tmp2_index_mask, _tmp2_index, 0), 1)[:, None]
    tl.store(out_ptr1 + 0 + tl.zeros([XBLOCK, 1], tl.int32), tmp2, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((), (), device='cuda', dtype=torch.int64)
        buf1 = empty_strided((), (), device='cuda', dtype=torch.int64)
        stream0 = get_cuda_stream(0)
        triton_fused_argmax_argmin_0.run(arg0_1, buf0, buf1, 1, 524288, grid=grid(1), stream=stream0)
        del arg0_1
        return (buf0, buf1, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 256, 256), (65536, 256, 1), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([arg0_1]))F