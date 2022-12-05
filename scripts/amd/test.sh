# clear
rm -rf triton_rocm_kernels

# export TRITON_LIBHIP=/opt/rocm/lib/libamdhip64.so

# export AMD_OCL_WAIT_COMMAND=1
# export AMD_LOG_LEVEL=3
# export HIP_LAUNCH_BLOCKING=1

# remove cache to avoid segfaults
# TODO: inform triton dev the cache cause segfault
# rm -rf /tmp/triton
sh scripts/amd/clean.sh

# test simple hip kernel
# sh scripts/amd/test_hip_kernel.sh

# pytest python/test
# pytest python/test/test_blocksparse.py

# pytest --verbose python/test/test_conv.py
# pytest --verbose python/test/test_blocksparse.py::test_matmul[sdd-False-False-16-float16]
# pytest --verbose python/test/test_blocksparse.py::test_attention_fwd_bwd
# python python/test/test_conv.py

# gdb -ex "set breakpoint pending on" \
#     -ex 'break add_passes_to_emit_bin' \
#     --args python python/test/test_add.py

# python python/test/test_empty.py
# -ex 'ignore 1 472' \

# pytest -rfs --verbose python/tests/test_core.py | tee /dockerx/triton/test_core.log
# pytest -rfs --verbose python/tests/test_core.py::test_empty_kernel[float32] 2>&1 | tee /dockerx/triton/test_empty_kernel.log
# pytest -rfs --verbose "python/tests/test_core_amd.py::test_bin_op" 2>&1 | tee /dockerx/triton/test_bin_op.log
# pytest -rfs --verbose python/tests/test_core_amd.py::test_bin_op[float32-float32-/] 2>&1 | tee /dockerx/triton/test_bin_op.log
# pytest -rfs --verbose python/tests/test_core.py::test_load_cache_modifier 2>&1 | tee /dockerx/triton/test_load.log

# python3 python/tutorials/01-vector-add.py
# pytest -rfs --verbose "python/tests/test_vecadd.py"
pytest -rfs --verbose "python/tests/test_vecadd.py::test_vecadd_fcmp_no_scf_masked[1-2-shape0]"
# pytest -rfs --verbose "python/tests/test_vecadd.py::test_vecadd_scf_mask"
# pytest -rfs --verbose "python/tests/test_vecadd.py::test_vecadd_no_scf_masked"
# pytest -rfs --verbose "python/tests/test_vecadd.py::test_vecadd_no_scf[1-256-shape2]"
# pytest --capture=tee-sys -rfs --verbose "python/tests/test_vecadd.py::test_vecadd_no_scf"
# pytest -rfs --verbose "python/tests/test_vecadd.py::test_vecadd_no_scf[1-32-shape0]"
# pytest -rfs --verbose "python/tests/test_vecadd.py::test_vecadd_no_scf[1-64-shape1]"
# pytest -rfs --verbose "python/tests/test_vecadd.py::test_vecadd_fcmp_no_scf_masked[1-128-shape0]"

# pytest -rfs --verbose "python/tests/test_vecadd.py::test_vecadd_scf_mask[shape0-2-128-1]"
# pytest -rfs --verbose "python/tests/test_vecadd.py::test_vecadd_no_scf[4-256-shape0]"
# pytest -rfs --verbose "python/tests/test_vecadd.py::test_vecadd_fcmp_no_scf_masked[4-256-shape3]"

# python3 -m triton.tools.aot test/Target/tritongpu_to_ptx.mlir --target=amdgcn --gfx=906
# pytest -rfs --verbose python/test/unit/language/test_core.py::test_masked_load[float32-128-2] \
# 	2>&1 | tee /dockerx/triton/test_masked_load.log

# pytest --verbose python/test/unit/language/test_core.py::test_empty_kernel[float32] 2>&1 | tee /dockerx/triton/test_empty_kernel.log
# pytest --verbose python/test/unit/language/test_core.py::test_bin_op[int32-uint32-+] 2>&1 | tee /dockerx/triton/test_bin_op.log
# pytest --verbose python/test/unit/language/test_core.py::test_atomic_rmw 2>&1 | tee /dockerx/triton/test_atomic_rmw.log
# pytest --verbose python/test/unit/language/test_core.py::test_atomic_rmw[add-float16-all_neg] 2>&1 | tee /dockerx/triton/test_atomic_rmw.log
#  pytest --verbose "python/test/unit/language/test_core.py::test_reduce1d" 2>&1 | tee /dockerx/triton/test_reduce1d.log
# pytest --verbose "python/test/unit/language/test_core.py::test_cast[float32-float32-False]" 2>&1 | tee /dockerx/triton/test_cast.log
# pytest --verbose "python/test/unit/language/test_core.py::test_load_cache_modifier" 2>&1 | tee /dockerx/triton/test_vectorization.log

# mismatch
# pytest --verbose "python/test/unit/language/test_core.py::test_bin_op[int8-float16-%]"" 2>&1 | tee /dockerx/triton/test_bin_op.log
# pytest --verbose "python/test/unit/language/test_core.py::test_bitwise_op[int32-int32-&1]" 2>&1 | tee /dockerx/triton/test_bitwise_op.log
# pytest --verbose "python/test/unit/language/test_core.py::test_shift_op[int32-int32->>]" 2>&1 | tee /dockerx/triton/test_shift_op.log
# pytest --verbose "python/test/unit/language/test_core.py::test_compare_op[float32-float32->-nan-real]" 2>&1 | tee /dockerx/triton/test_compare_op.log
# pytest --verbose "python/test/unit/language/test_core.py::test_cast[float32-float32-False]" 2>&1 | tee /dockerx/triton/test_compare_op.log

# pytest --capture=tee-sys --verbose  python/test/regression/test_performance.py | tee /dockerx/triton/test_performance.log
# pytest --capture=tee-sys --verbose  python/test/regression/test_performance.py::test_matmul | tee /dockerx/triton/test_performance_matmul.log
# pytest --capture=tee-sys --verbose  python/test/regression/test_performance.py::test_elementwise | tee /dockerx/triton/test_performance_elementwise.log

# pytest --capture=tee-sys --verbose  python/test/regression/test_performance.py::test_matmul[256-256-256]

# pytest --capture=tee-sys --verbose python/test/unit/language/test_core.py::test_empty_kernel[float32]
# pytest --verbose python/test/unit/language/test_core.py::test_load_and_store_op[float32-2]
# pytest --capture=tee-sys --verbose python/test/unit/language/test_core.py::test_load_and_store_op_with_mask
# pytest --verbose python/test/unit/language/test_core.py::test_program_id[float32]
# pytest --capture=tee-sys --verbose python/test/unit/language/test_core.py::test_num_programs[float32]
# pytest --verbose python/test/unit/language/test_core.py::test_unary_op
# pytest --verbose python/test/unit/language/test_core.py::test_bin_op
# pytest --verbose "python/test/unit/language/test_core.py::test_dot"
# pytest --verbose python/test/unit/language/test_core.py::test_cast
# pytest --verbose python/test/unit/language/test_core.py::test_reduce1d
# pytest --verbose python/test/unit/language/test_core.py::test_reduce2d
# pytest --verbose python/test/unit/language/test_core.py::test_math_op
# pytest --capture=tee-sys --verbose python/test/unit/language/test_core.py::test_atomic_rmw
# pytest --verbose python/test/unit/operators/test_blocksparse.py::test_matmul
# pytest --verbose python/test/unit/operators/test_blocksparse.py::test_matmul[DTYPE0-16-False-False-dds]
# pytest --verbose python/test/unit/operators/test_blocksparse.py::test_matmul[DTYPE0-64-False-False-dds]
# pytest --capture=tee-sys --verbose python/test/unit/language/test_core.py::test_matmul

# pytest --capture=tee-sys --verbose python/test/unit/language/test_core.py::test_load_and_store_op_with_mask
# pytest  --capture=tee-sys --verbose "python/test/unit/language/test_core.py::test_masked_load_shared_memory"
# pytest --verbose "python/test/unit/operators/test_blocksparse.py::test_softmax[DTYPE0-256-16]"
# pytest --verbose "python/test/unit/operators/test_blocksparse.py::test_softmax" #|& tee /dockerx/triton/test_softmax.log

# pytest --verbose "python/test/unit/operators/test_blocksparse.py::test_softmax[DTYPE0-1024-16]" # PASSED                                                                                [ 29%]
# pytest --verbose "python/test/unit/operators/test_blocksparse.py::test_softmax[DTYPE0-1024-32]" # FAILED
# pytest --verbose python/test/unit/language/test_core.py::test_permute
# pytest --verbose python/test/unit/language/test_core.py::test_load_cache_modifier

# pytest --verbose python/test/unit/language/test_core.py::test_math_op[log]
# pytest --capture=tee-sys --verbose python/test/unit/language/test_core.py::test_load_and_store_op[float64]
# pytest --verbose "python/test/unit/language/test_core.py::test_bin_op[int8-int64- x % y]"
# pytest --verbose "python/test/unit/language/test_core.py::test_dot[none]" |& tee /dockerx/triton/test_dot_none.log
# pytest --verbose "python/test/unit/language/test_core.py::test_dot[add-rows]"
# pytest --verbose "python/test/unit/language/test_core.py::test_dot[add-cols]"
# pytest --verbose "python/test/unit/language/test_core.py::test_cast[float32-float16-False]"

# pytest --verbose python/test/unit/operators/test_blocksparse.py::test_matmul[DTYPE0-32-False-False-sdd]
# pytest --capture=tee-sys --verbose python/test/unit/operators/test_blocksparse.py::test_softmax[DTYPE0-256-32]

# pytest --verbose python/test/unit/operators/test_blocksparse.py
# pytest --verbose python/test/unit/operators/test_blocksparse.py::test_matmul[DTYPE0-32-False-False-sdd]
# pytest --verbose scripts/amd/test_fptrunc.py
# pytest --verbose scripts/amd/test_fptrunc.py::test_fptrunc[float32-float32-False]
# pytest --verbose "python/test/unit/language/test_core.py::test_cast"
# pytest --verbose "python/test/unit/language/test_core.py::test_cast[float32-float16-False]"
# pytest --verbose "python/test/unit/language/test_core.py::test_cast[float32-bfloat16-False]"
# python python/test/unit/language/test_core.py

# pytest --capture=tee-sys --verbose python/test/unit/language/test_core.py::test_empty_kernel

# pytest --capture=tee-sys --verbose "python/test/unit/language/test_core.py::test_bin_op[int8-int64- x % y]"
# pytest --capture=tee-sys --verbose "python/test/unit/language/test_core.py::test_bin_op[int8-float32- x % y]"
# pytest --capture=tee-sys --verbose "python/test/unit/language/test_core.py::test_bin_op[int8-float16- x % y]"
# pytest --capture=tee-sys --verbose "python/test/unit/language/test_core.py::test_bin_op[float32-float64- x % y]"
# pytest --capture=tee-sys --verbose "python/test/unit/language/test_core.py::test_math_op[exp]"
# pytest --verbose "python/test/unit/operators/test_blocksparse.py"
# pytest --capture=tee-sys --verbose "python/test/unit/operators/test_blocksparse.py::test_matmul[sdd-False-False-16-float16]"

# pytest --capture=tee-sys --verbose "python/test/unit/language/test_core.py::test_arange"

# pytest --verbose "python/test/unit/language/test_core.py::test_masked_load_shared_memory"
# pytest --verbose "python/test/unit/language/test_core.py::test_dot_without_load"
# pytest --verbose "python/test/unit/language/test_core.py::test_fmadot"

# FAILING TESTS
# pytest --verbose "python/test/unit/language/test_core.py::test_bin_op[int8-float16- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[int8-float32- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[int8-float64- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[int16-float16- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[int16-float32- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[int16-float64- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[int32-float16- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[int32-float32- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[int32-float64- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[int64-float16- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[int64-float32- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[int64-float64- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[float16-int8- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[float16-int16- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[float16-int32- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[float16-int64- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[float16-float64- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[float32-int8- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[float32-int16- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[float32-int32- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[float32-int64- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[float32-float64- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[float64-int8- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[float64-int16- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[float64-int32- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[float64-int64- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[float64-float16- x % y]" \
#     "python/test/unit/language/test_core.py::test_bin_op[float64-float32- x % y]"

# do post test steps
# bash scripts/amd/post.sh # it should be run in the run script
