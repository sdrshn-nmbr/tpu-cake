builtin.module {
  "tpu_schedule.kernel"() <{sym_name = "matmul", target = "tpu7x", vmem_capacity_bytes = #builtin.int<1048576>, smem_capacity_bytes = #builtin.int<65536>}> ({
  ^bb0(%0: memref<16x32xbf16, #tpu_schedule<memory_space hbm>>, %1: memref<32x16xbf16, #tpu_schedule<memory_space hbm>>, %2: memref<16x16xf32, #tpu_schedule<memory_space hbm>>):
    %3 = "tpu_schedule.alloc"() <{role = "lhs_tile"}> : () -> memref<16x32xbf16, #tpu_schedule<memory_space vmem>>
    %4 = "tpu_schedule.alloc"() <{role = "rhs_tile"}> : () -> memref<32x16xbf16, #tpu_schedule<memory_space vmem>>
    %5 = "tpu_schedule.alloc"() <{role = "accumulator"}> : () -> memref<16x16xf32, #tpu_schedule<memory_space vmem>>
    %6 = "tpu_schedule.semaphore_alloc"() : () -> !tpu_schedule.semaphore
    %7 = "tpu_schedule.semaphore_alloc"() : () -> !tpu_schedule.semaphore
    %8 = "tpu_schedule.dma_start"(%0, %3, %6) <{stage = #builtin.int<0>}> : (memref<16x32xbf16, #tpu_schedule<memory_space hbm>>, memref<16x32xbf16, #tpu_schedule<memory_space vmem>>, !tpu_schedule.semaphore) -> !tpu_schedule.dma_token
    %9 = "tpu_schedule.dma_start"(%1, %4, %7) <{stage = #builtin.int<0>}> : (memref<32x16xbf16, #tpu_schedule<memory_space hbm>>, memref<32x16xbf16, #tpu_schedule<memory_space vmem>>, !tpu_schedule.semaphore) -> !tpu_schedule.dma_token
    "tpu_schedule.dma_wait"(%8) <{stage = #builtin.int<1>}> : (!tpu_schedule.dma_token) -> ()
    "tpu_schedule.dma_wait"(%9) <{stage = #builtin.int<1>}> : (!tpu_schedule.dma_token) -> ()
    "tpu_schedule.mxu_matmul"(%3, %4, %5) <{stage = #builtin.int<2>}> : (memref<16x32xbf16, #tpu_schedule<memory_space vmem>>, memref<32x16xbf16, #tpu_schedule<memory_space vmem>>, memref<16x16xf32, #tpu_schedule<memory_space vmem>>) -> ()
    "tpu_schedule.yield"() : () -> ()
  }) : () -> ()
}

