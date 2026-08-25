#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Ported from bench ml_kernels/fused/kbt_gemm_add_relu: C = relu(A*B + bias).
// Compares an in-core fused kernel (matmul + SIMT epilogue on-chip) against a DTCU
// matmul followed by a separate SIMT epilogue. The DTCU sits outside the cluster, so
// it must write A*B to global memory and the SIMT epilogue reads it back -- that
// round-trip is the cross-cluster hand-off overhead we measure.
//   mode 0: in-core fused  -> C = relu(A*B + bias)   (2D tile grid)
//   mode 1: DTCU matmul    -> D = A*B                (single leader)
//   mode 2: SIMT epilogue  -> C = relu(D + bias)     (2D tile grid, reads global D)

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

typedef struct {
  uint32_t mode;
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t bias_addr;   // [N]
  uint64_t D_addr;      // A*B intermediate (DTCU path)
  uint64_t C_addr;      // final output [M*N]
  uint64_t desc_addr;
} kernel_arg_t;

#endif
