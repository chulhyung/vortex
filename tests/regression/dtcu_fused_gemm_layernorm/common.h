#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Ported from bench ml_kernels/fused/kbt_gemm_layernorm: C = LayerNorm(A*B + bias) affine.
// In-core fused keeps each TM-row strip (TM x N) in local memory and reduces on-chip.
// The DTCU path writes A*B to global memory (mode 1) and a SIMT pass reads the full rows
// back to run layernorm (mode 2) -- the round-trip is the cross-cluster hand-off overhead.
//   mode 0: in-core fused  -> C = LayerNorm(A*B + bias)   (1D row-tile grid, strip in local)
//   mode 1: DTCU matmul    -> D = A*B
//   mode 2: SIMT epilogue  -> C = LayerNorm(D + bias)      (1D row-tile grid, reads global D)

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
  uint64_t ln_w_addr;   // [N] layernorm gamma
  uint64_t ln_b_addr;   // [N] layernorm beta
  uint64_t D_addr;      // A*B intermediate (DTCU path)
  uint64_t C_addr;      // final output [M*N]
  uint64_t desc_addr;
  float    eps;
} kernel_arg_t;

#endif
