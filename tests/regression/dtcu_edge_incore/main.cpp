// dtcu_edge/incore: in-core TCU GEMM + fused scale. Standalone (compare its kernel_cycles against
// dtcu_edge/dtcu, which runs the same GEMM+scale but offloaded to DTCU with an output reload).

#include "common.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <VX_types.h>
#include <rvfloats.h>
#include <tensor_cfg.h>
#include <util.h>
#include <vortex.h>

#define FLOAT_ULP 6
#define MAX_ERRORS 100

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 != _ret) {                                         \
      std::cerr << "Runtime Error: " << #_expr               \
                << " returned " << _ret << std::endl;        \
      return _ret;                                           \
    }                                                        \
  } while (false)

using namespace vortex;
namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_THREADS>;
using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;

template <typename T> struct Convert;
template <> struct Convert<vt::fp32> {
  using dtype = float;
  static inline dtype from_float(float f) { return f; }
  static inline float to_float(dtype x) { return x; }
};
template <> struct Convert<vt::fp16> {
  using dtype = uint16_t;
  static inline dtype from_float(float f) { return rv_ftoh_s(bit_cast<uint32_t>(f), 0, nullptr); }
  static inline float to_float(dtype x) { return bit_cast<float>(rv_htof_s(x, 0, nullptr)); }
};
template <> struct Convert<vt::bf16> {
  using dtype = uint16_t;
  static inline dtype from_float(float f) { return rv_ftob_s(bit_cast<uint32_t>(f), 0, nullptr); }
  static inline float to_float(dtype x) { return bit_cast<float>(rv_btof_s(x, 0, nullptr)); }
};

static inline int ulp_diff(float a, float b) {
  if (std::isnan(a) && std::isnan(b)) return 0;
  if (std::isinf(a) || std::isinf(b)) return (a == b) ? 0 : 0x7fffffff;
  int ia, ib;
  std::memcpy(&ia, &a, sizeof(int));
  std::memcpy(&ib, &b, sizeof(int));
  if (ia < 0) ia = 0x80000000 - ia;
  if (ib < 0) ib = 0x80000000 - ib;
  return std::abs(ia - ib);
}

static uint64_t kernel_span(const std::vector<uint64_t>& cyc, uint32_t total) {
  if (total == 0) return 0;
  uint64_t mn = std::numeric_limits<uint64_t>::max(), mx = 0;
  for (uint32_t i = 0; i < total; ++i) {
    if (cyc[i] < mn) mn = cyc[i];
    if (cyc[total + i] > mx) mx = cyc[total + i];
  }
  return (mx > mn) ? (mx - mn) : 0;
}

int main(int argc, char** argv) {
  (void)argc; (void)argv;
  const float scale = POST_SCALE;

  const uint32_t tcu_i_ratio = 4 / sizeof(itype_t);
  const uint32_t tcu_tileM = cfg::tileM;
  const uint32_t tcu_tileN = cfg::tileN;
  const uint32_t tcu_tileK = cfg::tileK * tcu_i_ratio;

  const uint32_t dtcu_tileM = 64, dtcu_tileN = 32;
  const uint32_t dtcu_tileK = (sizeof(itype_t) == 2) ? 16 : 8;
  const uint32_t M = SIZE_MULT * dtcu_tileM;
  const uint32_t N = SIZE_MULT * dtcu_tileN;
  const uint32_t K = SIZE_MULT * dtcu_tileK;

  if ((M % tcu_tileM) || (N % tcu_tileN) || (K % tcu_tileK)) { std::cerr << "incore: bad size\n"; return -1; }

  const uint32_t grid_x = N / tcu_tileN, grid_y = M / tcu_tileM;
  const uint32_t num_warps_total = grid_x * grid_y;

  std::vector<itype_t> hA(M * K), hB(K * N);
  std::vector<otype_t> hC(M * N);
  std::vector<float>   hRef(M * N);
  for (uint32_t i = 0; i < M; ++i) for (uint32_t k = 0; k < K; ++k)
    hA[i * K + k] = (itype_t)Convert<vt::ITYPE>::from_float(float((i * 13 + k * 7) % 11) - 5.0f);
  for (uint32_t k = 0; k < K; ++k) for (uint32_t j = 0; j < N; ++j)
    hB[j * K + k] = (itype_t)Convert<vt::ITYPE>::from_float(float((k * 5 + j * 17) % 9) - 4.0f);
  for (uint32_t i = 0; i < M; ++i) for (uint32_t j = 0; j < N; ++j)
    hC[i * N + j] = (otype_t)Convert<vt::OTYPE>::from_float(float((i * 9 + j * 11) % 13) - 6.0f);
  for (uint32_t i = 0; i < M; ++i) for (uint32_t j = 0; j < N; ++j) {
    float acc = Convert<vt::OTYPE>::to_float(hC[i * N + j]);
    for (uint32_t k = 0; k < K; ++k)
      acc += Convert<vt::ITYPE>::to_float(hA[i * K + k]) * Convert<vt::ITYPE>::to_float(hB[j * K + k]);
    hRef[i * N + j] = acc * scale;   // = (C + A*B) * scale
  }

  std::vector<otype_t> out(M * N);
  vx_buffer_h A_buf=nullptr, B_buf=nullptr, C_buf=nullptr, D_buf=nullptr, cyc_buf=nullptr, args_buffer=nullptr;
  std::cout << "dtcu_edge/incore: ---- in-core TCU GEMM + fused scale ----\n";

  vx_device_h device = nullptr;
  RT_CHECK(vx_dev_open(&device));
  vx_buffer_h krnl_buffer = nullptr;
  RT_CHECK(vx_upload_kernel_file(device, "kernel.vxbin", &krnl_buffer));

  kernel_arg_t karg{};
  karg.grid_dim[0] = grid_x; karg.grid_dim[1] = grid_y;
  karg.block_dim[0] = NUM_THREADS; karg.block_dim[1] = 1;
  karg.M = M; karg.N = N; karg.K = K;
  karg.scale = scale; karg.num_warps_total = num_warps_total;

  RT_CHECK(vx_mem_alloc(device, hA.size()*sizeof(itype_t), VX_MEM_READ, &A_buf));
  RT_CHECK(vx_mem_address(A_buf, &karg.A_addr));
  RT_CHECK(vx_mem_alloc(device, hB.size()*sizeof(itype_t), VX_MEM_READ, &B_buf));
  RT_CHECK(vx_mem_address(B_buf, &karg.B_addr));
  RT_CHECK(vx_mem_alloc(device, hC.size()*sizeof(otype_t), VX_MEM_READ, &C_buf));
  RT_CHECK(vx_mem_address(C_buf, &karg.C_addr));
  RT_CHECK(vx_mem_alloc(device, out.size()*sizeof(otype_t), VX_MEM_READ_WRITE, &D_buf));
  RT_CHECK(vx_mem_address(D_buf, &karg.D_addr));
  RT_CHECK(vx_mem_alloc(device, 2*num_warps_total*sizeof(uint64_t), VX_MEM_READ_WRITE, &cyc_buf));
  RT_CHECK(vx_mem_address(cyc_buf, &karg.cyc_addr));

  RT_CHECK(vx_copy_to_dev(A_buf, hA.data(), 0, hA.size()*sizeof(itype_t)));
  RT_CHECK(vx_copy_to_dev(B_buf, hB.data(), 0, hB.size()*sizeof(itype_t)));
  RT_CHECK(vx_copy_to_dev(C_buf, hC.data(), 0, hC.size()*sizeof(otype_t)));
  std::vector<otype_t> zeros(M*N, 0);
  RT_CHECK(vx_copy_to_dev(D_buf, zeros.data(), 0, zeros.size()*sizeof(otype_t)));
  std::vector<uint64_t> cyc_zero(2*num_warps_total, 0);
  RT_CHECK(vx_copy_to_dev(cyc_buf, cyc_zero.data(), 0, cyc_zero.size()*sizeof(uint64_t)));

  RT_CHECK(vx_upload_bytes(device, &karg, sizeof(kernel_arg_t), &args_buffer));
  RT_CHECK(vx_dcr_write(device, VX_DCR_BASE_MPM_CLASS, VX_DCR_MPM_CLASS_CORE));

  auto t0 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  auto t1 = std::chrono::high_resolution_clock::now();
  double host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  RT_CHECK(vx_copy_from_dev(out.data(), D_buf, 0, out.size()*sizeof(otype_t)));
  std::vector<uint64_t> hcyc(2*num_warps_total);
  RT_CHECK(vx_copy_from_dev(hcyc.data(), cyc_buf, 0, hcyc.size()*sizeof(uint64_t)));
  uint64_t kernel_cycles = kernel_span(hcyc, num_warps_total);

  uint64_t full_cycles = 0;
  RT_CHECK(vx_mpm_query(device, VX_CSR_MCYCLE, 0, &full_cycles));

  int errors = 0;
  for (uint32_t i = 0; i < M*N; ++i) {
    float got = Convert<vt::OTYPE>::to_float(out[i]);
    if (ulp_diff(got, hRef[i]) > FLOAT_ULP) { if (errors < MAX_ERRORS) std::cerr << "mismatch[" << i << "]: got=" << got << " exp=" << hRef[i] << "\n"; ++errors; }
  }

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "M=" << M << " N=" << N << " K=" << K << " scale=" << scale
            << " (in-core tile " << tcu_tileM << "x" << tcu_tileN << "x" << tcu_tileK << ")\n";
  std::cout << "[in-core]  kernel_cycles=" << kernel_cycles << "  host_ms=" << host_ms
            << "  (full MCYCLE=" << full_cycles << ")\n";

  vx_mem_free(A_buf); vx_mem_free(B_buf); vx_mem_free(C_buf); vx_mem_free(D_buf);
  vx_mem_free(cyc_buf); vx_mem_free(args_buffer); vx_mem_free(krnl_buffer);
  vx_dev_close(device);

  if (errors) { std::cerr << "FAILED: errors=" << errors << "\n"; return errors; }
  std::cout << "PASSED\n";
  return 0;
}
