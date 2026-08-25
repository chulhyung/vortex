// dtcu_pi0_i_k1: in-core TCU vs DTCU on the Pi0 FFN linear GEMM (D = A*B + bias),
// ported from bench robotics_inference/pi/thor/pi0_k1. Real fp32 fixtures (a/b/bias.bin)
// are down-converted to the WMMA input type. Both paths are checked against a CPU
// reference and each other; cycles + MEM/DTCU counters are reported for comparison.

#include "common.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <VX_types.h>
#include <rvfloats.h>
#include <tensor_cfg.h>
#include <util.h>
#include <vortex.h>

// Problem shape (fixed by the committed fixtures).
#define GEMM_M 128
#define GEMM_N 256
#define GEMM_K 256

#ifndef REL_TOL
#define REL_TOL 1e-1f   // fp16 inputs, fp32 accumulate over K=256
#endif
#ifndef CROSS_TOL
#define CROSS_TOL 5e-2f // in-core vs DTCU (both fp16 -> fp32)
#endif
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

// ---- type conversion ----
template <typename T>
struct Convert;

template <>
struct Convert<vt::fp32> {
  using dtype = float;
  static inline dtype from_float(float f) { return f; }
  static inline float to_float(dtype x) { return x; }
};

template <>
struct Convert<vt::fp16> {
  using dtype = uint16_t;
  static inline dtype from_float(float f) {
    return rv_ftoh_s(bit_cast<uint32_t>(f), 0, nullptr);
  }
  static inline float to_float(dtype x) {
    uint32_t bits = rv_htof_s(x, 0, nullptr);
    return bit_cast<float>(bits);
  }
};

template <>
struct Convert<vt::bf16> {
  using dtype = uint16_t;
  static inline dtype from_float(float f) {
    return rv_ftob_s(bit_cast<uint32_t>(f), 0, nullptr);
  }
  static inline float to_float(dtype x) {
    uint32_t bits = rv_btof_s(x, 0, nullptr);
    return bit_cast<float>(bits);
  }
};

template <>
struct Convert<vt::tf32> {
  using dtype = uint32_t;
  static inline dtype from_float(float f) {
    return rv_ftotf32_s(bit_cast<uint32_t>(f), 0, nullptr);
  }
  static inline float to_float(dtype x) {
    uint32_t bits = rv_tf32tof_s(x, 0, nullptr);
    return bit_cast<float>(bits);
  }
};

// Load a little-endian fp32 binary fixture.
static std::vector<float> load_bin_f32(const std::string& path, size_t count) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "cannot open fixture: " << path << std::endl;
    std::exit(1);
  }
  std::vector<float> v(count);
  f.read(reinterpret_cast<char*>(v.data()), count * sizeof(float));
  if ((size_t)f.gcount() != count * sizeof(float)) {
    std::cerr << "short read on " << path << " got " << f.gcount()
              << " want " << count * sizeof(float) << std::endl;
    std::exit(1);
  }
  return v;
}

// Refer to sw/kernel/include/vx_dtensor.h::dtensor_desc_t
struct dtensor_desc_t {
  uint64_t ptrA;
  uint64_t ptrB;
  uint64_t ptrC;
  uint64_t ptrD;
  uint32_t ldmA;
  uint32_t ldmB;
  uint32_t ldmC;
  uint32_t ldmD;
  uint16_t M;
  uint16_t N;
  uint16_t K;
  uint8_t  fmt_s;
  uint8_t  fmt_d;
  uint8_t  flags;
  uint8_t  shape_n_size;
  uint16_t shape_policy;
  uint32_t reserved2;
};

struct Stats {
  uint64_t cycles = 0;
  uint64_t instrs = 0;
  uint64_t loads = 0, stores = 0;
  uint64_t stall_lsu = 0, stall_tcu = 0;
  uint64_t instr_lsu = 0, instr_tcu = 0;
  uint64_t l2_reads = 0, l2_writes = 0;
  uint64_t mem_reads = 0, mem_writes = 0;
  double   host_ms = 0.0;
};

struct DtcuPerf {
  uint64_t op_reqs = 0, out_reqs = 0, compute = 0, next_k_load_stall = 0, tma_mem_wait = 0,
           tma_buf_starve = 0, tma_op_fill = 0, tma_addrgen = 0, tma_store_issue_stall = 0,
           store_drain = 0, smem_read_model = 0, next_tile_load_stall = 0, prev_tile_store_stall = 0,
           desc_wait = 0, busy = 0, tma_acc_init = 0;
};

// Run one GEMM. mode 0 = in-core TCU (2D tile grid, bias epilogue). mode 1 = DTCU
// (single thread fires a descriptor with C = bias broadcast). Fills out[] with D.
static int run_case(uint32_t mode,
                    uint32_t M, uint32_t N, uint32_t K,
                    uint32_t tcu_tileM, uint32_t tcu_tileN,
                    uint8_t shape_n_size,
                    const std::vector<itype_t>& hA,
                    const std::vector<itype_t>& hB,
                    const std::vector<otype_t>& hBias, // [N]
                    const std::vector<otype_t>& hC,    // [M*N] bias broadcast (DTCU)
                    std::vector<otype_t>& out,
                    Stats& stats,
                    DtcuPerf* dtcu_perf = nullptr) {
  vx_device_h device = nullptr;
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h queue = nullptr;
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  kernel_arg_t karg{};
  karg.mode = mode;
  karg.M = M;
  karg.N = N;
  karg.K = K;

  vx_buffer_h A_buf = nullptr, B_buf = nullptr, C_buf = nullptr, D_buf = nullptr,
              Bias_buf = nullptr, desc_buf = nullptr;
  RT_CHECK(vx_buffer_create(device, hA.size() * sizeof(itype_t), VX_MEM_READ, &A_buf));
  RT_CHECK(vx_buffer_address(A_buf, &karg.A_addr));
  RT_CHECK(vx_buffer_create(device, hB.size() * sizeof(itype_t), VX_MEM_READ, &B_buf));
  RT_CHECK(vx_buffer_address(B_buf, &karg.B_addr));
  RT_CHECK(vx_buffer_create(device, hC.size() * sizeof(otype_t), VX_MEM_READ, &C_buf));
  RT_CHECK(vx_buffer_address(C_buf, &karg.C_addr));
  RT_CHECK(vx_buffer_create(device, hBias.size() * sizeof(otype_t), VX_MEM_READ, &Bias_buf));
  RT_CHECK(vx_buffer_address(Bias_buf, &karg.bias_addr));
  RT_CHECK(vx_buffer_create(device, out.size() * sizeof(otype_t), VX_MEM_READ_WRITE, &D_buf));
  RT_CHECK(vx_buffer_address(D_buf, &karg.D_addr));

  dtensor_desc_t desc{};
  if (mode == 1) {
    desc.ptrA = karg.A_addr;
    desc.ptrB = karg.B_addr;
    desc.ptrC = karg.C_addr;
    desc.ptrD = karg.D_addr;
    desc.ldmA = K; // A row-major
    desc.ldmB = K; // B col-major
    desc.ldmC = N;
    desc.ldmD = N;
    desc.M = M;
    desc.N = N;
    desc.K = K;
    desc.fmt_s = vt::ITYPE::id;
    desc.fmt_d = vt::OTYPE::id;
    desc.flags = 0x0; // D = C + A*B (C = bias broadcast)
    desc.shape_n_size = shape_n_size;
    desc.shape_policy = 0;
    desc.reserved2 = 0;
    RT_CHECK(vx_buffer_create(device, sizeof(dtensor_desc_t), VX_MEM_READ, &desc_buf));
    RT_CHECK(vx_buffer_address(desc_buf, &karg.desc_addr));
  }

  RT_CHECK(vx_enqueue_write(queue, A_buf, 0, hA.data(), hA.size() * sizeof(itype_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, B_buf, 0, hB.data(), hB.size() * sizeof(itype_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, C_buf, 0, hC.data(), hC.size() * sizeof(otype_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, Bias_buf, 0, hBias.data(), hBias.size() * sizeof(otype_t), 0, nullptr, nullptr));
  if (mode == 1) {
    RT_CHECK(vx_enqueue_write(queue, desc_buf, 0, &desc, sizeof(dtensor_desc_t), 0, nullptr, nullptr));
  }

  const char* kernel_file = "kernel.vxbin";
  vx_module_h module_ = nullptr;
  vx_kernel_h kernel = nullptr;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel      = kernel;
  li.args_host   = &karg;
  li.args_size   = sizeof(karg);
  if (mode == 0) {
    li.ndim         = 2;
    li.grid_dim[0]  = N / tcu_tileN;
    li.grid_dim[1]  = M / tcu_tileM;
    li.block_dim[0] = NUM_THREADS;
    li.block_dim[1] = 1;
    li.lmem_size    = tcu_tileM * tcu_tileN * sizeof(otype_t); // accumulator tile spill
  } else {
    li.ndim         = 1;
    li.grid_dim[0]  = 1;
    li.block_dim[0] = 1;
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  RT_CHECK(vx_enqueue_read(queue, out.data(), D_buf, 0, out.size() * sizeof(otype_t), 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  auto t1 = std::chrono::high_resolution_clock::now();
  stats.host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  RT_CHECK(vx_mpm_query(device, 0, VX_CSR_MCYCLE, 0, &stats.cycles));
  RT_CHECK(vx_mpm_query(device, 0, VX_CSR_MINSTRET, 0, &stats.instrs));
  {
    const uint32_t cls = VX_DCR_MPM_CLASS_CORE;
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_LOADS,     0, &stats.loads));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STORES,    0, &stats.stores));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STALL_LSU, 0, &stats.stall_lsu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STALL_TCU, 0, &stats.stall_tcu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_INSTR_LSU, 0, &stats.instr_lsu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_INSTR_TCU, 0, &stats.instr_tcu));
  }
  {
    const uint32_t cls = VX_DCR_MPM_CLASS_MEM;
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_L2CACHE_READS,  0, &stats.l2_reads));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_L2CACHE_WRITES, 0, &stats.l2_writes));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_MEM_READS,      0, &stats.mem_reads));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_MEM_WRITES,     0, &stats.mem_writes));
  }
  if (mode == 1 && dtcu_perf) {
    const uint32_t cls = VX_DCR_MPM_CLASS_DTCU;
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_OP_REQS,     0, &dtcu_perf->op_reqs));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_OUT_REQS,    0, &dtcu_perf->out_reqs));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_COMPUTE,     0, &dtcu_perf->compute));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_NEXT_K_LOAD_STALL,    0, &dtcu_perf->next_k_load_stall));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_TMA_MEM_WAIT,    0, &dtcu_perf->tma_mem_wait));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_TMA_BUF_STARVE,    0, &dtcu_perf->tma_buf_starve));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_TMA_OP_FILL,   0, &dtcu_perf->tma_op_fill));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_TMA_ADDRGEN,     0, &dtcu_perf->tma_addrgen));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_TMA_STORE_ISSUE_STALL,  0, &dtcu_perf->tma_store_issue_stall));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_STORE_DRAIN, 0, &dtcu_perf->store_drain));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_SMEM_READ_MODEL,      0, &dtcu_perf->smem_read_model));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_NEXT_TILE_LOAD_STALL,  0, &dtcu_perf->next_tile_load_stall));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_PREV_TILE_STORE_STALL, 0, &dtcu_perf->prev_tile_store_stall));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_DESC_WAIT,   0, &dtcu_perf->desc_wait));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_BUSY,        0, &dtcu_perf->busy));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_TMA_ACC_INIT, 0, &dtcu_perf->tma_acc_init));
  }

  vx_event_release(read_ev);
  vx_event_release(launch_ev);
  vx_buffer_release(A_buf);
  vx_buffer_release(B_buf);
  vx_buffer_release(C_buf);
  vx_buffer_release(Bias_buf);
  vx_buffer_release(D_buf);
  if (desc_buf) vx_buffer_release(desc_buf);
  vx_kernel_release(kernel);
  vx_module_release(module_);
  vx_queue_release(queue);
  vx_device_dump_perf(device, stdout);
  vx_device_release(device);
  return 0;
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  const uint32_t M = GEMM_M, N = GEMM_N, K = GEMM_K;

  // in-core TCU native tile
  const uint32_t tcu_i_ratio = 4 / sizeof(itype_t);
  const uint32_t tcu_tileM = cfg::tileM;
  const uint32_t tcu_tileN = cfg::tileN;
  const uint32_t tcu_tileK = cfg::tileK * tcu_i_ratio;

  // DTCU native tile: tile_n = shape_n_size*16. Pick the largest that divides N.
  uint8_t shape_n_size = 8; // tile_n = 128
  while (shape_n_size > 1 && (N % (shape_n_size * 16u)) != 0) --shape_n_size;

  if ((M % tcu_tileM) || (N % tcu_tileN) || (K % tcu_tileK) ||
      (M % 64) || (N % (shape_n_size * 16u)) || (K % (8 * tcu_i_ratio))) {
    std::cerr << "dtcu_pi0_i_k1: shape unsupported M=" << M << " N=" << N << " K=" << K << std::endl;
    return -1;
  }

  const std::string dir = FIXTURE_DIR;
  std::vector<float> af = load_bin_f32(dir + "/a.bin", (size_t)M * K);      // M x K row-major
  std::vector<float> bf = load_bin_f32(dir + "/b.bin", (size_t)K * N);      // K x N row-major
  std::vector<float> biasf = load_bin_f32(dir + "/bias.bin", (size_t)N);    // N

  std::vector<itype_t> hA(M * K);
  std::vector<itype_t> hB((size_t)N * K);      // stored col-major: hB[n*K + k]
  std::vector<otype_t> hBias(N);
  std::vector<otype_t> hC((size_t)M * N);      // bias broadcast (DTCU accumulator)
  std::vector<float>   hRef((size_t)M * N);

  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t k = 0; k < K; ++k)
      hA[m * K + k] = (itype_t)Convert<vt::ITYPE>::from_float(af[m * K + k]);

  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; ++n)
      hB[(size_t)n * K + k] = (itype_t)Convert<vt::ITYPE>::from_float(bf[(size_t)k * N + n]);

  for (uint32_t n = 0; n < N; ++n)
    hBias[n] = (otype_t)Convert<vt::OTYPE>::from_float(biasf[n]);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t n = 0; n < N; ++n)
      hC[(size_t)m * N + n] = hBias[n];

  // CPU reference: D = A*B + bias, using the same reduced-precision operands.
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t n = 0; n < N; ++n) {
      float acc = Convert<vt::OTYPE>::to_float(hBias[n]);
      for (uint32_t k = 0; k < K; ++k) {
        float a = Convert<vt::ITYPE>::to_float(hA[m * K + k]);
        float b = Convert<vt::ITYPE>::to_float(hB[(size_t)n * K + k]);
        acc += a * b;
      }
      hRef[(size_t)m * N + n] = acc;
    }

  std::vector<otype_t> out_tcu((size_t)M * N, 0);
  std::vector<otype_t> out_dtcu((size_t)M * N, 0);
  Stats stats_tcu{}, stats_dtcu{};
  DtcuPerf dtcu_perf{};

  std::cout << "dtcu_pi0_i_k1: ---------- Running In-core TCU ----------" << std::endl;
  RT_CHECK(run_case(0, M, N, K, tcu_tileM, tcu_tileN, shape_n_size, hA, hB, hBias, hC, out_tcu, stats_tcu));
  std::cout << "dtcu_pi0_i_k1: ---------- Running DTCU ----------" << std::endl;
  RT_CHECK(run_case(1, M, N, K, tcu_tileM, tcu_tileN, shape_n_size, hA, hB, hBias, hC, out_dtcu, stats_dtcu, &dtcu_perf));

  std::cout << "dtcu_pi0_i_k1: ---------- RESULT ----------" << std::endl;
  int errors_tcu = 0, errors_dtcu = 0, cross_errors = 0;
  float max_rel_tcu = 0, max_rel_dtcu = 0, max_rel_cross = 0;
  auto rel = [](float got, float ref) {
    return std::fabs(got - ref) / (std::fabs(ref) + 1e-3f);
  };
  for (uint32_t i = 0; i < M * N; ++i) {
    float ref      = hRef[i];
    float got_tcu  = Convert<vt::OTYPE>::to_float(out_tcu[i]);
    float got_dtcu = Convert<vt::OTYPE>::to_float(out_dtcu[i]);
    float rt = rel(got_tcu, ref), rd = rel(got_dtcu, ref), rc = rel(got_tcu, got_dtcu);
    max_rel_tcu = std::max(max_rel_tcu, rt);
    max_rel_dtcu = std::max(max_rel_dtcu, rd);
    max_rel_cross = std::max(max_rel_cross, rc);
    if (rt > REL_TOL) {
      if (errors_tcu < MAX_ERRORS)
        std::cerr << "TCU mismatch [" << i << "]: got=" << got_tcu << " exp=" << ref << "\n";
      ++errors_tcu;
    }
    if (rd > REL_TOL) {
      if (errors_dtcu < MAX_ERRORS)
        std::cerr << "DTCU mismatch [" << i << "]: got=" << got_dtcu << " exp=" << ref << "\n";
      ++errors_dtcu;
    }
    if (rc > CROSS_TOL) {
      if (cross_errors < MAX_ERRORS)
        std::cerr << "Cross mismatch [" << i << "]: tcu=" << got_tcu << " dtcu=" << got_dtcu << "\n";
      ++cross_errors;
    }
  }

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "M=" << M << " N=" << N << " K=" << K
            << " max_rel_tcu=" << max_rel_tcu << " max_rel_dtcu=" << max_rel_dtcu
            << " max_rel_cross=" << max_rel_cross << std::endl;
  auto print_stats = [](const char* tag, const Stats& s) {
    std::cout << tag << " host_ms=" << s.host_ms
              << " cycles=" << s.cycles << " instrs=" << s.instrs
              << " IPC=" << (s.cycles ? double(s.instrs) / double(s.cycles) : 0.0) << std::endl;
    std::cout << "    core: loads=" << s.loads << " stores=" << s.stores
              << " instr_lsu=" << s.instr_lsu << " instr_tcu=" << s.instr_tcu
              << " stall_lsu=" << s.stall_lsu << " stall_tcu=" << s.stall_tcu << std::endl;
    std::cout << "    mem:  l2_reads=" << s.l2_reads << " l2_writes=" << s.l2_writes
              << " mem_reads=" << s.mem_reads << " mem_writes=" << s.mem_writes << std::endl;
  };
  print_stats("[In-core TCU]", stats_tcu);
  print_stats("[DTCU]       ", stats_dtcu);
  std::cout << "[Ratio DTCU/TCU] cycles="
            << (stats_tcu.cycles ? double(stats_dtcu.cycles) / double(stats_tcu.cycles) : 0.0)
            << " mem_total="
            << ((stats_tcu.mem_reads + stats_tcu.mem_writes)
                  ? double(stats_dtcu.mem_reads + stats_dtcu.mem_writes) / double(stats_tcu.mem_reads + stats_tcu.mem_writes) : 0.0)
            << std::endl;
  std::cout << "[DTCU MPM] op_reqs=" << dtcu_perf.op_reqs
            << " out_reqs=" << dtcu_perf.out_reqs
            << " compute=" << dtcu_perf.compute
            << " tma_mem_wait=" << dtcu_perf.tma_mem_wait
            << " tma_buf_starve=" << dtcu_perf.tma_buf_starve
            << " tma_op_fill=" << dtcu_perf.tma_op_fill
            << " tma_acc_init=" << dtcu_perf.tma_acc_init
            << " store_drain=" << dtcu_perf.store_drain
            << " desc_wait=" << dtcu_perf.desc_wait
            << " busy=" << dtcu_perf.busy << std::endl;

  if (errors_tcu || errors_dtcu || cross_errors) {
    std::cerr << "FAILED: errors_tcu=" << errors_tcu
              << " errors_dtcu=" << errors_dtcu
              << " cross_errors=" << cross_errors << std::endl;
    return errors_tcu + errors_dtcu + cross_errors;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
