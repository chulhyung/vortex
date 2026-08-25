// dtcu_fused_gemm_add_relu: in-core fused GEMM+bias+ReLU vs DTCU matmul + SIMT epilogue.
// The in-core path keeps the A*B tile on-chip (local memory) and fuses the epilogue.
// The DTCU path writes A*B to global memory (mode 1) and a separate SIMT pass reads it
// back to apply bias+relu (mode 2) -- the D write+read is the cross-cluster hand-off
// overhead. Synthetic inputs (fused fixtures are 16x16x16, too small for the DTCU tile).

#include "common.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include <VX_types.h>
#include <rvfloats.h>
#include <tensor_cfg.h>
#include <util.h>
#include <vortex.h>

#define GEMM_M 128
#define GEMM_N 256
#define GEMM_K 256

#ifndef REL_TOL
#define REL_TOL 1e-1f
#endif
#ifndef CROSS_TOL
#define CROSS_TOL 5e-2f
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

// Refer to sw/kernel/include/vx_dtensor.h::dtensor_desc_t
struct dtensor_desc_t {
  uint64_t ptrA, ptrB, ptrC, ptrD;
  uint32_t ldmA, ldmB, ldmC, ldmD;
  uint16_t M, N, K;
  uint8_t  fmt_s, fmt_d, flags, shape_n_size;
  uint16_t shape_policy;
  uint32_t reserved2;
};

struct Stats {
  uint64_t cycles = 0, instrs = 0;
  uint64_t loads = 0, stores = 0, stall_lsu = 0, stall_tcu = 0, instr_lsu = 0, instr_tcu = 0;
  uint64_t l2_reads = 0, l2_writes = 0, mem_reads = 0, mem_writes = 0;
  double   host_ms = 0.0;
};
struct DtcuPerf {
  uint64_t op_reqs = 0, out_reqs = 0, compute = 0, next_k_load_stall = 0, tma_mem_wait = 0,
           tma_buf_starve = 0, tma_op_fill = 0, tma_addrgen = 0, tma_store_issue_stall = 0,
           store_drain = 0, smem_read_model = 0, next_tile_load_stall = 0, prev_tile_store_stall = 0,
           desc_wait = 0, busy = 0, tma_acc_init = 0;
};

static void read_stats(vx_device_h device, Stats& s) {
  vx_mpm_query(device, 0, VX_CSR_MCYCLE, 0, &s.cycles);
  vx_mpm_query(device, 0, VX_CSR_MINSTRET, 0, &s.instrs);
  const uint32_t cc = VX_DCR_MPM_CLASS_CORE;
  vx_mpm_query(device, cc, VX_CSR_MPM_LOADS, 0, &s.loads);
  vx_mpm_query(device, cc, VX_CSR_MPM_STORES, 0, &s.stores);
  vx_mpm_query(device, cc, VX_CSR_MPM_STALL_LSU, 0, &s.stall_lsu);
  vx_mpm_query(device, cc, VX_CSR_MPM_STALL_TCU, 0, &s.stall_tcu);
  vx_mpm_query(device, cc, VX_CSR_MPM_INSTR_LSU, 0, &s.instr_lsu);
  vx_mpm_query(device, cc, VX_CSR_MPM_INSTR_TCU, 0, &s.instr_tcu);
  const uint32_t mc = VX_DCR_MPM_CLASS_MEM;
  vx_mpm_query(device, mc, VX_CSR_MPM_L2CACHE_READS, 0, &s.l2_reads);
  vx_mpm_query(device, mc, VX_CSR_MPM_L2CACHE_WRITES, 0, &s.l2_writes);
  vx_mpm_query(device, mc, VX_CSR_MPM_MEM_READS, 0, &s.mem_reads);
  vx_mpm_query(device, mc, VX_CSR_MPM_MEM_WRITES, 0, &s.mem_writes);
}
static void read_dtcu(vx_device_h device, DtcuPerf& d) {
  const uint32_t c = VX_DCR_MPM_CLASS_DTCU;
  vx_mpm_query(device, c, VX_CSR_MPM_DTCU_OP_REQS, 0, &d.op_reqs);
  vx_mpm_query(device, c, VX_CSR_MPM_DTCU_OUT_REQS, 0, &d.out_reqs);
  vx_mpm_query(device, c, VX_CSR_MPM_DTCU_COMPUTE, 0, &d.compute);
  vx_mpm_query(device, c, VX_CSR_MPM_DTCU_NEXT_K_LOAD_STALL, 0, &d.next_k_load_stall);
  vx_mpm_query(device, c, VX_CSR_MPM_DTCU_TMA_MEM_WAIT, 0, &d.tma_mem_wait);
  vx_mpm_query(device, c, VX_CSR_MPM_DTCU_TMA_BUF_STARVE, 0, &d.tma_buf_starve);
  vx_mpm_query(device, c, VX_CSR_MPM_DTCU_TMA_OP_FILL, 0, &d.tma_op_fill);
  vx_mpm_query(device, c, VX_CSR_MPM_DTCU_TMA_ADDRGEN, 0, &d.tma_addrgen);
  vx_mpm_query(device, c, VX_CSR_MPM_DTCU_TMA_STORE_ISSUE_STALL, 0, &d.tma_store_issue_stall);
  vx_mpm_query(device, c, VX_CSR_MPM_DTCU_STORE_DRAIN, 0, &d.store_drain);
  vx_mpm_query(device, c, VX_CSR_MPM_DTCU_SMEM_READ_MODEL, 0, &d.smem_read_model);
  vx_mpm_query(device, c, VX_CSR_MPM_DTCU_NEXT_TILE_LOAD_STALL, 0, &d.next_tile_load_stall);
  vx_mpm_query(device, c, VX_CSR_MPM_DTCU_PREV_TILE_STORE_STALL, 0, &d.prev_tile_store_stall);
  vx_mpm_query(device, c, VX_CSR_MPM_DTCU_DESC_WAIT, 0, &d.desc_wait);
  vx_mpm_query(device, c, VX_CSR_MPM_DTCU_BUSY, 0, &d.busy);
  vx_mpm_query(device, c, VX_CSR_MPM_DTCU_TMA_ACC_INIT, 0, &d.tma_acc_init);
}

// path 0 = in-core fused (single launch). path 1 = DTCU matmul + SIMT epilogue (two launches).
static int run_case(uint32_t path,
                    uint32_t M, uint32_t N, uint32_t K,
                    uint32_t tcu_tileM, uint32_t tcu_tileN, uint8_t shape_n_size,
                    const std::vector<itype_t>& hA, const std::vector<itype_t>& hB,
                    const std::vector<otype_t>& hBias,
                    std::vector<otype_t>& out, Stats& stats,
                    Stats* epi_stats = nullptr, DtcuPerf* dtcu_perf = nullptr) {
  vx_device_h device = nullptr;
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h queue = nullptr;
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  vx_buffer_h A_buf=nullptr,B_buf=nullptr,Bias_buf=nullptr,D_buf=nullptr,C_buf=nullptr,desc_buf=nullptr;
  uint64_t A_addr,B_addr,Bias_addr,D_addr,C_addr,desc_addr=0;
  RT_CHECK(vx_buffer_create(device, hA.size()*sizeof(itype_t), VX_MEM_READ, &A_buf));
  RT_CHECK(vx_buffer_address(A_buf, &A_addr));
  RT_CHECK(vx_buffer_create(device, hB.size()*sizeof(itype_t), VX_MEM_READ, &B_buf));
  RT_CHECK(vx_buffer_address(B_buf, &B_addr));
  RT_CHECK(vx_buffer_create(device, hBias.size()*sizeof(otype_t), VX_MEM_READ, &Bias_buf));
  RT_CHECK(vx_buffer_address(Bias_buf, &Bias_addr));
  RT_CHECK(vx_buffer_create(device, out.size()*sizeof(otype_t), VX_MEM_READ_WRITE, &D_buf));
  RT_CHECK(vx_buffer_address(D_buf, &D_addr));
  RT_CHECK(vx_buffer_create(device, out.size()*sizeof(otype_t), VX_MEM_READ_WRITE, &C_buf));
  RT_CHECK(vx_buffer_address(C_buf, &C_addr));

  RT_CHECK(vx_enqueue_write(queue, A_buf, 0, hA.data(), hA.size()*sizeof(itype_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, B_buf, 0, hB.data(), hB.size()*sizeof(itype_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, Bias_buf, 0, hBias.data(), hBias.size()*sizeof(otype_t), 0, nullptr, nullptr));

  dtensor_desc_t desc{};
  if (path == 1) {
    desc.ptrA = A_addr; desc.ptrB = B_addr; desc.ptrC = 0; desc.ptrD = D_addr;
    desc.ldmA = K; desc.ldmB = K; desc.ldmC = 0; desc.ldmD = N;
    desc.M = M; desc.N = N; desc.K = K;
    desc.fmt_s = vt::ITYPE::id; desc.fmt_d = vt::OTYPE::id;
    desc.flags = 0x1; // ZERO_ACC -> D = A*B
    desc.shape_n_size = shape_n_size; desc.shape_policy = 0;
    RT_CHECK(vx_buffer_create(device, sizeof(dtensor_desc_t), VX_MEM_READ, &desc_buf));
    RT_CHECK(vx_buffer_address(desc_buf, &desc_addr));
    RT_CHECK(vx_enqueue_write(queue, desc_buf, 0, &desc, sizeof(desc), 0, nullptr, nullptr));
  }

  vx_module_h module_ = nullptr; vx_kernel_h kernel = nullptr;
  RT_CHECK(vx_module_load_file(device, "kernel.vxbin", &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // one arg blob per launch (mode differs)
  kernel_arg_t a_mm{}, a_epi{};
  a_mm.M=a_epi.M=M; a_mm.N=a_epi.N=N; a_mm.K=a_epi.K=K;
  a_mm.A_addr=a_epi.A_addr=A_addr; a_mm.B_addr=a_epi.B_addr=B_addr;
  a_mm.bias_addr=a_epi.bias_addr=Bias_addr; a_mm.D_addr=a_epi.D_addr=D_addr;
  a_mm.C_addr=a_epi.C_addr=C_addr; a_mm.desc_addr=a_epi.desc_addr=desc_addr;

  // MPM counters (MCYCLE, MEM/DTCU classes) reset per launch, so the DTCU path reads
  // stats after the DTCU matmul (mode 1) and again after the SIMT epilogue (mode 2),
  // and the caller sums them. This also gives a clean matmul-vs-hand-off breakdown.
  auto t0 = std::chrono::high_resolution_clock::now();
  vx_event_h ev1=nullptr, ev2=nullptr, read_ev=nullptr;
  if (path == 0) {
    a_mm.mode = 0;
    vx_launch_info_t li{}; li.struct_size=sizeof(li); li.kernel=kernel;
    li.args_host=&a_mm; li.args_size=sizeof(a_mm);
    li.ndim=2; li.grid_dim[0]=N/tcu_tileN; li.grid_dim[1]=M/tcu_tileM;
    li.block_dim[0]=NUM_THREADS; li.block_dim[1]=1;
    li.lmem_size=tcu_tileM*tcu_tileN*sizeof(otype_t);
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &ev1));
    RT_CHECK(vx_enqueue_read(queue, out.data(), C_buf, 0, out.size()*sizeof(otype_t), 1, &ev1, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    read_stats(device, stats);
  } else {
    // mode 1: DTCU matmul -> D
    a_mm.mode = 1;
    vx_launch_info_t li_mm{}; li_mm.struct_size=sizeof(li_mm); li_mm.kernel=kernel;
    li_mm.args_host=&a_mm; li_mm.args_size=sizeof(a_mm);
    li_mm.ndim=1; li_mm.grid_dim[0]=1; li_mm.block_dim[0]=1;
    RT_CHECK(vx_enqueue_launch(queue, &li_mm, 0, nullptr, &ev1));
    RT_CHECK(vx_event_wait_value(ev1, 1, VX_TIMEOUT_INFINITE));
    read_stats(device, stats);                    // DTCU matmul stats
    if (dtcu_perf) read_dtcu(device, *dtcu_perf);
    // mode 2: SIMT epilogue -> C
    a_epi.mode = 2;
    vx_launch_info_t li_epi{}; li_epi.struct_size=sizeof(li_epi); li_epi.kernel=kernel;
    li_epi.args_host=&a_epi; li_epi.args_size=sizeof(a_epi);
    li_epi.ndim=2; li_epi.grid_dim[0]=N/tcu_tileN; li_epi.grid_dim[1]=M/tcu_tileM;
    li_epi.block_dim[0]=NUM_THREADS; li_epi.block_dim[1]=1;
    RT_CHECK(vx_enqueue_launch(queue, &li_epi, 0, nullptr, &ev2));
    RT_CHECK(vx_enqueue_read(queue, out.data(), C_buf, 0, out.size()*sizeof(otype_t), 1, &ev2, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    if (epi_stats) read_stats(device, *epi_stats); // SIMT epilogue stats
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  stats.host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  if (read_ev) vx_event_release(read_ev);
  if (ev2) vx_event_release(ev2);
  if (ev1) vx_event_release(ev1);
  vx_buffer_release(A_buf); vx_buffer_release(B_buf); vx_buffer_release(Bias_buf);
  vx_buffer_release(D_buf); vx_buffer_release(C_buf);
  if (desc_buf) vx_buffer_release(desc_buf);
  vx_kernel_release(kernel); vx_module_release(module_); vx_queue_release(queue);
  vx_device_dump_perf(device, stdout);
  vx_device_release(device);
  return 0;
}

int main(int argc, char** argv) {
  (void)argc; (void)argv;
  const uint32_t M=GEMM_M, N=GEMM_N, K=GEMM_K;
  const uint32_t tcu_i_ratio = 4 / sizeof(itype_t);
  const uint32_t tcu_tileM = cfg::tileM, tcu_tileN = cfg::tileN, tcu_tileK = cfg::tileK * tcu_i_ratio;
  uint8_t shape_n_size = 8;
  while (shape_n_size > 1 && (N % (shape_n_size * 16u)) != 0) --shape_n_size;
  if ((M%tcu_tileM)||(N%tcu_tileN)||(K%tcu_tileK)||(M%64)||(N%(shape_n_size*16u))||(K%(8*tcu_i_ratio))) {
    std::cerr << "dtcu_fused_gemm_add_relu: shape unsupported" << std::endl; return -1;
  }

  std::vector<itype_t> hA((size_t)M*K), hB((size_t)N*K);
  std::vector<otype_t> hBias(N);
  std::vector<float> hRef((size_t)M*N);

  for (uint32_t m=0;m<M;++m)
    for (uint32_t k=0;k<K;++k) {
      float v = float(int((m*7 + k*3) % 9) - 4);
      hA[(size_t)m*K+k] = (itype_t)Convert<vt::ITYPE>::from_float(v);
    }
  for (uint32_t n=0;n<N;++n)
    for (uint32_t k=0;k<K;++k) {
      float v = float(int((k*5 + n*2) % 7) - 3);
      hB[(size_t)n*K+k] = (itype_t)Convert<vt::ITYPE>::from_float(v);
    }
  for (uint32_t n=0;n<N;++n)
    hBias[n] = (otype_t)Convert<vt::OTYPE>::from_float(float(int((n*11) % 13) - 6));

  for (uint32_t m=0;m<M;++m)
    for (uint32_t n=0;n<N;++n) {
      float acc = Convert<vt::OTYPE>::to_float(hBias[n]);
      for (uint32_t k=0;k<K;++k)
        acc += Convert<vt::ITYPE>::to_float(hA[(size_t)m*K+k]) * Convert<vt::ITYPE>::to_float(hB[(size_t)n*K+k]);
      hRef[(size_t)m*N+n] = (acc > 0.0f) ? acc : 0.0f; // relu
    }

  std::vector<otype_t> out_ic((size_t)M*N,0), out_dt((size_t)M*N,0);
  Stats st_ic{}, st_mm{}, st_epi{}; DtcuPerf dp{};
  std::cout << "dtcu_fused_gemm_add_relu: ---------- Running In-core fused ----------" << std::endl;
  RT_CHECK(run_case(0, M,N,K, tcu_tileM,tcu_tileN, shape_n_size, hA,hB,hBias, out_ic, st_ic));
  std::cout << "dtcu_fused_gemm_add_relu: ---------- Running DTCU + SIMT epilogue ----------" << std::endl;
  RT_CHECK(run_case(1, M,N,K, tcu_tileM,tcu_tileN, shape_n_size, hA,hB,hBias, out_dt, st_mm, &st_epi, &dp));

  std::cout << "dtcu_fused_gemm_add_relu: ---------- RESULT ----------" << std::endl;
  int e_ic=0,e_dt=0,e_cx=0; float mr_ic=0,mr_dt=0,mr_cx=0;
  auto rel=[](float g,float r){ return std::fabs(g-r)/(std::fabs(r)+1e-3f); };
  for (uint32_t i=0;i<M*N;++i) {
    float r=hRef[i], gi=Convert<vt::OTYPE>::to_float(out_ic[i]), gd=Convert<vt::OTYPE>::to_float(out_dt[i]);
    float a=rel(gi,r), b=rel(gd,r), c=rel(gi,gd);
    mr_ic=std::max(mr_ic,a); mr_dt=std::max(mr_dt,b); mr_cx=std::max(mr_cx,c);
    if (a>REL_TOL){ if(e_ic<MAX_ERRORS) std::cerr<<"IC mismatch ["<<i<<"] got="<<gi<<" exp="<<r<<"\n"; ++e_ic; }
    if (b>REL_TOL){ if(e_dt<MAX_ERRORS) std::cerr<<"DTCU mismatch ["<<i<<"] got="<<gd<<" exp="<<r<<"\n"; ++e_dt; }
    if (c>CROSS_TOL){ if(e_cx<MAX_ERRORS) std::cerr<<"Cross mismatch ["<<i<<"] ic="<<gi<<" dtcu="<<gd<<"\n"; ++e_cx; }
  }

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "M="<<M<<" N="<<N<<" K="<<K
            << " max_rel_ic="<<mr_ic<<" max_rel_dtcu="<<mr_dt<<" max_rel_cross="<<mr_cx << std::endl;
  auto ps=[](const char* tag, const Stats& s){
    std::cout<<tag<<" host_ms="<<s.host_ms<<" cycles="<<s.cycles<<" instrs="<<s.instrs
             <<" IPC="<<(s.cycles?double(s.instrs)/double(s.cycles):0.0)<<std::endl;
    std::cout<<"    core: loads="<<s.loads<<" stores="<<s.stores
             <<" instr_lsu="<<s.instr_lsu<<" instr_tcu="<<s.instr_tcu
             <<" stall_lsu="<<s.stall_lsu<<" stall_tcu="<<s.stall_tcu<<std::endl;
    std::cout<<"    mem:  l2_reads="<<s.l2_reads<<" l2_writes="<<s.l2_writes
             <<" mem_reads="<<s.mem_reads<<" mem_writes="<<s.mem_writes<<std::endl;
  };
  // DTCU path total = matmul (mode 1) + SIMT epilogue (mode 2); counters reset per launch.
  uint64_t dt_cycles = st_mm.cycles + st_epi.cycles;
  uint64_t dt_mem    = st_mm.mem_reads + st_mm.mem_writes + st_epi.mem_reads + st_epi.mem_writes;
  uint64_t ic_mem    = st_ic.mem_reads + st_ic.mem_writes;
  ps("[In-core fused]      ", st_ic);
  ps("[DTCU matmul]        ", st_mm);
  ps("[DTCU SIMT epilogue] ", st_epi);
  std::cout << "[DTCU path total] cycles=" << dt_cycles
            << " mem_reads=" << (st_mm.mem_reads+st_epi.mem_reads)
            << " mem_writes=" << (st_mm.mem_writes+st_epi.mem_writes) << std::endl;
  // Hand-off = the A*B (D) round-trip DTCU must make to reach SIMT: DTCU writes D in the
  // matmul phase, the SIMT epilogue reads it back. In-core never leaves the chip.
  std::cout << "[Hand-off] D_bytes=" << (uint64_t)M*N*sizeof(otype_t)
            << " matmul_D_write_lines=" << dp.out_reqs
            << " epi_D_read_loads=" << st_epi.loads
            << " epi_mem_stall=" << st_epi.stall_lsu
            << " simt_epilogue_cycles=" << st_epi.cycles << std::endl;
  std::cout << "[Ratio DTCU/in-core] cycles="
            << (st_ic.cycles?double(dt_cycles)/double(st_ic.cycles):0.0)
            << " mem_total=" << (ic_mem?double(dt_mem)/double(ic_mem):0.0)
            << " simt_epilogue_frac=" << (dt_cycles?double(st_epi.cycles)/double(dt_cycles):0.0)
            << std::endl;
  std::cout << "[DTCU MPM] op_reqs="<<dp.op_reqs<<" out_reqs="<<dp.out_reqs<<" compute="<<dp.compute
            << " tma_mem_wait="<<dp.tma_mem_wait<<" tma_buf_starve="<<dp.tma_buf_starve
            << " tma_op_fill="<<dp.tma_op_fill<<" tma_acc_init="<<dp.tma_acc_init
            << " store_drain="<<dp.store_drain<<" desc_wait="<<dp.desc_wait<<" busy="<<dp.busy<<std::endl;

  if (e_ic||e_dt||e_cx){ std::cerr<<"FAILED: e_ic="<<e_ic<<" e_dtcu="<<e_dt<<" e_cross="<<e_cx<<std::endl; return e_ic+e_dt+e_cx; }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
