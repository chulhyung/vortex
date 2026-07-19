# DTCU Perf Stat & Gate

> DTCU의 내부 카운터(TMA 메모리 트래픽 + 오버랩 사이클)를 **MPM CSR로 노출**해 어떤 커널/테스트에서든 `vx_mpm_query`로 읽히게 한다. 더불어 DTCU를 in-core TCU와 **독립적으로 켜는 게이트 `VX_CFG_EXT_DTCU_ENABLE`**를 둔다.
> 설계 확정: 2026-06-22 (bug_fix_320). **v3.0 이식: 2026-07** — 클래스 번호·게이트·읽기 경로가 v3.0에 맞게 바뀜(§ 각 항목). 관련: [[DTCU Implementation]], [[DTCU Latency Modeling]], [[DTCU Test Cases]].

---

## 0. 왜 CSR인가 (getter 아님)

처음엔 "Dtcu에 getter 노출 → 테스트 main.cpp가 읽어 출력"을 고려했으나 **그 main을 가진 테스트에서만** 나온다. in-core TCU 카운터(`STALL_TCU`, `INSTR_TCU`)나 L2 카운터(`L2CACHE_READS`)처럼 **MPM CSR에 저장**하면 **어떤 커널/테스트든** `vx_mpm_query`로 읽혀 표준 PERF 출력에 들어간다. → CSR 방식 채택.

DTCU는 **cluster-level 단일 엔진**이므로 카운터는 **클러스터 전역**(per-core 합산 아님). 읽기 경로: `vx_mpm_query`(core 0) → SimX가 `core_->socket()->cluster()->perf_stats().dtcu` 접근(L2/DXA stat이 cluster로 읽히는 것과 동일 패턴, v3.0에선 [csr_unit.cpp](sim/simx/csr_unit.cpp)의 `get_csr()` MPM 분기 — 구 `emulator.cpp`가 v3.0에서 `csr_unit.cpp`로 분리됨).

---

## 1. VX_CFG_EXT_DTCU_ENABLE 게이트

- **목적**: DTCU를 in-core TCU와 **독립**으로 빌드 on/off. (bug_fix에선 `EXT_TCU_ENABLE` 하나로 둘이 같이 켜졌음.)
- **방식 (v3.0에서 바뀜)**: bug_fix는 "toml에 없는 순수 `-DDTCU_ENABLE`"였으나, **v3.0은 `VX_config.toml`에 정식 등록**한다 — `VX_CFG_EXT_DTCU_ENABLE`(기본 false) + `VX_CFG_EXT_DTCU_ENABLED` expr + **MISA 커스텀 확장 비트(bit 11)**. `ci/gen_config.py`가 `-DVX_CFG_EXT_DTCU_ENABLE`로 방출하고 SimX 코드는 `#ifdef VX_CFG_EXT_DTCU_ENABLE`로 가른다. (DTCU는 여전히 SimX-only지만, v3.0의 TOML→헤더 파이프라인 때문에 config 홈이 생김. RTL이 생기면 그 비트가 그대로 쓰임.)
  - **주의**: `VX_CFG_EXT_DTCU_ENABLE`은 `VX_CFG_EXT_TCU_ENABLE`을 **함의하지 않는다** — 둘은 독립 게이트. 단 in-core TCU와 DTCU를 **함께** 비교하는 `dtcu_compare`는 두 플래그 모두 필요(§4).
- **무엇을 gate하나** (`#ifdef VX_CFG_EXT_DTCU_ENABLE`):
  - `cluster.h`/`cluster.cpp`: `Dtcu`/`DtcuTma` include·멤버·`dtcu()` 접근자, `Dtcu` 생성 + **L2 arbiter 행(row) bind**. ⚠️ arbiter 가드가 **두 곳**(생성+소켓결선 / arbiter→l2cache) — 둘 다 DTCU 포함해야 함(하나 빠지면 DTCU-only 빌드가 행).
  - `types.h`: `DtcuType{START,POLL}` + `IntrDtcuArgs` + OpType/IntrArgs variant 등록.
  - `decode.cpp`: **EXT3 (RISCV_CUSTOM2)** 케이스 — funct3=1 START / funct3=2 POLL → `FUType::SFU`. (bug_fix의 CUSTOM0/funct7=2/`DTCU_Control`에서 완전 분리 — v3.0 TCU가 그 인코딩을 점유하기 때문.)
  - `sfu_unit.cpp`: SFU PE에서 `cluster()->dtcu()->start()/poll()` 라우팅. (bug_fix의 `execute.cpp`/`DtcuControlUnit` 대체 — v3.0엔 `execute.cpp`가 없고 실행이 각 FuncUnit `on_tick`으로 분산.)
  - `csr_unit.cpp`: DTCU MPM 클래스(9) readout.
- **빌드**: `dtcu_compare`는 `-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DTCU_ENABLE`, DTCU 단독 테스트(`dtcu_basic`)는 `-DVX_CFG_EXT_DTCU_ENABLE`만. (테스트 Makefile이 `$(if $(findstring ...))`로 필요한 플래그를 강제 — sgemm_tcu_sp 등 확장 테스트와 동일 관행.)

---

## 2. MPM 카운터 (클래스 9 = `VX_DCR_MPM_CLASS_DTCU`)

**v3.0에서 클래스 번호가 바뀜.** bug_fix는 DTCU=3이었으나, **v3.0은 클래스 3을 TEX가 점유**(BASE=0/CORE=1/MEM=2/**TEX=3**/RASTER=4/OM=5/DXA=6/TCU=7/VM=8). 그래서 **DTCU = 9** 신설. 같은 CSR 주소공간(0xB03~)을 클래스로 multiplex(다른 클래스와 충돌 없음 — 클래스 선택 후 읽음). 클러스터급 엔진 선례인 **DXA(클래스 6)** 를 1:1 미러링.

> **2026-07-19 전면 개명** (커밋 eccb41ede~): 카운터 전수조사 결과를 반영해 **관찰자-가족 체계**로 재명명. canonical name 하나가 멤버/PerfStats/CSR/모든 출력 라벨에 동일하게 쓰인다. **CSR 주소는 전부 불변** — 구 라벨(`wait_tma=` 등)을 파싱하던 스크립트만 갱신 필요.

**읽기 규칙 3줄**: ① FSM 가족(표 A)만 서로 합산 — `busy = ΣFSM + desc_wait + 전이tick(~18)`이 정확히 성립 ② 엔진 가족(표 B, `tma_` 접두)은 compute와 **동시에** 도는 다른 렌즈 — FSM 값과 절대 합산 금지 ③ `smem_read_model`은 compute **내부** 성분(합산 금지), op/out_reqs는 사이클이 아니라 **cache line 수**.

### 표 A — FSM 관찰 (타임라인 분할, 합산 가능)

| CSR | addr | 의미 | 실측 OFF/ON* |
|---|---|---|---|
| `..._COMPUTE` | 0xB05 | 연산 파이프라인 **점유** = max(MAC, SRAM읽기, acc-RMW)+lat — 순수 MAC 아님 (현 shape은 acc-bound 2049>2048) | 16,440 / 16,440 |
| `..._NEXT_K_LOAD_STALL` | 0xB06 | 같은 타일 안 다음 K타일 operand 대기 (구 wait_tma — TMA 끄면 커지던 역설 해소) | 4,430 / 0 |
| `..._NEXT_TILE_LOAD_STALL` | 0xB0E | 타일 경계 K0 노출 대기 (T0 cold-start 포함) | 10,912 / 4,744 |
| `..._PREV_TILE_STORE_STALL` | 0xB0F | store 핸드오프가 **이전 타일** store에 막힘 (store-bound 마커) | 0 / 0 |
| `..._STORE_DRAIN` | 0xB0C | 노출된 store 배출 (overlap: 마지막 타일만 / blocking: 매 타일). "배출"=발행+acc읽기, 메모리 완료 아님 | 4,092 / 1,023 |
| `..._DESC_WAIT` | 0xB10 | 디스크립터 페치 창 (DESC_REQ+DESC_WAIT) | 24 / 24 |
| `..._BUSY` | 0xB11 | busy_인 총 사이클. `MCYCLE−busy` = 커널측 오버헤드(~550) | ≈35.9k / 22,249 |

### 표 B — 엔진 관찰 (`tma_` 접두, compute와 동시 — FSM과 합산 금지)

| CSR | addr | 의미 | 실측 OFF/ON* |
|---|---|---|---|
| `..._TMA_MEM_WAIT` | 0xB07 | load 채널 응답 대기 = **고유 DRAM 지연 + MSHR 스로틀 + 경합의 합** (경합만이 아님!) | 6,842 / 6,870 |
| `..._TMA_BUF_STARVE` | 0xB08 | load 채널 idle + 일감 존재(타일 내 K 또는 다음 타일 K0) + 빈 버퍼 없음 = **3번째 버퍼의 headroom** | 0 / 2,795 |
| `..._TMA_OP_FILL` | 0xB09 | operand scratchpad 채우기 (구 buf_write에서 acc 분리) | 3,080 / 3,080 |
| `..._TMA_ACC_INIT` | 0xB12 | K0 fill의 accumulator 초기화 몫 (별도 SRAM; C-preload 또는 zero-fill) | 4,096 / 4,096 |
| `..._TMA_ADDRGEN` | 0xB0A | AGU setup — 킥당 고정 상수(3)의 에코. store AGU는 0-cost 미모델 | 24 / 24 |
| `..._TMA_STORE_ISSUE_STALL` | 0xB0B | store 라인 발행 못 함 (포트 양보/큐). fire-and-forget이라 메모리측 대기는 어디에도 없음 | 0 / 0 |

### 표 C — 모델 성분 / 트래픽

| CSR | addr | 의미 | 실측 |
|---|---|---|---|
| `..._SMEM_READ_MODEL` | 0xB0D | SRAM 읽기 **모델 추정치** — compute의 max() 내부 항, **어디에도 합산 금지**. swizzle A/B 비교 전용 | 4,104 |
| `..._OP_REQS` | 0xB03 | operand(A·B + K0의 C) coalesced **cache line** 수 | 1,280 |
| `..._OUT_REQS` | 0xB04 | output(D) cache line 수 (desc 1라인은 별도, 미포함) | 512 |

\* M=128 N=64 K=32 fp16, tma_on_off 기준 blocking/overlap+lookahead. `_H`(high 32b)는 +0x80 (0xB83~0xB92). `CSR_READ_64`가 lo/hi 두 case를 자동 생성한다([csr_unit.cpp](sim/simx/csr_unit.cpp)).

### 단위 주의 — line 수, 코어와 비교 시
- `DTCU_OP_REQS`/`OUT_REQS` = **coalesce된 cache-line 수**(`coalesce_to_lines`→`tma_req_lines_.size()`, [dtcu_tma.cpp](sim/simx/dtcu/dtcu_tma.cpp)).
- 코어의 **L2/memory req 카운터도 cache-line 단위**(MemReq당 `++reads/++writes`, v3.0 [cache.cpp](sim/simx/mem/cache.cpp)) → **DTCU 엔진 line ↔ 코어 L2/memory line 직접 비교 가능**. (v3.0 실측 교차검증: DTCU MEM-class `mem_reads`≈`op_reqs`+desc, `mem_writes`≈`out_reqs`+store — TLM 트래픽이 표준 메모리 카운터에도 그대로 잡힘.)
- ⚠️ 코어 **dcache `reqs`**는 LSU-lane 레벨(coalesce 전)이라 DTCU 엔진 line과 직접 비교 금지. 총 메모리 트래픽 비교는 **L2/memory line 기준**으로.

### 왜 이게 중요한가 (지난 의문 해소)
"DTCU 쓰면 코어 메모리 트래픽이 적다"는 일부 **회계**다 — operand load가 코어 dcache가 아니라 **엔진 TMA 경로**로 가서 코어 카운터에 안 잡힐 뿐. `DTCU_OP_REQS`/`OUT_REQS`를 노출하면 **엔진 트래픽 + 코어 트래픽 = 총 메모리 트래픽**을 볼 수 있어, "코어만 적은 거냐 진짜 총량이 적은 거냐"를 정량으로 답할 수 있다. (엔진과 코어는 같은 L2/2-뱅크 메모리를 공유·경쟁 — v3.0에선 DTCU가 L2 arbiter 행 하나를 차지, [cluster.cpp](sim/simx/cluster.cpp), `#ifdef VX_CFG_EXT_DTCU_ENABLE`.)

---

## 3. 구현 지점 요약 (v3.0, 구현+검증 완료)

**MPM CSR stat (게이트 무관 — 클래스 9 정의는 모든 빌드에서 존재; 미빌드 시 query는 0):**
| 파일 | 변경 |
|---|---|
| `VX_types.toml` | `VX_DCR_MPM_CLASS_DTCU = 9` + `[csr_mpm_dtcu]` 11개 CSR(+`_H`). **`configure`/`gen_config.py`가 `sw/VX_types.h`·`hw/VX_types.vh`를 재생성**(bug_fix의 `make -C hw config` 대체 — v3.0은 `*.toml`→헤더 파이프라인). |
| `sim/simx/dtcu/dtcu.h` | `struct PerfStats {...}` + `PerfStats perf_stats() const` (private 카운터 묶어 반환). |
| `sim/simx/cluster.h`/`.cpp` | `Cluster::PerfStats`에 `Dtcu::PerfStats dtcu;` 슬롯 + `perf_stats()`에서 `perf_stats.dtcu = dtcu_->perf_stats();` 채움. |
| `sim/simx/csr_unit.cpp` | `get_csr()` MPM 분기에 `case VX_DCR_MPM_CLASS_DTCU:` — `cluster()->perf_stats().dtcu.*`를 `CSR_READ_64`. `#ifdef VX_CFG_EXT_DTCU_ENABLE`(미빌드 시 case 부재, query는 default로 0). (bug_fix의 `emulator.cpp` 대체.) |
| `tests/regression/dtcu_compare/main.cpp` | `vx_mpm_query(device, VX_DCR_MPM_CLASS_DTCU, VX_CSR_MPM_DTCU_*, 0, &v)`로 11개 읽어 `[DTCU MPM]` 출력. |

**`VX_CFG_EXT_DTCU_ENABLE` 게이트 (v3.0 ISA 경로):**
| 파일 | 변경 |
|---|---|
| `sim/simx/types.h` | `DtcuType{START,POLL}` + `IntrDtcuArgs` + variant 등록(`#ifdef VX_CFG_EXT_DTCU_ENABLE`). (bug_fix의 `FUType::DTCU_Control` 대신 `FUType::SFU` 재사용.) |
| `sim/simx/decode.cpp` | `case Opcode::EXT3:` funct3=1 START(rs1=desc_addr)/2 POLL(rd) → `FUType::SFU`. (bug_fix CUSTOM0/funct7=2에서 이동 — 충돌 회피.) |
| `sim/simx/sfu_unit.cpp` | SFU PE 분기: START→`dtcu->start(src)`, POLL→`dst=dtcu->poll()`. (DXA.ISSUE 선례.) |
| `sim/simx/cluster.h`/`.cpp` | include·`dtcu_` 멤버·생성 + L2 arbiter 행 bind(**가드 2곳 모두**)+reset. |
| `sw/kernel/include/vx_dtensor.h` | `dtensor_desc_t` + `dtensor_start/poll` (`.insn r RISCV_CUSTOM2, funct3=1/2`). |
| 테스트 Makefile | `-DVX_CFG_EXT_TCU_ENABLE`/`-DVX_CFG_EXT_DTCU_ENABLE`를 `findstring`으로 강제. |

**검증**: (a) `-DVX_CFG_EXT_DTCU_ENABLE`만 → `dtcu_basic` PASS. (b) `-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DTCU_ENABLE` → `dtcu_compare` PASS(in-core TCU == DTCU == CPU ref) + `[DTCU MPM]` readback이 엔진 내부 카운터와 완전 일치.

## 4. 읽는 법

DTCU 카운터는 다른 MPM stat(core=1, mem=2, dxa=6)과 **같은 방식**으로 본다. DTCU는 **클래스 9**.

### (1) 표준 방법 — explicit `vx_mpm_query` (v3.0 현재 경로)
**v3.0의 `vx_mpm_query`는 클래스를 인자로 받는다** — bug_fix의 별도 `vx_dcr_write(VX_DCR_BASE_MPM_CLASS, ...)`가 없어짐:
```c
uint64_t op_reqs = 0, wait_tma = 0;
vx_mpm_query(device, VX_DCR_MPM_CLASS_DTCU, VX_CSR_MPM_DTCU_OP_REQS,  0, &op_reqs); // = class 9
vx_mpm_query(device, VX_DCR_MPM_CLASS_DTCU, VX_CSR_MPM_DTCU_WAIT_TMA, 0, &wait_tma);
```
DTCU는 클러스터 단일 엔진이라 **core 0에서 한 번**만 읽으면 된다(코어별 합산 X). `dtcu_compare`가 이 방식으로 11개를 읽어 출력한다.

### (2) `--perf=9` 자동 dump — ✅ 배선 완료 (2026-07-17)
bug_fix에선 `--perf=3`이 `runtime/stub/perf.cpp`의 dtcu case를 타 `PERF: dtcu:` 3줄을 자동 출력했다. **v3.0에선 perf-dump가 [sw/runtime/common/legacy_perf.cpp](sw/runtime/common/legacy_perf.cpp)로 이동**했다. 이제 두 곳을 배선함:
1. **[legacy_perf.cpp](sw/runtime/common/legacy_perf.cpp) `case VX_DCR_MPM_CLASS_DTCU:`** — DXA(클래스 6) 블록을 미러링. cluster당 대표 core를 `vx_mpm_query(mpm_class=9)`로 합산해 `PERF: dtcu:` 3줄 출력(traffic / overlap / detail). **호스트 런타임 코드라 `#ifdef VX_CFG_EXT_DTCU_ENABLE` 안 씀** — DTCU 미빌드 빌드에선 outer default가 0을 돌려주므로 그냥 0으로 찍힘([csr_unit.cpp](sim/simx/csr_unit.cpp) MPM switch default, 에러 아님).
2. **[dtcu_compare/main.cpp](tests/regression/dtcu_compare/main.cpp)** — `run_case()`의 device release 직전 `vx_device_dump_perf(device, stdout)` 추가. (dump는 자동이 아니라 각 테스트가 명시 호출해야 함 — dxa_copy/sgemm_tcu 선례. 이게 없어서 `--perf=9`가 `PERF:` 한 줄도 안 냈던 것.)

**주의**: `--perf`는 `VORTEX_PROFILING` env로 전달됨([blackbox.sh:72,173](ci/blackbox.sh#L72) — `--perf=9` → `-DPERF_ENABLE` + `VORTEX_PROFILING=9`). dtcu_compare는 mode마다 device를 새로 열고 닫으므로 dump가 **2번** 나옴: mode 0(in-core)는 DTCU 카운터 전부 0(엔진 미사용), mode 1(DTCU)이 실측치.

**검증 (2026-07-17, `--perf=9`, dtcu_compare 기본 SIZE_MULT):** auto-dump 3줄이 explicit `[DTCU MPM]` readback과 **11개 전부 일치**, `PASSED!`:
```
PERF: dtcu: mem_reqs=1792 (op=1280, out=512) [L2 cache lines]
PERF: dtcu: compute=16440, wait_for_tma=0, mem_wait=8699, wait_for_buf=0
PERF: dtcu: buf_write=7176, addrgen=24, store_wait=3543, store_drain=1023, opread=4104
```

> ⚠️ **아래 수치는 bug_fix_320 측정치**(M=128 N=64 K=32). v3.0 재측정 예정 — 구조·경로는 v3.0 기준으로 갱신됨. (v3.0 동일 GEMM 실측에서도 전 카운터가 엔진 stdout과 일치함은 확인.)
> ```
> PERF: dtcu: mem_reqs=1792 (op=1280, out=512) [L2 cache lines]
> PERF: dtcu: compute=18488, wait_for_tma=0, mem_wait=8457, wait_for_buf=0
> PERF: dtcu: buf_write=904, addrgen=24, store_wait=4021, store_drain=132, opread=2568
> ```
> 검증: 위 수치는 엔진 자체 stdout(`[DTCU] overlap cycles` / `L2 MemReq count`, v3.0에선 `DP(2,...)` 디버그 게이트)과 **전 카운터 일치**. `mem_reqs`(1792)는 op+out만 — engine desc 요청 1개는 제외(엔진 stdout total=1793 = +desc).
