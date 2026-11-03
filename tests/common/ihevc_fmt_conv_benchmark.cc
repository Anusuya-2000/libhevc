/******************************************************************************
 *
 * Copyright (C) 2026 Ittiam Systems Pvt Ltd, Bangalore
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ******************************************************************************/

#include <benchmark/benchmark.h>

#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "ihevc_typedefs.h"
extern "C" {
#include "ihevcd_fmt_conv.h"
#include "iv.h"
}
// clang-format on

#include "BenchmarkCommon.h"
#include "TestCommon.h"

namespace {

enum class FmtConvType {
  k422spTo420p,
  k444spTo420p,
  k444spTo444p,
};

struct FmtConvBenchConfig {
  int wd;
  int ht;
  FmtConvType type;
  IV_ARCH_T arch;
};

ihevcd_fmt_conv_422sp_to_420p_ft* GetFmtConv422spTo420pFn(IV_ARCH_T arch) {
  if (arch == ARCH_NA) {
    return ihevcd_fmt_conv_422sp_to_420p;
  }
#if defined(__aarch64__) || defined(__arm__)
  if (arch == ARCH_ARMV8_GENERIC || arch == ARCH_ARM_A9Q ||
      arch == ARCH_ARM_NEONINTR) {
    return ihevcd_fmt_conv_422sp_to_420p_neonintr;
  }
#endif
  return nullptr;
}

ihevcd_fmt_conv_444sp_to_420p_ft* GetFmtConv444spTo420pFn(IV_ARCH_T arch) {
  if (arch == ARCH_NA) {
    return ihevcd_fmt_conv_444sp_to_420p;
  }
#if defined(__aarch64__) || defined(__arm__)
  if (arch == ARCH_ARMV8_GENERIC || arch == ARCH_ARM_A9Q ||
      arch == ARCH_ARM_NEONINTR) {
    return ihevcd_fmt_conv_444sp_to_420p_neonintr;
  }
#endif
  return nullptr;
}

ihevcd_fmt_conv_444sp_to_444p_ft* GetFmtConv444spTo444pFn(IV_ARCH_T arch) {
  if (arch == ARCH_NA) {
    return ihevcd_fmt_conv_444sp_to_444p;
  }
#if defined(__aarch64__) || defined(__arm__)
  if (arch == ARCH_ARMV8_GENERIC || arch == ARCH_ARM_A9Q ||
      arch == ARCH_ARM_NEONINTR) {
    return ihevcd_fmt_conv_444sp_to_444p_neonintr;
  }
#endif
  return nullptr;
}

bool HasFmtConvFn(IV_ARCH_T arch, FmtConvType type) {
  switch (type) {
    case FmtConvType::k422spTo420p:
      return GetFmtConv422spTo420pFn(arch) != nullptr;
    case FmtConvType::k444spTo420p:
      return GetFmtConv444spTo420pFn(arch) != nullptr;
    case FmtConvType::k444spTo444p:
      return GetFmtConv444spTo444pFn(arch) != nullptr;
  }
  return false;
}

void BM_FmtConv(benchmark::State& state, FmtConvBenchConfig config) {
  const int wd = config.wd;
  const int ht = config.ht;
  const FmtConvType type = config.type;
  const IV_ARCH_T arch = config.arch;

  const bool is_444_src = (type == FmtConvType::k444spTo420p ||
                           type == FmtConvType::k444spTo444p);
  const bool is_444_dst = (type == FmtConvType::k444spTo444p);

  const int src_y_strd = wd;
  const int dst_y_strd = wd;
  const int src_uv_strd = is_444_src ? (2 * wd) : wd;
  const int dst_uv_strd = is_444_dst ? wd : (wd / 2);
  const int dst_uv_ht = is_444_dst ? ht : (ht / 2);

  std::vector<UWORD8> y_src(src_y_strd * ht);
  std::vector<UWORD8> uv_src(src_uv_strd * ht);
  FillRandom(y_src, static_cast<UWORD8>(0), static_cast<UWORD8>(255), 101);
  FillRandom(uv_src, static_cast<UWORD8>(0), static_cast<UWORD8>(255), 202);

  std::vector<UWORD8> y_dst(dst_y_strd * ht, 0xCD);
  std::vector<UWORD8> u_dst(dst_uv_strd * dst_uv_ht, 0xCD);
  std::vector<UWORD8> v_dst(dst_uv_strd * dst_uv_ht, 0xCD);

  std::vector<UWORD8> y_ref(dst_y_strd * ht, 0xCD);
  std::vector<UWORD8> u_ref(dst_uv_strd * dst_uv_ht, 0xCD);
  std::vector<UWORD8> v_ref(dst_uv_strd * dst_uv_ht, 0xCD);

  auto run_planar_bench = [&](auto fn, auto ref_fn, auto invoke) {
    if (!fn) {
      state.SkipWithError("Target function pointer is null");
      return;
    }
    if (arch != ARCH_NA && ref_fn) {
      invoke(ref_fn, y_ref.data(), u_ref.data(), v_ref.data());
      invoke(fn, y_dst.data(), u_dst.data(), v_dst.data());
      if (!VerifyOutput2D(y_ref.data(), y_dst.data(), wd, ht, dst_y_strd) ||
          !VerifyOutput2D(u_ref.data(), u_dst.data(), dst_uv_strd, dst_uv_ht,
                          dst_uv_strd) ||
          !VerifyOutput2D(v_ref.data(), v_dst.data(), dst_uv_strd, dst_uv_ht,
                          dst_uv_strd)) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
    for (auto _ : state) {
      invoke(fn, y_dst.data(), u_dst.data(), v_dst.data());
      benchmark::DoNotOptimize(y_dst.data());
      benchmark::DoNotOptimize(u_dst.data());
      benchmark::DoNotOptimize(v_dst.data());
      benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * wd * ht);
    state.SetBytesProcessed(state.iterations() *
                            (y_dst.size() + u_dst.size() + v_dst.size()));
  };

  switch (type) {
    case FmtConvType::k422spTo420p:
      run_planar_bench(GetFmtConv422spTo420pFn(arch),
                       GetFmtConv422spTo420pFn(ARCH_NA),
                       [&](auto f, UWORD8* yd, UWORD8* ud, UWORD8* vd) {
                         f(y_src.data(), uv_src.data(), yd, ud, vd, wd, ht,
                           src_y_strd, src_uv_strd, dst_y_strd, dst_uv_strd);
                       });
      break;
    case FmtConvType::k444spTo420p:
      run_planar_bench(GetFmtConv444spTo420pFn(arch),
                       GetFmtConv444spTo420pFn(ARCH_NA),
                       [&](auto f, UWORD8* yd, UWORD8* ud, UWORD8* vd) {
                         f(y_src.data(), uv_src.data(), yd, ud, vd, wd, ht,
                           src_y_strd, src_uv_strd, dst_y_strd, dst_uv_strd);
                       });
      break;
    case FmtConvType::k444spTo444p:
      run_planar_bench(GetFmtConv444spTo444pFn(arch),
                       GetFmtConv444spTo444pFn(ARCH_NA),
                       [&](auto f, UWORD8* yd, UWORD8* ud, UWORD8* vd) {
                         f(y_src.data(), uv_src.data(), yd, ud, vd, wd, ht,
                           src_y_strd, src_uv_strd, dst_y_strd, dst_uv_strd);
                       });
      break;
  }
}

}  // namespace

void RegisterAllBenchmarks() {
  struct FmtConvOpInfo {
    FmtConvType type;
    const char* name;
  };

  const FmtConvOpInfo ops[] = {
      {FmtConvType::k422spTo420p, "422sp_to_420p"},
      {FmtConvType::k444spTo420p, "444sp_to_420p"},
      {FmtConvType::k444spTo444p, "444sp_to_444p"},
  };

  const std::pair<int, int> resolutions[] = {
      {640, 360},
      {854, 480},
      {1280, 720},
      {1920, 1080},
      {3840, 2160},
  };

  for (const auto& op : ops) {
    for (const auto& res : resolutions) {
      const int wd = res.first;
      const int ht = res.second;
      for (auto arch : GetBenchmarkArchitectures()) {
        if (arch != ARCH_NA && !HasFmtConvFn(arch, op.type)) {
          continue;
        }
        std::string arch_name = GetArchName(arch);
        std::string name = "BM_FmtConv/" + std::string(op.name) + "/" +
                           std::to_string(wd) + "x" + std::to_string(ht) + "/" +
                           arch_name;

        FmtConvBenchConfig cfg{wd, ht, op.type, arch};
        benchmark::RegisterBenchmark(name.c_str(),
                                     [cfg](benchmark::State& st) {
                                       BM_FmtConv(st, cfg);
                                     })
            ->Unit(benchmark::kMicrosecond);
      }
    }
  }
}
