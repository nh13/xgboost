/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once
#include <dmlc/common.h>  // for OMPException
#include <dmlc/omp.h>     // for omp_get_thread_num, omp_get_max_threads, omp_in_parallel

#include <algorithm>  // for min
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t

#include "executor.h"         // for Sched, BindWorker
#include "xgboost/logging.h"  // for CHECK_GE

#if defined(_MSC_VER)
#include <limits>       // for numeric_limits
#include <type_traits>  // for conditional_t, is_signed

#include "xgboost/base.h"  // for omp_ulong
#endif

namespace xgboost::common {
// Without OpenMP, the pragmas are ignored and dmlc's OpenMP stubs run loops serially.
struct OpenMPExecutor {
  [[nodiscard]] static std::int32_t MaxNumWorkers() noexcept {
#if !defined(_OPENMP)
    return 1;
#else
    if (omp_in_parallel()) {
      return 1;
    }
#if defined(_MSC_VER)
    // MSVC implements OpenMP 2.0, which lacks omp_get_thread_limit and omp_get_max_active_levels.
    auto limit = std::numeric_limits<std::int32_t>::max();
#else
    // With no active parallel level allowed (OMP_MAX_ACTIVE_LEVELS=0), a region gets one thread,
    // and Execute2d would fail its team-size check if more were requested.
    if (omp_get_max_active_levels() < 1) {
      return 1;
    }
    auto limit = omp_get_thread_limit();
#endif
    CHECK_GE(limit, 1) << "Invalid thread limit for OpenMP.";
    return std::min({omp_get_num_procs(), omp_get_max_threads(), limit});
#endif
  }

  template <typename Index, typename Func>
  static void Execute(Index size, std::int32_t n_threads, Sched sched, Func&& fn) {
#if defined(_MSC_VER)
    // msvc doesn't support unsigned integer as openmp index.
    using OmpInd = std::conditional_t<std::is_signed<Index>::value, Index, omp_ulong>;
#else
    using OmpInd = Index;
#endif
    OmpInd length = static_cast<OmpInd>(size);
    CHECK_GE(n_threads, 1);
    if (size <= 0) {
      return;
    }
    n_threads = static_cast<std::int32_t>(std::min<std::size_t>(n_threads, size));

    auto&& callback = detail::BindWorker(fn, [] { return Worker{omp_get_thread_num()}; });
    dmlc::OMPException exc;
    switch (sched.sched) {
      case Sched::kAuto: {
#pragma omp parallel for num_threads(n_threads)
        for (OmpInd i = 0; i < length; ++i) {
          exc.Run(callback, i);
        }
        break;
      }
      case Sched::kDynamic: {
        if (sched.chunk == 0) {
#pragma omp parallel for num_threads(n_threads) schedule(dynamic)
          for (OmpInd i = 0; i < length; ++i) {
            exc.Run(callback, i);
          }
        } else {
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, sched.chunk)
          for (OmpInd i = 0; i < length; ++i) {
            exc.Run(callback, i);
          }
        }
        break;
      }
      case Sched::kStatic: {
        if (sched.chunk == 0) {
#pragma omp parallel for num_threads(n_threads) schedule(static)
          for (OmpInd i = 0; i < length; ++i) {
            exc.Run(callback, i);
          }
        } else {
#pragma omp parallel for num_threads(n_threads) schedule(static, sched.chunk)
          for (OmpInd i = 0; i < length; ++i) {
            exc.Run(callback, i);
          }
        }
        break;
      }
      case Sched::kGuided: {
#pragma omp parallel for num_threads(n_threads) schedule(guided)
        for (OmpInd i = 0; i < length; ++i) {
          exc.Run(callback, i);
        }
        break;
      }
    }
    exc.Rethrow();
  }

  // Split the blocks into one contiguous chunk per requested thread. The chunks assume a full
  // team, so a smaller one (OMP_DYNAMIC, a thread limit, or a nested region with nesting off) is
  // an error rather than a silent skip of the remaining chunks.
  template <typename Space, typename Func>
  static void Execute2d(Space const& space, std::int32_t n_threads, Func&& fn) {
    std::size_t n_blocks_in_space = space.Size();
    CHECK_GE(n_threads, 1);
#if !defined(_OPENMP)
    n_threads = 1;
#endif

    dmlc::OMPException exc;
#pragma omp parallel num_threads(n_threads)
    {
      exc.Run([&]() {
        std::size_t tid = omp_get_thread_num();
        auto team = omp_get_num_threads();
        if (team != n_threads) {
          // Every thread sees the same team size; one reports it and none runs a chunk.
          if (tid == 0) {
            LOG(FATAL) << "OpenMP started " << team << " of the " << n_threads
                       << " requested threads. The work is split by the requested count, so a "
                          "smaller team would skip part of it. OMP_DYNAMIC=true, a thread limit "
                          "or a nested parallel region can cause this; turn them off, or set "
                          "nthread=1.";
          }
          return;
        }
        auto&& callback =
            detail::BindWorker(fn, [tid] { return Worker{static_cast<std::int32_t>(tid)}; });
        std::size_t chunck_size = n_blocks_in_space / n_threads + !!(n_blocks_in_space % n_threads);

        std::size_t begin = chunck_size * tid;
        std::size_t end = std::min(begin + chunck_size, n_blocks_in_space);
        for (auto i = begin; i < end; i++) {
          callback(space.GetFirstDimension(i), space.GetRange(i));
        }
      });
    }
    exc.Rethrow();
  }
};
}  // namespace xgboost::common
