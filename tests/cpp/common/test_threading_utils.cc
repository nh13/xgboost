/**
 * Copyright 2019-2026, XGBoost Contributors
 */
#include <dmlc/omp.h>  // for omp_get_thread_num, omp_get_max_threads, omp_set_max_active_levels
#include <gtest/gtest.h>

#include <array>      // for array
#include <atomic>     // for atomic
#include <chrono>     // for milliseconds
#include <cstddef>    // for size_t
#include <memory>     // for make_shared, make_unique, shared_ptr
#include <stdexcept>  // for runtime_error
#include <string>     // for string
#include <thread>     // for sleep_for, thread
#include <vector>     // for vector

#include "../../../src/common/threading_utils.h"  // BlockedSpace2d,ParallelFor2d,ParallelFor
#include "../../../src/common/utils.h"            // for MakeCleanup
#include "xgboost/global_config.h"                // for GlobalConfigThreadLocalStore

namespace xgboost::common {
TEST(ParallelFor2d, CreateBlockedSpace2d) {
  constexpr size_t kDim1 = 5;
  constexpr size_t kDim2 = 3;
  constexpr size_t kGrainSize = 1;

  BlockedSpace2d space(kDim1, [&](size_t) { return kDim2; }, kGrainSize);

  ASSERT_EQ(kDim1 * kDim2, space.Size());

  for (size_t i = 0; i < kDim1; i++) {
    for (size_t j = 0; j < kDim2; j++) {
      ASSERT_EQ(space.GetFirstDimension(i * kDim2 + j), i);
      ASSERT_EQ(j, space.GetRange(i * kDim2 + j).begin());
      ASSERT_EQ(j + kGrainSize, space.GetRange(i * kDim2 + j).end());
    }
  }
}

TEST(ParallelFor2d, FewerThreadsThanRequested) {
#if defined(_OPENMP) && !defined(_MSC_VER)
  // With nesting off, a region inside another one starts one thread although four are requested.
  // The blocks are split for a full team, so ParallelFor2d raises instead of skipping blocks.
  auto const max_levels = omp_get_max_active_levels();
  auto restore = MakeCleanup([&] { omp_set_max_active_levels(max_levels); });
  omp_set_max_active_levels(1);
  BlockedSpace2d space(7, [](std::size_t) { return std::size_t{1}; }, 1);
  std::atomic<int> n_outer{0}, n_errors{0}, n_visits{0};
  std::string message;
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    n_outer = omp_get_num_threads();
    try {
      ParallelFor2d(space, 4, [&](std::size_t, Range1d) { ++n_visits; });
    } catch (dmlc::Error const& e) {
      if (++n_errors == 1) {
        message = e.what();
      }
    }
  }
  if (n_outer < 2) {
    GTEST_SKIP() << "OpenMP didn't start the outer team.";
  }
  EXPECT_EQ(n_errors.load(), 2);
  EXPECT_EQ(n_visits.load(), 0);
  EXPECT_NE(message.find("OpenMP started 1 of the 4 requested threads"), std::string::npos)
      << message;
  EXPECT_NE(message.find("nested parallel region"), std::string::npos) << message;
#else
  GTEST_SKIP() << "Needs OpenMP 3.0 (omp_set_max_active_levels), which MSVC lacks.";
#endif
}

TEST(ParallelFor2d, Coverage) {
  std::vector<std::size_t> sizes{0, 1, 5, 8, 19};
  BlockedSpace2d space(sizes.size(), [&](auto i) { return sizes[i]; }, 3);
  for (auto n_threads : {1, 4}) {
    std::vector<std::vector<int>> visits;
    for (auto size : sizes) {
      visits.emplace_back(size, 0);
    }
    auto visit = [&](std::size_t i, Range1d range) {
      for (auto j = range.begin(); j < range.end(); ++j) {
        ++visits[i][j];
      }
    };
    if (n_threads == 1) {
      ParallelFor2d(space, n_threads, visit);
    } else {
      try {
        ParallelFor2d(space, n_threads,
                      WithWorker([&, limit = std::make_unique<int>(n_threads)](
                                     std::size_t i, Range1d range, Worker worker) {
                        ASSERT_GE(worker.Id(), 0);
                        ASSERT_LT(worker.Id(), *limit);
                        visit(i, range);
                      }));
      } catch (dmlc::Error const& e) {
        // ParallelFor2d raises when OpenMP starts fewer threads (OMP_DYNAMIC, a thread limit);
        // FewerThreadsThanRequested covers that. With a full team, or for any other error, it
        // must not raise.
#if defined(_OPENMP)
        bool short_team = std::string(e.what()).find("requested threads") != std::string::npos;
        int team = 0;
#pragma omp parallel num_threads(n_threads)
        {
#pragma omp master
          team = omp_get_num_threads();
        }
        if (short_team && (team < n_threads || omp_get_dynamic())) {
          GTEST_SKIP() << "OpenMP didn't start a full team: " << e.what();
        }
#endif
        FAIL() << e.what();
      }
    }
    for (auto const& row : visits) {
      for (auto count : row) {
        EXPECT_EQ(count, 1);
      }
    }
  }
}

TEST(ParallelFor, WithWorker) {
  for (auto n_threads : {1, 4}) {
    // Owning the thread limit makes the callback move-only.
    auto make_callback = [n_threads] {
      return WithWorker([limit = std::make_unique<int>(n_threads)](auto, Worker worker) {
#if defined(_OPENMP)
        EXPECT_EQ(worker.Id(), omp_get_thread_num());
#endif
        EXPECT_GE(worker.Id(), 0);
        EXPECT_LT(worker.Id(), *limit);
      });
    };
    ParallelFor(std::size_t{17}, n_threads, make_callback());
    ParallelFor1d<3>(17, n_threads, make_callback());
    ParallelForBlock(17, n_threads, make_callback());
  }
  // Nested serial loops report worker ID 0, not the outer thread ID.
  ParallelFor(4, 4, [](auto) {
    ParallelFor(1, 1, WithWorker([](auto, Worker inner) { EXPECT_EQ(inner.Id(), 0); }));
  });
}

TEST(ParallelFor, Schedules) {
  for (auto sched : {Sched::Auto(), Sched::Static(), Sched::Static(2), Sched::Dyn(), Sched::Dyn(2),
                     Sched::Guided()}) {
    for (int size : {0, 1, 19}) {
      std::array<std::atomic<int>, 19> visits{};
      ParallelFor(size, 4, sched, [&](auto i) { ++visits[i]; });
      for (std::size_t i = 0; i < visits.size(); ++i) {
        EXPECT_EQ(visits[i].load(), i < static_cast<std::size_t>(size) ? 1 : 0);
      }
    }
  }
}

TEST(ParallelFor, ThreadsCappedBySize) {
  // A loop never uses more workers than it has iterations.
  for (auto sched : {Sched::Auto(), Sched::Static(), Sched::Dyn(), Sched::Guided()}) {
    for (int size : {1, 2, 3}) {
      ParallelFor(size, 4, sched, WithWorker([&](auto, Worker worker) {
                    EXPECT_LT(worker.Id(), size);
#if defined(_OPENMP)
                    EXPECT_LE(omp_get_num_threads(), size);
#endif
                  }));
    }
  }
}

TEST(ParallelFor, ExceptionDrainsTasks) {
  bool finished{false};
  EXPECT_THROW(ParallelFor(2, 2,
                           [&](auto i) {
                             if (i == 0) {
                               throw std::runtime_error{"task failed"};
                             }
                             std::this_thread::sleep_for(std::chrono::milliseconds{10});
                             finished = true;
                           }),
               std::runtime_error);
  EXPECT_TRUE(finished);
}

TEST(GetNumThreads, ConfiguredLimit) {
#if defined(_OPENMP)
  auto previous = omp_get_max_threads();
  omp_set_num_threads(2);
#else
  auto* config = GlobalConfigThreadLocalStore::Get();
  auto previous = config->nthread;
  config->nthread = 2;
#endif
  EXPECT_LE(GetNumThreads(0), 2);
  EXPECT_EQ(GetNumThreads(8), GetNumThreads(0));
  EXPECT_EQ(GetNumThreads(1), 1);
#if defined(_OPENMP)
  ParallelFor(2, 2, [](auto) { EXPECT_EQ(GetNumThreads(0), 1); });
  omp_set_num_threads(previous);
#else
  config->nthread = previous;
#endif
}

TEST(GetNumThreads, NoActiveLevels) {
#if defined(_OPENMP) && !defined(_MSC_VER)
  // With no active parallel level allowed, every region runs on one thread, so XGBoost uses one
  // thread instead of asking ParallelFor2d for a team it can't get.
  if (GetNumThreads(0) < 2) {
    GTEST_SKIP() << "Needs more than one thread.";
  }
  auto previous = omp_get_max_active_levels();
  auto restore = MakeCleanup([&] { omp_set_max_active_levels(previous); });
  omp_set_max_active_levels(0);
  auto n_threads = GetNumThreads(0);
  auto n_capped = GetNumThreads(8);
  BlockedSpace2d space(3, [](std::size_t) { return std::size_t{1}; }, 1);
  std::array<std::atomic<int>, 3> visits{};
  ParallelFor2d(space, n_threads, [&](std::size_t i, Range1d) { ++visits[i]; });
  EXPECT_EQ(n_threads, 1);
  EXPECT_EQ(n_capped, 1);
  for (auto const& count : visits) {
    EXPECT_EQ(count.load(), 1);
  }
#else
  GTEST_SKIP() << "Needs OpenMP 3.0 (omp_set_max_active_levels), which MSVC lacks.";
#endif
}

TEST(MemStackAllocator, ObjectLifetime) {
  auto value = std::make_shared<int>(7);
  for (std::size_t n : {0, 2, 3}) {
    {
      MemStackAllocator<std::shared_ptr<int>, 2> storage(n, value);
      ASSERT_EQ(value.use_count(), n + 1);
      for (std::size_t i = 0; i < n; ++i) {
        ASSERT_EQ(storage[i], value);
      }
    }
    ASSERT_EQ(value.use_count(), 1);
  }
}
}  // namespace xgboost::common
