/**
 * Copyright 2024-2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/global_config.h>  // for GlobalConfigThreadLocalStore

#include <algorithm>  // for min
#include <atomic>     // for atomic
#include <chrono>     // for milliseconds, microseconds
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t
#include <future>     // for async, future, launch, promise
#include <stdexcept>  // for runtime_error
#include <string>     // for string, to_string
#include <thread>     // for this_thread, thread
#include <vector>     // for vector

#if !defined(_WIN32)
#include <sys/wait.h>  // for waitpid, WIFEXITED, WEXITSTATUS
#include <unistd.h>    // for fork, alarm, _exit
#endif

#include "../../../src/common/threadpool.h"

namespace xgboost::common {
namespace {
using Exec = GlobalThreadPoolExecutor;

// Run one loop and check that every iteration ran once, on the worker that owns its block.
void CheckLoop(std::size_t size, std::int32_t n_threads, Sched sched) {
  std::vector<std::atomic<int>> visits(size);
  std::vector<std::int32_t> owner(size, -1);
  Exec::Execute(size, n_threads, sched, WithWorker{[&](std::size_t i, Worker worker) {
                  ++visits[i];
                  owner[i] = worker.Id();
                }});
  auto const n = static_cast<std::size_t>(std::min<std::size_t>(n_threads, size));
  auto const block = size / n + (size % n != 0);
  for (std::size_t i = 0; i < size; ++i) {
    ASSERT_EQ(visits[i].load(), 1) << i;
    // The executor documents this mapping: block k of ceil(size / n) iterations runs as worker k.
    ASSERT_EQ(owner[i], static_cast<std::int32_t>(i / block)) << i;
  }
}

// Thread ids of the workers, indexed by worker id.
std::vector<std::thread::id> WorkerThreads(std::int32_t n_threads) {
  std::vector<std::thread::id> ids(n_threads);
  Exec::Execute(n_threads, n_threads, Sched::Static(),
                WithWorker{[&](int, Worker w) { ids[w.Id()] = std::this_thread::get_id(); }});
  return ids;
}

// Restore the default worker budget and drop the teams a test created.
struct ResetTeams {
  ~ResetTeams() {
    detail::SetTeamWorkerBudget(0);
    detail::DestroyIdleTeams();
  }
};
}  // namespace

TEST(ThreadPool, ExecutorCoverage) {
  // The scheduling hint is ignored, so one is enough. Sizes 9 and 10 leave trailing blocks empty
  // with 4 and 8 threads, and the thread counts both grow and shrink the team.
  for (std::size_t size : {1, 2, 3, 7, 9, 10, 64, 1000}) {
    for (std::int32_t n_threads : {1, 2, 4, 8, 3, 16}) {
      CheckLoop(size, n_threads, Sched::Static());
    }
  }
}

TEST(ThreadPool, ExecutorKeepsWorkersOnThreads) {
  // Worker 0 is the calling thread, and worker t runs on the same thread from one loop to the next.
  auto first = WorkerThreads(4);
  auto second = WorkerThreads(4);
  EXPECT_EQ(first[0], std::this_thread::get_id());
  EXPECT_NE(first[1], first[0]);
  EXPECT_EQ(first, second);
}

TEST(ThreadPool, ExecutorException) {
  // A failure on the calling thread or on a worker propagates after every block has finished.
  for (std::int32_t failing : {0, 3}) {
    std::atomic<int> finished{0};
    EXPECT_THROW(Exec::Execute(4, 4, Sched::Static(),
                               [&](int i) {
                                 if (i == failing) {
                                   throw std::runtime_error{"task failed"};
                                 }
                                 std::this_thread::sleep_for(std::chrono::milliseconds{5});
                                 ++finished;
                               }),
                 std::runtime_error);
    EXPECT_EQ(finished.load(), 3);
  }
  // When every block throws, the lowest block's exception is the one reported.
  for (int rep = 0; rep < 20; ++rep) {
    try {
      Exec::Execute(4, 4, Sched::Static(),
                    [](int i) { throw std::runtime_error{"block " + std::to_string(i)}; });
      FAIL() << "No exception.";
    } catch (std::runtime_error const& e) {
      EXPECT_EQ(std::string{e.what()}, "block 0");
    }
  }
  // The team went back to the registry: the executor still works on the same threads.
  auto before = WorkerThreads(4);
  EXPECT_THROW(Exec::Execute(4, 4, Sched::Static(),
                             [](int i) {
                               if (i == 2) {
                                 throw std::runtime_error{"task failed"};
                               }
                             }),
               std::runtime_error);
  EXPECT_EQ(WorkerThreads(4), before);
  CheckLoop(16, 4, Sched::Static());
}

TEST(ThreadPool, ExecutorSleepingThreads) {
  // Waits long enough for idle workers and the waiting caller to fall asleep.
  for (int rep = 0; rep < 3; ++rep) {
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    CheckLoop(8, 4, Sched::Static());
  }
  // The calling thread finishes first and sleeps until the slow worker is done.
  std::atomic<int> done{0};
  Exec::Execute(2, 2, Sched::Static(), [&](int i) {
    if (i == 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds{5});
    }
    ++done;
  });
  EXPECT_EQ(done.load(), 2);
}

TEST(ThreadPool, ExecutorSpinToSleepHandoff) {
  // Gaps around the 500 us spin time land dispatches and completions next to the moment a worker
  // or the waiting caller gives up spinning and goes to sleep.
  for (int rep = 0; rep < 300; ++rep) {
    auto const gap = std::chrono::microseconds{300 + (rep * 37) % 400};
    std::this_thread::sleep_for(gap);
    std::atomic<int> done{0};
    Exec::Execute(4, 4, Sched::Static(), [&](int i) {
      if (i == rep % 4) {
        std::this_thread::sleep_for(gap);
      }
      ++done;
    });
    ASSERT_EQ(done.load(), 4) << rep;
  }
}

TEST(ThreadPool, ExecutorWorkerBudget) {
  ResetTeams reset;
  detail::DestroyIdleTeams();
  ASSERT_EQ(detail::NumTeamWorkers(), 0);
  detail::SetTeamWorkerBudget(4);
  // One call may go over the budget.
  CheckLoop(64, 8, Sched::Static());
  EXPECT_EQ(detail::NumTeamWorkers(), 7);
  // Concurrent callers share the budget; a caller that gets a small team or none still sees every
  // iteration once with the right worker id.
  detail::DestroyIdleTeams();
  std::vector<std::future<void>> callers;
  for (std::int32_t c = 0; c < 4; ++c) {
    callers.emplace_back(std::async(std::launch::async, [] {
      for (int rep = 0; rep < 100; ++rep) {
        CheckLoop(37, 3, Sched::Static());
      }
    }));
  }
  for (auto& caller : callers) {
    caller.get();
  }
  EXPECT_LE(detail::NumTeamWorkers(), 4);
}

TEST(ThreadPool, ExecutorSmallTeam) {
  ResetTeams reset;
  detail::DestroyIdleTeams();
  detail::SetTeamWorkerBudget(4);
  // Another thread holds a team of 3 workers, so this caller's team only gets 1 of the 3 workers
  // it asks for, and each of the 2 team threads runs 2 of the 4 blocks.
  std::promise<void> holding, release;
  auto other = std::async(std::launch::async, [&] {
    Exec::Execute(4, 4, Sched::Static(), [&](int i) {
      if (i == 0) {
        holding.set_value();
        release.get_future().wait();
      }
    });
  });
  holding.get_future().wait();
  CheckLoop(64, 4, Sched::Static());
  auto ids = WorkerThreads(4);
  release.set_value();
  other.get();
  EXPECT_EQ(detail::NumTeamWorkers(), 4);
  EXPECT_EQ(ids[0], std::this_thread::get_id());
  EXPECT_EQ(ids[2], ids[0]);
  EXPECT_NE(ids[1], ids[0]);
  EXPECT_EQ(ids[3], ids[1]);
}

TEST(ThreadPool, ExecutorDestroyTeams) {
  ResetTeams reset;
  // Destroy teams whose workers are asleep and teams whose workers are still spinning.
  for (bool asleep : {true, false}) {
    CheckLoop(64, 4, Sched::Static());
    if (asleep) {
      std::this_thread::sleep_for(std::chrono::milliseconds{5});
    }
    detail::DestroyIdleTeams();
    EXPECT_EQ(detail::NumTeamWorkers(), 0);
  }
  // New teams start afterwards.
  CheckLoop(64, 4, Sched::Static());
  EXPECT_EQ(detail::NumTeamWorkers(), 3);
}

TEST(ThreadPool, ExecutorWorkerConfig) {
  ResetTeams reset;
  detail::DestroyIdleTeams();
  // Workers take the global configuration of the thread that created their team.
  auto* config = GlobalConfigThreadLocalStore::Get();
  auto const original = config->verbosity;
  config->verbosity = 3;
  std::vector<int> seen(4, -1);
  Exec::Execute(4, 4, Sched::Static(), WithWorker{[&](int, Worker w) {
                  seen[w.Id()] = GlobalConfigThreadLocalStore::Get()->verbosity;
                }});
  config->verbosity = original;
  EXPECT_EQ(seen, (std::vector<int>{3, 3, 3, 3}));
}

#if !defined(_WIN32)
TEST(ThreadPool, ExecutorAfterFork) {
#if defined(__SANITIZE_THREAD__)
  GTEST_SKIP() << "ThreadSanitizer can't start threads in a forked child.";
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
  GTEST_SKIP() << "ThreadSanitizer can't start threads in a forked child.";
#endif
#endif
  // The child gets new teams instead of waiting on workers that don't exist in it.
  CheckLoop(64, 4, Sched::Static());
  auto pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    alarm(30);  // A hang ends the child with SIGALRM.
    std::atomic<int> visits{0};
    Exec::Execute(64, 4, Sched::Static(), [&](int) { ++visits; });
    _exit(visits.load() == 64 ? 0 : 1);
  }
  int status{0};
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status)) << "The child did not exit normally: " << status;
  EXPECT_EQ(WEXITSTATUS(status), 0);
}
#endif

TEST(ThreadPool, ExecutorConcurrentCallers) {
  // Several threads run loops at the same time; each checks out its own team.
  std::vector<std::future<void>> callers;
  for (std::int32_t c = 0; c < 4; ++c) {
    callers.emplace_back(std::async(std::launch::async, [c] {
      for (int rep = 0; rep < 200; ++rep) {
        CheckLoop(37, 2 + c, Sched::Static());
      }
    }));
  }
  for (auto& caller : callers) {
    caller.get();
  }
}

TEST(ThreadPool, NestedExecutor) {
  auto const n_threads = Exec::MaxNumWorkers();
  std::vector<int> counts(2);
  Exec::Execute(2, 2, Sched::Static(), [&](int i) {
    ASSERT_EQ(Exec::MaxNumWorkers(), 1);
    auto const worker = std::this_thread::get_id();
    // Explicit thread counts must also run inline inside an executor task.
    Exec::Execute(3, 2, Sched::Static(), WithWorker{[&](int, Worker inner) {
                    EXPECT_EQ(std::this_thread::get_id(), worker);
                    EXPECT_EQ(inner.Id(), 0);
                    ++counts[i];
                  }});
    EXPECT_THROW(Exec::Execute(2, 2, Sched::Static(),
                               [](int) { throw std::runtime_error{"nested task failed"}; }),
                 std::runtime_error);
    EXPECT_EQ(Exec::MaxNumWorkers(), 1);
  });
  EXPECT_EQ(counts, (std::vector<int>{3, 3}));
  EXPECT_EQ(Exec::MaxNumWorkers(), n_threads);
}

TEST(ThreadPool, TaskException) {
  ThreadPool pool{"test", 1, [] {}};
  auto void_result = pool.Submit([] { throw std::runtime_error{"task failed"}; });
  EXPECT_THROW(void_result.get(), std::runtime_error);
  auto value_result = pool.Submit([]() -> int { throw std::runtime_error{"task failed"}; });
  EXPECT_THROW(value_result.get(), std::runtime_error);
  // The same worker can still execute tasks after an exception.
  EXPECT_EQ(pool.Submit([] { return 42; }).get(), 42);
}

TEST(ThreadPool, Basic) {
  auto config = *GlobalConfigThreadLocalStore::Get();
  auto original = config.verbosity;
  config.verbosity = 4;
  ThreadPool pool{"test", 2, [config] { *GlobalConfigThreadLocalStore::Get() = config; }};
  EXPECT_EQ(pool.Submit([] { return GlobalConfigThreadLocalStore::Get()->verbosity; }).get(), 4);
  EXPECT_EQ(GlobalConfigThreadLocalStore::Get()->verbosity, original);

  std::vector<std::future<std::size_t>> futures;
  for (std::size_t i = 0; i < 32; ++i) {
    futures.emplace_back(pool.Submit([i] { return i; }));
  }
  for (std::size_t i = 0; i < futures.size(); ++i) {
    EXPECT_EQ(futures[i].get(), i);
  }

  int value{0};
  pool.Submit([&] { value = 3; }).get();
  EXPECT_EQ(value, 3);
}
}  // namespace xgboost::common
