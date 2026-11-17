/**
 * Copyright 2024-2026, XGBoost Contributors
 */
#pragma once
#include <algorithm>           // for max, min
#include <atomic>              // for atomic
#include <condition_variable>  // for condition_variable
#include <cstddef>             // for size_t
#include <cstdint>             // for int32_t
#include <exception>           // for current_exception, exception_ptr, rethrow_exception
#include <functional>          // for function
#include <future>              // for future, promise
#include <memory>              // for make_shared
#include <mutex>               // for mutex, unique_lock
#include <queue>               // for queue
#include <string>              // for string
#include <thread>              // for thread
#include <type_traits>         // for invoke_result_t, is_void_v
#include <utility>             // for forward, move
#include <vector>              // for vector

#include "executor.h"             // for Sched, Worker
#include "utils.h"                // for MakeCleanup
#include "xgboost/logging.h"      // for CHECK_GE
#include "xgboost/string_view.h"  // for StringView

namespace xgboost::common {
/**
 * @brief Simple implementation of a thread pool.
 */
class ThreadPool {
  std::string name_;
  std::function<void()> worker_fn_;
  std::mutex mu_;
  std::queue<std::function<void()>> tasks_;
  std::condition_variable cv_;
  std::vector<std::thread> pool_;
  bool stop_{false};

  void Start(std::int32_t n_threads);

 public:
  /**
   * @param name      Name prefix for threads.
   * @param n_threads The number of threads this pool should hold.
   * @param init_fn   Copyable function copied to and called once on each worker.
   */
  template <typename InitFn>
  explicit ThreadPool(StringView name, std::int32_t n_threads, InitFn&& init_fn)
      : name_{static_cast<std::string>(name)}, worker_fn_{[this, init_fn] {
          init_fn();

          while (true) {
            std::unique_lock lock{mu_};
            cv_.wait(lock, [this] { return !this->tasks_.empty() || stop_; });

            if (this->stop_) {
              // Drain
              while (!tasks_.empty()) {
                auto fn = tasks_.front();
                tasks_.pop();
                fn();
              }
              return;
            }

            auto fn = tasks_.front();
            tasks_.pop();
            lock.unlock();
            fn();
          }
        }} {
    this->Start(n_threads);
  }

  ~ThreadPool() {
    std::unique_lock lock{mu_};
    stop_ = true;
    lock.unlock();

    for (auto& t : pool_) {
      if (t.joinable()) {
        std::unique_lock lock{mu_};
        this->cv_.notify_one();
        lock.unlock();
      }
    }

    for (auto& t : pool_) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  /**
   * @brief Submit a function that doesn't take any argument.
   */
  template <typename Fn, typename R = std::invoke_result_t<Fn>>
  auto Submit(Fn&& fn) {
    // Use shared ptr to make the task copy constructible.
    auto p{std::make_shared<std::promise<R>>()};
    auto fut = p->get_future();

    auto ffn = std::function{[task = std::move(p), fn = std::forward<Fn>(fn)]() mutable {
      try {
        if constexpr (std::is_void_v<R>) {
          fn();
          task->set_value();
        } else {
          task->set_value(fn());
        }
      } catch (...) {
        task->set_exception(std::current_exception());
      }
    }};

    std::unique_lock lock{mu_};
    this->tasks_.push(std::move(ffn));
    lock.unlock();

    cv_.notify_one();
    return fut;
  }

  [[nodiscard]] auto NumWorkers() const {
    // Use the same type as the typical n_threads parameter in XGBoost.
    return static_cast<std::int32_t>(pool_.size());
  }
};

namespace detail {
/**
 * @brief One parallel loop run by @ref RunOnTeam. Team member `m` runs blocks m, m + stride, ...
 *        where stride is the team size, and the calling thread is member 0.
 */
struct TeamJob {
  void (*run)(void const* ctx, std::int32_t t);
  void const* ctx;
  std::int32_t n_blocks{0};
  std::int32_t stride{1};
  // Workers decrement this as they finish while the caller polls it, so keep it away from the
  // read-only fields above.
  alignas(64) std::atomic<std::int32_t> remaining{0};
  std::mutex error_mu;
  std::int32_t error_block{-1};  // Guarded by error_mu.
  std::exception_ptr error;      // Guarded by error_mu.

  // Run the blocks of team member `member`.
  void RunMember(std::int32_t member) noexcept {
    for (auto t = member; t < n_blocks; t += stride) {
      this->Run(t);
    }
  }
  // Run block `t`. When several blocks throw, keep the exception from the lowest block so the
  // reported error does not depend on timing.
  void Run(std::int32_t t) noexcept {
    try {
      run(ctx, t);
    } catch (...) {
      std::lock_guard lock{error_mu};
      if (error_block < 0 || t < error_block) {
        error_block = t;
        error = std::current_exception();
      }
    }
  }
};

class ThreadTeam;

/**
 * @brief Run blocks [0, n_blocks) of `job` and wait for all of them. Block 0 runs on the calling
 *        thread and block `k` on worker `k` of a team checked out for this call.
 *
 * Each concurrent caller gets its own team, and teams are kept for reuse until the process exits.
 * Workers take the thread-local global configuration of the thread that created their team. The
 * total number of team workers is capped (see @ref SetTeamWorkerBudget). When the cap, or a
 * failure to start a thread, leaves the team with fewer than n_blocks - 1 workers, each member
 * runs several blocks in turn; with no workers at all, the calling thread runs every block.
 */
void RunOnTeam(TeamJob* job, std::int32_t n_blocks);

/**
 * @brief Set the cap on team workers summed over all teams. 0 restores the default, twice the
 *        number of hardware threads. A call that asks for more workers than the cap may still bring the
 *        total up to its own n_blocks - 1.
 */
void SetTeamWorkerBudget(std::int32_t n_workers);
/** @brief The number of team workers started and not yet destroyed. */
[[nodiscard]] std::int32_t NumTeamWorkers();
/** @brief Join and destroy every idle team. Teams in use are kept. */
void DestroyIdleTeams();
}  // namespace detail

struct GlobalThreadPoolExecutor {
 private:
  inline static thread_local bool in_parallel_{false};
  friend class detail::ThreadTeam;

 public:
  [[nodiscard]] static std::int32_t MaxNumWorkers() noexcept;

  /**
   * @brief Run `fn` over [0, size) on `n_threads` threads.
   *
   * The scheduling hint is ignored. The range is split into contiguous blocks of
   * ceil(size / n_threads) iterations, and block `k` runs as worker `k`: on the calling thread for
   * `k == 0` and otherwise on the same team thread from one loop to the next, so per-worker
   * buffers stay on one thread. When the worker cap leaves fewer threads than blocks, a thread
   * runs several blocks one after another (see detail::RunOnTeam). Loops nested inside a block
   * run serially.
   */
  template <typename Index, typename Func>
  static void Execute(Index size, std::int32_t n_threads, Sched, Func&& fn) {
    CHECK_GE(n_threads, 1);
    if (size <= 0) {
      return;
    }
    n_threads = in_parallel_ ? 1 : std::min<std::size_t>(n_threads, size);
    auto const length = static_cast<std::size_t>(size);
    auto const block_size = length / n_threads + (length % n_threads != 0);
    auto run = [&](std::int32_t t) {
      auto&& callback = detail::BindWorker(fn, [t] { return Worker{t}; });
      auto const begin = t * block_size;
      auto const end = std::min(begin + block_size, length);
      for (auto i = begin; i < end; ++i) {
        callback(static_cast<Index>(i));
      }
    };
    if (n_threads == 1) {
      run(0);
      return;
    }

    using Run = decltype(run);
    detail::TeamJob job{[](void const* ctx, std::int32_t t) { (*static_cast<Run const*>(ctx))(t); },
                        &run};
    // The calling thread runs block 0; nested loops inside it run serially, as on the workers.
    in_parallel_ = true;
    {
      auto reset = common::MakeCleanup([] { in_parallel_ = false; });
      detail::RunOnTeam(&job, n_threads);
    }
    if (job.error) {
      std::rethrow_exception(job.error);
    }
  }

  template <typename Space, typename Func>
  static void Execute2d(Space const& space, std::int32_t n_threads, Func&& fn) {
    Execute(space.Size(), n_threads, Sched::Static(),
            detail::TransformCallback(fn, [&](auto& callback, auto i) {
              callback(space.GetFirstDimension(i), space.GetRange(i));
            }));
  }
};
}  // namespace xgboost::common
