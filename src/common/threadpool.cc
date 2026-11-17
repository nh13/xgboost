/**
 * Copyright 2026, XGBoost Contributors
 */
#include "threadpool.h"

#include <algorithm>           // for clamp, find, find_if, max, min
#include <atomic>              // for atomic
#include <chrono>              // for steady_clock, microseconds
#include <condition_variable>  // for condition_variable
#include <cstdint>             // for int32_t
#include <memory>              // for make_unique, unique_ptr
#include <mutex>               // for mutex, unique_lock, lock_guard
#include <string>              // for string, to_string
#include <system_error>        // for system_error
#include <thread>              // for thread, yield
#include <utility>             // for move
#include <vector>              // for vector

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>  // for _mm_pause
#endif

#if !defined(_WIN32)
#include <pthread.h>  // for pthread_atfork
#endif

#include "threading_utils.h"        // for NameThread
#include "xgboost/global_config.h"  // for GlobalConfigThreadLocalStore, InitNewThread
#include "xgboost/logging.h"        // for CHECK_GE, LOG

namespace xgboost::common {
std::int32_t GlobalThreadPoolExecutor::MaxNumWorkers() noexcept {
  if (in_parallel_) {
    return 1;
  }
  static auto max_n_threads = std::thread::hardware_concurrency();
  auto limit = std::max(1u, max_n_threads);
  auto configured = GlobalConfigThreadLocalStore::Get()->nthread;
  return configured > 0 ? std::min(limit, static_cast<unsigned>(configured)) : limit;
}

void ThreadPool::Start(std::int32_t n_threads) {
  for (auto i = static_cast<std::int32_t>(pool_.size()); i < n_threads; ++i) {
    pool_.emplace_back(worker_fn_);
    std::string name_i = name_ + "-" + std::to_string(i);
    NameThread(&pool_.back(), name_i);
  }
}

namespace detail {
namespace {
// How long an idle thread spins before it sleeps. Training and prediction often run parallel
// loops separated by a few hundred microseconds of serial work, and spinning through that gap
// saves a system call to wake each worker for the next loop. The cost is that an idle worker
// keeps a core busy for up to this long after each loop, which the worker cap bounds.
constexpr std::chrono::microseconds kSpinTime{500};

void CpuRelax() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  _mm_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#else
  std::this_thread::yield();
#endif
}

// Spin until `ready()` holds or kSpinTime has passed. Returns whether `ready()` holds.
template <typename Ready>
bool SpinUntil(Ready&& ready) {
  if (ready()) {
    return true;
  }
  auto const deadline = std::chrono::steady_clock::now() + kSpinTime;
  while (true) {
    for (std::int32_t i = 0; i < 64; ++i) {
      CpuRelax();
      if (ready()) {
        return true;
      }
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return ready();
    }
  }
}
}  // namespace

/**
 * @brief Persistent worker threads that run parallel loops for one calling thread at a time.
 *
 * Worker `k` always runs block `k` of a loop. Each worker spins briefly on its own slot before
 * sleeping. Workers mark themselves as inside a parallel loop for good, so loops nested in a
 * block run serially on them.
 */
class ThreadTeam {
 public:
  ThreadTeam() = default;
  ~ThreadTeam();
  ThreadTeam(ThreadTeam const&) = delete;
  ThreadTeam& operator=(ThreadTeam const&) = delete;

  [[nodiscard]] std::int32_t NumWorkers() const { return static_cast<std::int32_t>(slots_.size()); }
  // Start workers until there are `n_workers`. Throws std::system_error when a thread cannot be
  // started, keeping the workers started so far.
  void EnsureWorkers(std::int32_t n_workers);
  // Run `job` on the calling thread and min(NumWorkers(), n_blocks - 1) workers, then wait for
  // all of them. Requires NumWorkers() >= 1.
  void Run(TeamJob* job, std::int32_t n_blocks);

 private:
  struct Slot;
  void Work(Slot* slot, std::int32_t t);

  InitNewThread init_;
  std::vector<std::unique_ptr<Slot>> slots_;
  std::vector<std::thread> workers_;
  std::mutex done_mu_;
  std::condition_variable done_cv_;
  std::atomic<bool> caller_waiting_{false};
};

struct alignas(64) ThreadTeam::Slot {
  std::atomic<TeamJob*> job{nullptr};
  std::atomic<bool> sleeping{false};
  bool stop{false};  // Guarded by mu.
  std::mutex mu;
  std::condition_variable cv;
};

ThreadTeam::~ThreadTeam() {
  for (auto& slot : slots_) {
    std::lock_guard lock{slot->mu};
    slot->stop = true;
    slot->cv.notify_one();
  }
  for (auto& worker : workers_) {
    worker.join();
  }
}

void ThreadTeam::EnsureWorkers(std::int32_t n_workers) {
  if (n_workers <= this->NumWorkers()) {
    return;
  }
  // Reserve first so that, once a thread is running, publishing its slot cannot throw.
  slots_.reserve(n_workers);
  workers_.reserve(n_workers);
  for (auto t = this->NumWorkers() + 1; t <= n_workers; ++t) {
    auto slot = std::make_unique<Slot>();
    std::thread worker{[this, s = slot.get(), t] { this->Work(s, t); }};
    slots_.push_back(std::move(slot));
    workers_.push_back(std::move(worker));
    NameThread(&workers_.back(), "te-" + std::to_string(t));
  }
}

void ThreadTeam::Work(Slot* slot, std::int32_t t) {
  init_();
  GlobalThreadPoolExecutor::in_parallel_ = true;
  // The sleep handshakes below store one flag and then load the other, on both sides, so they use
  // sequentially consistent operations.
  auto has_job = [slot] {
    return slot->job.load() != nullptr;
  };
  while (true) {
    if (!SpinUntil(has_job)) {
      std::unique_lock lock{slot->mu};
      slot->sleeping.store(true);
      slot->cv.wait(lock, [&] { return slot->stop || has_job(); });
      slot->sleeping.store(false);
      if (slot->stop && !has_job()) {
        return;
      }
    }
    auto* job = slot->job.exchange(nullptr);
    job->RunMember(t);
    // `job` lives on the caller's stack and can be gone once `remaining` reaches zero.
    if (job->remaining.fetch_sub(1) == 1 && caller_waiting_.load()) {
      std::lock_guard lock{done_mu_};
      done_cv_.notify_one();
    }
  }
}

void ThreadTeam::Run(TeamJob* job, std::int32_t n_blocks) {
  CHECK_GE(this->NumWorkers(), 1) << "The thread team has no workers.";
  auto const n_workers = std::min(this->NumWorkers(), n_blocks - 1);
  job->n_blocks = n_blocks;
  job->stride = n_workers + 1;
  job->remaining.store(n_workers, std::memory_order_relaxed);
  for (std::int32_t t = 1; t <= n_workers; ++t) {
    auto& slot = *slots_[t - 1];
    slot.job.store(job);
    if (slot.sleeping.load()) {
      std::lock_guard lock{slot.mu};
      slot.cv.notify_one();
    }
  }
  job->RunMember(0);
  auto done = [job] {
    return job->remaining.load() == 0;
  };
  if (!SpinUntil(done)) {
    std::unique_lock lock{done_mu_};
    caller_waiting_.store(true);
    done_cv_.wait(lock, done);
    caller_waiting_.store(false);
  }
}

namespace {
struct TeamRegistry {
  std::mutex mu;
  // Everything below is guarded by mu.
  std::vector<std::unique_ptr<ThreadTeam>> teams;
  std::vector<ThreadTeam*> idle;  // Capacity is kept at teams.size(), so pushing never allocates.
  std::int32_t n_workers{0};      // Workers started or reserved, summed over all teams.
  std::int32_t budget{0};         // 0 means the default, see Budget().

  TeamRegistry() {
#if !defined(_WIN32)
    // A forked child has only the forking thread, so its teams have no workers. Keep the registry
    // lock across fork() and give the child an empty registry.
    pthread_atfork([] { Registry().mu.lock(); }, [] { Registry().mu.unlock(); },
                   [] { Registry().ResetInChild(); });
#endif
  }

  [[nodiscard]] std::int32_t Budget() const {
    if (budget > 0) {
      return budget;
    }
    // Twice the hardware threads: idle teams keep their workers, so a cap at the hardware thread
    // count would leave a second concurrent caller with few threads once one big team exists.
    // hardware_concurrency() is a system call on some platforms, so ask once.
    static auto const n_threads =
        static_cast<std::int32_t>(std::max(1u, std::thread::hardware_concurrency()));
    return 2 * n_threads;
  }

  void ResetInChild() {
    // The teams' threads don't exist here: joining them would fail, so leak the teams.
    for (auto& team : teams) {
      static_cast<void>(team.release());
    }
    teams.clear();
    idle.clear();
    n_workers = 0;
    mu.unlock();
  }

  static TeamRegistry& Registry() {
    static TeamRegistry registry;
    return registry;
  }
};

void ReleaseTeam(ThreadTeam* team) noexcept;

// The team this thread used last; reusing it keeps blocks on the same workers.
thread_local ThreadTeam* last_team{nullptr};

// Check out a team that no other thread is using, grown towards `n_workers` workers as far as
// the budget allows. Returns nullptr when that leaves it with no workers.
ThreadTeam* AcquireTeam(std::int32_t n_workers) {
  auto& registry = TeamRegistry::Registry();
  ThreadTeam* team{nullptr};
  std::int32_t target{0};
  {
    std::lock_guard lock{registry.mu};
    auto& idle = registry.idle;
    auto size = [](ThreadTeam* t) {
      return t->NumWorkers();
    };
    // Prefer this thread's last team when it is big enough, then the smallest team that is big
    // enough, then the biggest team.
    auto it = std::find(idle.begin(), idle.end(), last_team);
    if (it == idle.end() || size(*it) < n_workers) {
      auto best = idle.end();
      for (auto cand = idle.begin(); cand != idle.end(); ++cand) {
        if (best == idle.end()) {
          best = cand;
          continue;
        }
        bool fits = size(*cand) >= n_workers, best_fits = size(*best) >= n_workers;
        if ((fits && (!best_fits || size(*cand) < size(*best))) ||
            (!fits && !best_fits && size(*cand) > size(*best))) {
          best = cand;
        }
      }
      if (best != idle.end() && (it == idle.end() || size(*best) > size(*it))) {
        it = best;
      }
    }
    auto const have = it == idle.end() ? 0 : size(*it);
    // The cap is never below what one call asks for, so a single caller always gets a full team.
    auto const limit = std::max(registry.Budget(), n_workers);
    auto const grow = std::clamp(n_workers - have, 0, std::max(0, limit - registry.n_workers));
    target = have + grow;
    if (target == 0) {
      return nullptr;
    }
    if (it != idle.end()) {
      team = *it;
      idle.erase(it);
    } else {
      registry.teams.reserve(registry.teams.size() + 1);
      registry.idle.reserve(registry.teams.size() + 1);
      team = registry.teams.emplace_back(std::make_unique<ThreadTeam>()).get();
    }
    registry.n_workers += grow;
  }
  // Only the thread holding the team changes its workers.
  try {
    team->EnsureWorkers(target);
  } catch (std::system_error const& e) {
    {
      std::lock_guard lock{registry.mu};
      registry.n_workers -= target - team->NumWorkers();
    }
    static std::once_flag warned;
    std::call_once(warned, [&] {
      LOG(WARNING) << "Failed to start a worker thread (" << e.what()
                   << "); parallel loops will use fewer threads.";
    });
    if (team->NumWorkers() == 0) {
      ReleaseTeam(team);
      return nullptr;
    }
  }
  last_team = team;
  return team;
}

void ReleaseTeam(ThreadTeam* team) noexcept {
  auto& registry = TeamRegistry::Registry();
  std::lock_guard lock{registry.mu};
  registry.idle.push_back(team);
}
}  // namespace

void RunOnTeam(TeamJob* job, std::int32_t n_blocks) {
  auto* team = AcquireTeam(n_blocks - 1);
  if (team == nullptr) {
    job->n_blocks = n_blocks;
    job->stride = 1;
    job->RunMember(0);
    return;
  }
  auto release = common::MakeCleanup([&] { ReleaseTeam(team); });
  team->Run(job, n_blocks);
}

void SetTeamWorkerBudget(std::int32_t n_workers) {
  auto& registry = TeamRegistry::Registry();
  std::lock_guard lock{registry.mu};
  registry.budget = std::max(0, n_workers);
}

std::int32_t NumTeamWorkers() {
  auto& registry = TeamRegistry::Registry();
  std::lock_guard lock{registry.mu};
  return registry.n_workers;
}

void DestroyIdleTeams() {
  auto& registry = TeamRegistry::Registry();
  std::vector<std::unique_ptr<ThreadTeam>> doomed;
  {
    std::lock_guard lock{registry.mu};
    auto& teams = registry.teams;
    for (auto* team : registry.idle) {
      auto it =
          std::find_if(teams.begin(), teams.end(), [&](auto const& t) { return t.get() == team; });
      registry.n_workers -= team->NumWorkers();
      doomed.push_back(std::move(*it));
      teams.erase(it);
    }
    registry.idle.clear();
  }
  // Join the workers outside the lock.
  doomed.clear();
}
}  // namespace detail
}  // namespace xgboost::common
