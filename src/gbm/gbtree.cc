/**
 * Copyright 2014-2026, XGBoost Contributors
 *
 * \file gbtree.cc
 * \brief gradient boosted tree implementation.
 * \author Tianqi Chen
 */
#include "gbtree.h"

#include <dmlc/omp.h>
#include <dmlc/parameter.h>

#include <algorithm>  // for equal, any_of
#include <cstddef>    // for size_t
#include <cstdint>    // for uint32_t
#include <future>     // for future
#include <memory>
#include <optional>  // for optional
#include <string>
#include <tuple>  // for ignore
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>  // for getpid
#endif               // !defined(_WIN32)

#include "../collective/communicator-inl.h"  // for IsDistributed
#include "../common/common.h"
#include "../common/cuda_rt_utils.h"  // for AllVisibleGPUs
#include "../common/error_msg.h"  // for UnknownDevice, WarnOldSerialization, InplacePredictProxy
#include "../common/kernel.h"
#include "../common/threading_utils.h"
#include "../common/timer.h"
#include "../common/utils.h"        // for MakeCleanup
#include "../data/proxy_dmatrix.h"  // for DMatrixProxy, HostAdapterDispatch
#include "../predictor/prediction_kernel.h"
#include "gbtree_model.h"
#include "random_projection.h"  // for MakeRandomProjection, ProjectGradientKernel
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/gbm.h"
#include "xgboost/global_config.h"  // for InitNewThread
#include "xgboost/host_device_vector.h"
#include "xgboost/json.h"
#include "xgboost/logging.h"
#include "xgboost/model.h"
#include "xgboost/objective.h"
#include "xgboost/string_view.h"  // for StringView
#include "xgboost/tree_model.h"   // for RegTree
#include "xgboost/tree_updater.h"

namespace xgboost::gbm {
DMLC_REGISTRY_FILE_TAG(gbtree);

namespace {
/** @brief Map the `tree_method` parameter to the `updater` parameter. */
std::string MapTreeMethodToUpdaters(Context const* ctx, TreeMethod tree_method) {
  // Choose updaters according to tree_method parameters
  if (ctx->IsCUDA()) {
    common::AssertGPUSupport();
  }

  switch (tree_method) {
    case TreeMethod::kAuto:  // Use hist as default in 2.0
    case TreeMethod::kHist: {
      return ctx->DispatchDevice([] { return "grow_quantile_histmaker"; },
                                 [] { return "grow_gpu_hist"; },
                                 [] { return "grow_quantile_histmaker_sycl"; });
    }
    case TreeMethod::kApprox: {
      return ctx->DispatchDevice([] { return "grow_histmaker"; }, [] { return "grow_gpu_approx"; });
    }
    case TreeMethod::kExact:
      CHECK(ctx->IsCPU()) << "The `exact` tree method is not supported on GPU.";
      return "grow_colmaker,prune";
    default:
      auto tm = static_cast<std::underlying_type_t<TreeMethod>>(tree_method);
      LOG(FATAL) << "Unknown tree_method: `" << tm << "`.";
  }

  LOG(FATAL) << "unreachable";
  return "";
}

bool UpdatersMatched(std::vector<std::string> updater_seq,
                     std::vector<std::unique_ptr<TreeUpdater>> const& updaters) {
  if (updater_seq.size() != updaters.size()) {
    return false;
  }

  return std::equal(updater_seq.cbegin(), updater_seq.cend(), updaters.cbegin(),
                    [](std::string const& name, std::unique_ptr<TreeUpdater> const& up) {
                      return name == up->Name();
                    });
}

std::unique_ptr<TreeUpdater> CreateUpdater(std::string const& name, Context const* ctx,
                                           ObjInfo const* task, Json const& config) {
  std::unique_ptr<TreeUpdater> up{TreeUpdater::Create(name, ctx, task)};
  up->LoadConfig(config);
  return up;
}

// Divides the learning rate by the number of trees built per target, until the guard is
// destroyed.
[[nodiscard]] auto ScaleLearningRate(tree::TrainParam* param, std::size_t n_trees) {
  auto const lr = param->learning_rate;
  param->learning_rate /= static_cast<float>(n_trees);
  return common::MakeCleanup([param, lr] { param->learning_rate = lr; });
}

void ResetPositions(std::vector<HostDeviceVector<bst_node_t>>* positions, std::size_t n_trees) {
  positions->resize(n_trees);
  for (auto& position : *positions) {
    // Empty positions tell the updaters that none were written for this tree, for example
    // `prune` computes them if it runs first. Resizing keeps the memory for the next tree.
    position.Resize(0);
  }
}

// Adds the leaf values of trees at the rows' leaf positions to out_preds. Returns false, leaving
// out_preds unchanged, if the updaters didn't record a position for every row.
bool PredictFromPositions(Context const* ctx, std::vector<HostDeviceVector<bst_node_t>>* positions,
                          TreesOneGroup const& trees, linalg::MatrixView<float> out_preds) {
  CHECK_EQ(positions->size(), trees.size());
  for (auto& position : *positions) {
    if (out_preds.Shape(0) != 0 && position.Size() != out_preds.Shape(0)) {
      return false;
    }
    position.SetDevice(out_preds.Device());
  }
  std::vector<RegTree const*> tree_ptrs;
  tree_ptrs.reserve(trees.size());
  for (auto const& tree : trees) {
    tree_ptrs.push_back(tree.get());
  }
  common::DispatchKernel<predictor::PredictFromLeafIdsKernel>(ctx, common::Span{*positions},
                                                              common::Span{tree_ptrs}, out_preds);
  return true;
}

std::int64_t CurrentProcessId() {
#if defined(_WIN32)
  return 0;  // Windows has no fork.
#else
  return static_cast<std::int64_t>(getpid());
#endif  // defined(_WIN32)
}

}  // namespace

std::set<std::string> GBTree::Configure(Args const& cfg) {
  auto used = UpdateAndGetUsedParameters(&tparam_, cfg);
  used.merge(UpdateAndGetUsedParameters(&dparam_, cfg));
  used.merge(UpdateAndGetUsedParameters(&tree_param_, cfg));

  used.merge(model_.Configure(cfg));

  // for the 'update' process_type, move trees into trees_to_update
  if (tparam_.process_type == TreeProcessType::kUpdate) {
    model_.InitTreesToUpdate();
  }

  // `updater` parameter was manually specified
  specified_updater_ =
      std::any_of(cfg.cbegin(), cfg.cend(), [](auto const& arg) { return arg.first == "updater"; });
  if (specified_updater_) {
    error::WarnManualUpdater();
  }
  LOG(DEBUG) << "Using tree method: " << static_cast<int>(tparam_.tree_method);

  if (!specified_updater_) {
    this->tparam_.updater_seq = MapTreeMethodToUpdaters(ctx_, tparam_.tree_method);
  }

  auto up_names = common::Split(tparam_.updater_seq, ',');
  if (!UpdatersMatched(up_names, updaters_)) {
    updaters_.clear();
    for (auto const& name : up_names) {
      std::unique_ptr<TreeUpdater> up(
          TreeUpdater::Create(name.c_str(), ctx_, &model_.learner_model_state->task));
      updaters_.push_back(std::move(up));
    }
  }

  for (auto& up : updaters_) {
    used.merge(up->Configure(cfg));
  }
  // The workers are recreated on the next iteration if the updaters' configuration changed.
  return used;
}

void GBTreeModel::InitTreesToUpdate() {
  if (trees_to_update.empty()) {
    for (auto& tree : trees) {
      trees_to_update.push_back(std::move(tree));
    }

    trees.clear();
    param.num_trees = 0;
    tree_info.HostVector().clear();

    iteration_indptr.clear();
    iteration_indptr.push_back(0);
  }
}

void GPUCopyGradient(Context const*, linalg::Matrix<GradientPair> const*, bst_group_t,
                     linalg::Matrix<GradientPair>*)
#if defined(XGBOOST_USE_CUDA)
    ;  // NOLINT
#else
{
  common::AssertGPUSupport();
}
#endif

void CopyGradient(Context const* ctx, linalg::Matrix<GradientPair> const* in_gpair,
                  bst_group_t group_id, linalg::Matrix<GradientPair>* out_gpair) {
  out_gpair->SetDevice(ctx->Device());
  out_gpair->Reshape(in_gpair->Shape(0), 1);
  if (ctx->IsCUDA()) {
    GPUCopyGradient(ctx, in_gpair, group_id, out_gpair);
  } else {
    auto const& in = *in_gpair;
    auto h_tmp = out_gpair->HostView();
    auto h_in = in.HostView().Slice(linalg::All(), group_id);
    CHECK_EQ(h_tmp.Size(), h_in.Size());
    common::ParallelFor(h_in.Size(), ctx->Threads(), [&](auto i) { h_tmp(i) = h_in(i); });
  }
}

/** Increment the prediction on GPU.
 *
 * \param out_predts Prediction for the whole model.
 * \param predts     Prediction for current tree.
 * \param tree_w     Tree weight.
 */
void GPUDartPredictInc(common::Span<float>, common::Span<float>, float, size_t, bst_group_t,
                       bst_group_t)
#if defined(XGBOOST_USE_CUDA)
    ;  // NOLINT
#else
{
  common::AssertGPUSupport();
}
#endif

void GBTree::DoBoost(std::shared_ptr<DMatrix> p_fmat, GradientContainer* in_gpair,
                     ObjFunction const*) {
  auto predt = prediction_cache_.Cache(p_fmat, ctx_->Device());
  if (model_.learner_model_state->IsVectorLeaf()) {
    CHECK(tparam_.tree_method == TreeMethod::kHist || tparam_.tree_method == TreeMethod::kAuto)
        << "Only the hist tree method is supported for building multi-target trees with vector "
           "leaf.";
  }
  if (tree_param_.multi_projection_ratio < 1.0f) {
    CHECK(model_.learner_model_state->IsVectorLeaf())
        << "multi_projection_ratio requires multi_strategy=multi_output_tree.";
    CHECK(ctx_->IsCPU() || ctx_->IsCUDA())
        << "multi_projection_ratio supports CPU and CUDA training.";
    CHECK_EQ(tparam_.updater_seq, ctx_->IsCUDA() ? "grow_gpu_hist" : "grow_quantile_histmaker")
        << "multi_projection_ratio requires the hist tree method.";
    CHECK(!tree_param_.HasMonotone())
        << "Monotonic constraints are not supported with projected gradients.";
  }

  TreesOneIter new_trees;
  bst_target_t const n_groups = model_.learner_model_state->OutputLength();
  monitor_.Start("BoostNewTrees");

  // Define the categories.
  if (this->model_.Cats()->Empty() && !p_fmat->Cats()->Empty()) {
    auto in_cats = p_fmat->Cats();
    this->model_.Cats()->Copy(this->ctx_, *in_cats);
    this->model_.Cats()->Sort(this->ctx_);
  } else {
    CHECK_EQ(this->model_.Cats()->NumCatsTotal(), p_fmat->Cats()->NumCatsTotal())
        << "A new dataset with different categorical features is used for training an existing "
           "model.";
  }

  predt->predictions.SetDevice(ctx_->Device());
  if (predt->predictions.Size() == 0 && p_fmat->Info().num_row_ != 0) {
    CHECK_EQ(predt->version, 0);
    predictor::InitOutPredictions(ctx_, p_fmat->Info(), &predt->predictions, model_);
  }
  auto out = linalg::MakeTensorView(ctx_, &predt->predictions, p_fmat->Info().num_row_,
                                    model_.learner_model_state->OutputLength());
  CHECK_NE(n_groups, 0);

  if (model_.learner_model_state->IsVectorLeaf() ||
      model_.learner_model_state->OutputLength() == 1u) {
    TreesOneGroup ret;
    BoostNewTrees(in_gpair, p_fmat.get(), 0, &node_position_, &ret);
    if (PredictFromPositions(ctx_, &node_position_, ret, out)) {
      predt->Update(1);
    }
    new_trees.push_back(std::move(ret));
  } else {
    // Multi-target, scalar leaf
    CHECK_EQ(in_gpair->gpair.Size() % n_groups, 0U)
        << "Must have exactly n_groups * n_samples gpairs.";
    auto const n_workers = this->NumTargetWorkers(p_fmat.get(), n_groups);
    bool const cache_updated =
        this->BoostTargets(in_gpair, p_fmat.get(), n_workers, out, &new_trees);
    if (cache_updated) {
      predt->Update(1);
    }
  }

  monitor_.Stop("BoostNewTrees");
  this->CommitModel(std::move(new_trees));
}

std::vector<RegTree*> GBTree::InitNewTrees(bst_target_t bst_group, TreesOneGroup* ret) {
  std::vector<RegTree*> new_trees;
  ret->clear();
  // create the trees
  for (int i = 0; i < model_.param.num_parallel_tree; ++i) {
    if (tparam_.process_type == TreeProcessType::kDefault) {
      CHECK(!updaters_.empty());
      CHECK(!updaters_.front()->CanModifyTree())
          << "Updater: `" << updaters_.front()->Name() << "` "
          << "can not be used to create new trees. "
          << "Set `process_type` to `update` if you want to update existing "
             "trees.";
      // create new tree
      std::unique_ptr<RegTree> ptr(new RegTree{this->model_.learner_model_state->LeafLength(),
                                               this->model_.learner_model_state->num_feature});
      new_trees.push_back(ptr.get());
      ret->push_back(std::move(ptr));
    } else if (tparam_.process_type == TreeProcessType::kUpdate) {
      for (auto const& up : updaters_) {
        CHECK(up->CanModifyTree())
            << "Updater: `" << up->Name() << "` "
            << "can not be used to modify existing trees. "
            << "Set `process_type` to `default` if you want to build new trees.";
      }
      CHECK_LT(model_.trees.size(), model_.trees_to_update.size())
          << "No more tree left for updating.  For updating existing trees, "
          << "boosting rounds can not exceed previous training rounds";
      // move an existing tree from trees_to_update
      auto t = std::move(model_.trees_to_update[model_.trees.size() +
                                                bst_group * model_.param.num_parallel_tree + i]);
      new_trees.push_back(t.get());
      ret->push_back(std::move(t));
    }
  }
  return new_trees;
}

std::int32_t GBTree::NumTargetWorkers(DMatrix* p_fmat, bst_target_t n_groups) {
  if (!tparam_.parallel_target_trees) {
    return 1;
  }
  auto const n_threads = ctx_->Threads();
  std::string reason;
#if defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  // R allows its console functions, which log messages, only on R's main thread.
  reason = "the R package logs only from R's main thread";
#elif defined(XGBOOST_USE_THREADPOOL)
  // The workers' parallel loops would all run on the one global pool, which grows only to the
  // largest worker's share, instead of on a team of threads each.
  reason = "this build runs parallel loops on one shared thread pool";
#else
  if (n_threads < 2) {
    reason = "it needs more than one thread";
  } else if (!ctx_->IsCPU()) {
    reason = "it supports only CPU training";
  } else if (tparam_.process_type != TreeProcessType::kDefault) {
    reason = "it supports only `process_type=default`";
  } else if (collective::IsDistributed()) {
    reason = "it doesn't support distributed training";
  } else {
    for (auto const& up : updaters_) {
      if (!up->PrepareConcurrentBuild(&tree_param_, p_fmat)) {
        reason = "the `" + std::string(up->Name()) + "` updater doesn't support it with this data";
        break;
      }
    }
  }
#endif  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  if (reason != target_fallback_reason_ && !reason.empty()) {
    LOG(WARNING) << "`parallel_target_trees` builds the targets one after another: " << reason
                 << ".";
  }
  target_fallback_reason_ = reason;
  if (!reason.empty()) {
    return 1;
  }
  return std::min(static_cast<std::int32_t>(n_groups), n_threads);
}

GBTree::~GBTree() { this->ResetTargetWorkers(); }

void GBTree::ResetTargetWorkers() {
  if (target_pool_ && target_pool_pid_ != CurrentProcessId()) {
    // A forked child has none of the pool's threads, so joining them would fail: abandon it.
    std::ignore = target_pool_.release();
  }
  // Join the threads before destroying the workers they could run.
  target_pool_.reset();
  target_workers_.clear();
  target_workers_config_.clear();
}

void GBTree::InitTargetWorkers(std::int32_t n_workers) {
  // Split the threads between the workers, the first ones taking the remainder.
  auto const n_threads = ctx_->Threads();
  auto threads_of = [&](std::int32_t w) {
    return n_threads / n_workers + static_cast<std::int32_t>(w < n_threads % n_workers);
  };
  // The workers copy the updaters' configuration, so they are recreated when it changes.
  std::vector<Json> configs;
  Json all_configs{Array{}};
  for (auto const& up : updaters_) {
    Json config{Object{}};
    up->SaveConfig(&config);
    configs.push_back(config);
    get<Array>(all_configs).emplace_back(Object{});
    get<Array>(all_configs).back()["name"] = String{up->Name()};
    get<Array>(all_configs).back()["config"] = config;
  }
  std::string config_str;
  Json::Dump(all_configs, &config_str);

  bool reuse = target_pool_ && target_pool_pid_ == CurrentProcessId() &&
               target_workers_config_ == config_str &&
               target_workers_.size() == static_cast<std::size_t>(n_workers);
  for (std::int32_t w = 0; reuse && w < n_workers; ++w) {
    reuse = target_workers_[w].ctx->nthread == threads_of(w);
  }
  if (reuse) {
    return;
  }
  this->ResetTargetWorkers();
  for (std::int32_t w = 0; w < n_workers; ++w) {
    TargetWorker worker;
    worker.ctx = std::make_unique<Context>(*ctx_);
    worker.ctx->nthread = threads_of(w);
    for (std::size_t i = 0; i < updaters_.size(); ++i) {
      worker.updaters.push_back(CreateUpdater(updaters_[i]->Name(), worker.ctx.get(),
                                              &model_.learner_model_state->task, configs[i]));
    }
    target_workers_.push_back(std::move(worker));
  }
  // Each thread is a separate OpenMP root with its own team, so the workers' parallel regions
  // run side by side instead of nesting. BoostTargets initializes the threads for each task.
  target_pool_ =
      std::make_unique<common::ThreadPool>(StringView{"tgt-trees"}, n_workers - 1, [] {});
  target_pool_pid_ = CurrentProcessId();
  target_workers_config_ = std::move(config_str);
  ++n_target_worker_builds_;
}

bool GBTree::BoostTargets(GradientContainer* in_gpair, DMatrix* p_fmat, std::int32_t n_workers,
                          linalg::MatrixView<float> out_preds, TreesOneIter* out_trees) {
  auto const n_groups = model_.learner_model_state->OutputLength();
  // Everything that touches the model or the booster's state happens on this thread, before
  // the workers start. InitNewTrees moves trees of the model with process_type=update.
  out_trees->resize(n_groups);
  std::vector<std::vector<RegTree*>> new_trees(n_groups);
  for (bst_target_t gid = 0; gid < n_groups; ++gid) {
    new_trees[gid] = this->InitNewTrees(gid, &(*out_trees)[gid]);
  }
  // With parallel_target_trees, each target samples from its own seed, drawn in target order,
  // so the model doesn't depend on whether, or by which worker, the targets are built at the
  // same time. Building on this thread reseeds the booster's generator, which is restored after.
  std::vector<RandomEngine::result_type> seeds;
  if (tparam_.parallel_target_trees) {
    for (bst_target_t gid = 0; gid < n_groups; ++gid) {
      seeds.push_back(ctx_->Rng()());
    }
  }
  std::optional<RandomEngine> rng_after_seeds;
  if (!seeds.empty()) {
    rng_after_seeds = ctx_->Rng();
  }
  auto restore_rng = common::MakeCleanup([&] {
    if (rng_after_seeds) {
      ctx_->Rng() = *rng_after_seeds;
    }
  });
  // The updaters keep the address of the training parameter, so it must outlive this call.
  auto restore_lr = ScaleLearningRate(&tree_param_, new_trees.front().size());

  std::vector<char> cache_updated(n_groups, 0);
  auto build = [&](Context const* ctx, std::vector<std::unique_ptr<TreeUpdater>> const& updaters,
                   GradientContainer* gpair, std::vector<HostDeviceVector<bst_node_t>>* positions,
                   bst_target_t gid) {
    CopyGradient(ctx, &in_gpair->gpair, gid, &gpair->gpair);
    this->CheckGradientSize(*gpair, p_fmat);
    ResetPositions(positions, new_trees[gid].size());
    if (!seeds.empty()) {
      ctx->Rng().seed(seeds[gid]);
    }
    for (auto const& up : updaters) {
      up->Update(&tree_param_, gpair, p_fmat, common::Span{*positions}, new_trees[gid]);
    }
    // Each target writes its own column of the cache, so workers can update it concurrently.
    auto v_preds = out_preds.Slice(linalg::All(), linalg::Range(gid, gid + 1));
    cache_updated[gid] = PredictFromPositions(ctx, positions, (*out_trees)[gid], v_preds);
  };

  if (n_workers == 1) {
    // Free the workers of earlier concurrent builds.
    this->ResetTargetWorkers();
    GradientContainer gpair;
    for (bst_target_t gid = 0; gid < n_groups; ++gid) {
      build(ctx_, updaters_, &gpair, &node_position_, gid);
    }
  } else {
    monitor_.Start("BoostTargetsConcurrently");
    this->InitTargetWorkers(n_workers);
    // Every worker reads the gradient. Bringing it to the host here keeps its lazy copy, which
    // writes to it, off the workers.
    std::ignore = in_gpair->gpair.HostView();
    auto run = [&](std::int32_t w) {
      auto& worker = target_workers_[w];
      // A fixed assignment keeps each target on the same worker, and so the same thread count,
      // from run to run.
      for (bst_target_t gid = w; gid < n_groups; gid += n_workers) {
        build(worker.ctx.get(), worker.updaters, &worker.gpair, &worker.positions, gid);
      }
    };
    // The global configuration and the log callback are thread-local, so each task starts from
    // this thread's. Training is on the CPU, so the tasks don't select a GPU.
    InitNewThread init;
    init.device = -1;
    auto const log_callback = LogCallbackRegistryStore::Get()->Get();
    std::vector<std::future<void>> futures;
    futures.reserve(n_workers - 1);
    // Wait for every started worker before leaving, even on an error: they use this frame.
    auto drain = common::MakeCleanup([&] {
      for (auto& future : futures) {
        if (future.valid()) {
          future.wait();
        }
      }
    });
    for (std::int32_t w = 1; w < n_workers; ++w) {
      futures.push_back(target_pool_->Submit([&run, &init, log_callback, w] {
        init();
        LogCallbackRegistryStore::Get()->Register(log_callback);
        run(w);
      }));
    }
    run(0);
    for (auto& future : futures) {
      future.get();
    }
    monitor_.Stop("BoostTargetsConcurrently");
  }
  return std::all_of(cache_updated.cbegin(), cache_updated.cend(),
                     [](char updated) { return updated != 0; });
}

void GBTree::CheckGradientSize(GradientContainer const& gpair, DMatrix const* p_fmat) const {
  auto n_out = model_.learner_model_state->OutputLength() * p_fmat->Info().num_row_;
  StringView msg{
      "Mismatching size between number of rows from input data and size of gradient vector."};
  if (!model_.learner_model_state->IsVectorLeaf() && p_fmat->Info().num_row_ != 0) {
    CHECK_EQ(n_out % gpair.gpair.Size(), 0) << msg;
  } else if (model_.learner_model_state->IsVectorLeaf()) {
    // vector leaf
    CHECK_EQ(gpair.gpair.Size(), n_out) << msg;
  }
}

void GBTree::BoostNewTrees(GradientContainer* gpair, DMatrix* p_fmat, int bst_group,
                           std::vector<HostDeviceVector<bst_node_t>>* out_position,
                           TreesOneGroup* ret) {
  std::vector<RegTree*> new_trees = this->InitNewTrees(bst_group, ret);

  // update the trees
  this->CheckGradientSize(*gpair, p_fmat);
  ResetPositions(out_position, new_trees.size());

  auto n_targets = gpair->gpair.Shape(1);
  auto n_components = static_cast<std::size_t>(tree_param_.multi_projection_ratio * n_targets);
  CHECK_GT(n_components, 0) << "multi_projection_ratio must retain at least one target.";

  // Rescale learning rate according to the number of trees
  auto restore_lr = ScaleLearningRate(&tree_param_, new_trees.size());
  if (n_components < n_targets) {
    gpair->value_gpair = std::move(gpair->gpair);
    auto full_view = std::as_const(gpair->value_gpair).View(ctx_->Device());
    gpair->gpair = linalg::Empty<GradientPair>(ctx_, full_view.Shape(0), n_components);
    // Share one projection across all parallel trees in this boosting round.
    monitor_.Start("ProjectGradient");
    auto basis = MakeRandomProjection(ctx_, n_targets, n_components);
    common::DispatchKernel<ProjectGradientKernel>(ctx_, full_view, basis.View(ctx_->Device()),
                                                  gpair->gpair.View(ctx_->Device()));
    monitor_.Stop("ProjectGradient");
  }
  for (auto& up : updaters_) {
    up->Update(&tree_param_, gpair, p_fmat, common::Span{*out_position}, new_trees);
  }
  if (n_components < n_targets) {
    gpair->gpair = std::exchange(gpair->value_gpair, {});
  }
}

void GBTree::CommitModel(TreesOneIter&& new_trees) {
  monitor_.Start("CommitModel");
  auto n_old_trees = model_.trees.size();
  auto has_tree_weights = !model_.weight_drop.empty();
  auto dropout_configured = dparam_.HasDropout();
  auto track_tree_weights = has_tree_weights || dropout_configured;
  if (track_tree_weights && model_.weight_drop.size() < n_old_trees) {
    model_.weight_drop.insert(model_.weight_drop.cend(), n_old_trees - model_.weight_drop.size(),
                              1.0f);
  }
  auto n_new_trees = model_.CommitModel(std::forward<TreesOneIter>(new_trees));
  if (track_tree_weights) {
    auto num_drop = this->NormalizeTrees(n_new_trees);
    LOG(INFO) << "drop " << num_drop << " trees, "
              << "weight = " << model_.weight_drop.back();
  }
  monitor_.Stop("CommitModel");
}

void GBTree::LoadConfig(Json const& in) {
  auto name = get<String const>(in["name"]);
  CHECK(name == "gbtree" || name == "dart")
      << "Unknown booster name in model JSON: `" << name
      << "`. Only `gbtree` or legacy `dart` boosters are accepted here.";
  auto const& config = name == "dart" ? in["gbtree"] : in;
  FromJson(config["gbtree_train_param"], &tparam_);
  FromJson(config["tree_train_param"], &tree_param_);
  auto const& obj = get<Object const>(config);
  auto it = obj.find("dart_train_param");
  if (it != obj.cend()) {
    FromJson(it->second, &dparam_);
  } else if (name == "dart") {
    FromJson(in["dart_train_param"], &dparam_);
  } else {
    dparam_ = {};
  }

  // Process type cannot be kUpdate from loaded model
  // This would cause all trees to be pushed to trees_to_update
  // e.g. updating a model, then saving and loading it would result in an empty model
  tparam_.process_type = TreeProcessType::kDefault;
  std::int32_t const n_gpus = curt::AllVisibleGPUs();

  std::vector<Json> updater_seq;
  if (IsA<Object>(config["updater"])) {
    // before 2.0
    error::WarnOldSerialization();
    for (auto const& kv : get<Object const>(config["updater"])) {
      auto name = kv.first;
      auto config = kv.second;
      config["name"] = name;
      updater_seq.push_back(config);
    }
  } else {
    // after 2.0
    auto const& j_updaters = get<Array const>(config["updater"]);
    updater_seq = j_updaters;
  }

  updaters_.clear();

  for (auto const& config : updater_seq) {
    auto name = get<String>(config["name"]);
    if (n_gpus == 0 && name == "grow_gpu_hist") {
      name = "grow_quantile_histmaker";
      LOG(WARNING) << "Changing updater from `grow_gpu_hist` to `grow_quantile_histmaker`.";
    }
    updaters_.push_back(CreateUpdater(name, ctx_, &model_.learner_model_state->task, config));
  }
  this->ResetTargetWorkers();

  specified_updater_ = get<Boolean>(config["specified_updater"]);
}

void GBTree::SaveConfig(Json* p_out) const {
  auto& out = *p_out;
  out["name"] = String("gbtree");
  out["gbtree_train_param"] = ToJson(tparam_);
  out["tree_train_param"] = ToJson(tree_param_);
  out["dart_train_param"] = ToJson(dparam_);

  // Process type cannot be kUpdate from loaded model
  // This would cause all trees to be pushed to trees_to_update
  // e.g. updating a model, then saving and loading it would result in an empty
  // model
  out["gbtree_train_param"]["process_type"] = String("default");
  // Duplicated from SaveModel so that user can get `num_parallel_tree` without parsing
  // the model. We might remove this once we can deprecate `best_ntree_limit` so that the
  // language binding doesn't need to know about the forest size.
  out["gbtree_model_param"] = ToJson(model_.param);

  out["updater"] = Array{};
  auto& j_updaters = get<Array>(out["updater"]);

  for (auto const& up : this->updaters_) {
    Json up_config{Object{}};
    up_config["name"] = String{up->Name()};
    up->SaveConfig(&up_config);
    j_updaters.emplace_back(up_config);
  }
  out["specified_updater"] = Boolean{specified_updater_};
}

void GBTree::LoadModel(Json const& in) {
  auto name = get<String const>(in["name"]);
  CHECK(name == "gbtree" || name == "dart");
  auto const& model = name == "dart" ? in["gbtree"] : in;
  model_.LoadModel(model["model"]);
  auto const& obj = get<Object const>(name == "dart" ? in : model);
  // Compatibility for older models that stored DART weights beside the tree model.
  auto it = obj.find("weight_drop");
  if (it != obj.cend()) {
    auto const& j_weight_drop = get<Array const>(it->second);
    model_.weight_drop.resize(j_weight_drop.size());
    for (size_t i = 0; i < model_.weight_drop.size(); ++i) {
      model_.weight_drop[i] = get<Number const>(j_weight_drop[i]);
    }
  }
  CHECK_LE(model_.weight_drop.size(), model_.trees.size());
}

void GBTree::SaveModel(Json* p_out) const {
  auto& out = *p_out;
  out["name"] = String("gbtree");
  out["model"] = Object();
  model_.SaveModel(&out["model"]);
}

std::vector<float> GBTree::DropTrees(bool is_training) {
  if (!is_training) {
    return {};
  }
  auto dropout_configured =
      dparam_.rate_drop != 0.0f || dparam_.one_drop || dparam_.skip_drop != 0.0f;
  if (model_.weight_drop.empty()) {
    if (!dropout_configured || model_.trees.empty()) {
      return {};
    }
    model_.weight_drop.resize(model_.trees.size(), 1.0f);
  }
  idx_drop_.clear();

  std::uniform_real_distribution<> runif(0.0, 1.0);
  auto& rnd = ctx_->Rng();
  bool skip = false;
  if (dparam_.skip_drop > 0.0) {
    skip = (runif(rnd) < dparam_.skip_drop);
  }
  if (skip) {
    return {};
  }

  if (dparam_.sample_type == DartSampleType::kWeighted) {
    bst_float sum_weight = 0.0;
    for (auto elem : model_.weight_drop) {
      sum_weight += elem;
    }
    for (size_t i = 0; i < model_.weight_drop.size(); ++i) {
      if (runif(rnd) <
          dparam_.rate_drop * model_.weight_drop.size() * model_.weight_drop[i] / sum_weight) {
        idx_drop_.push_back(i);
      }
    }
    if (dparam_.one_drop && idx_drop_.empty() && !model_.weight_drop.empty()) {
      size_t i = std::discrete_distribution<size_t>(
          model_.weight_drop.size(), 0., static_cast<double>(model_.weight_drop.size()),
          [this](double x) -> double { return model_.weight_drop[static_cast<size_t>(x)]; })(rnd);
      idx_drop_.push_back(i);
    }
  } else {
    for (size_t i = 0; i < model_.weight_drop.size(); ++i) {
      if (runif(rnd) < dparam_.rate_drop) {
        idx_drop_.push_back(i);
      }
    }
    if (dparam_.one_drop && idx_drop_.empty() && !model_.weight_drop.empty()) {
      size_t i = std::uniform_int_distribution<size_t>(0, model_.weight_drop.size() - 1)(rnd);
      idx_drop_.push_back(i);
    }
  }

  if (idx_drop_.empty()) {
    return {};
  }

  auto dropped_weights = model_.weight_drop;
  for (auto idx : idx_drop_) {
    dropped_weights.at(idx) = 0.0f;
  }
  return dropped_weights;
}

[[nodiscard]] std::size_t GBTree::NormalizeTrees(std::size_t size_new_trees) {
  CHECK(tree_param_.GetInitialised());
  // Parallel trees each use eta / num_parallel_tree, so the complete layer uses eta.
  auto lr = tree_param_.learning_rate;
  size_t num_drop = idx_drop_.size();
  if (num_drop == 0) {
    for (size_t i = 0; i < size_new_trees; ++i) {
      model_.weight_drop.push_back(1.0);
    }
  } else if (dparam_.normalize_type == 1) {
    float factor = 1.0 / (1.0 + lr);
    for (auto i : idx_drop_) {
      model_.weight_drop[i] *= factor;
    }
    for (size_t i = 0; i < size_new_trees; ++i) {
      model_.weight_drop.push_back(factor);
    }
  } else {
    float factor = 1.0 * num_drop / (num_drop + lr);
    for (auto i : idx_drop_) {
      model_.weight_drop[i] *= factor;
    }
    for (size_t i = 0; i < size_new_trees; ++i) {
      model_.weight_drop.push_back(1.0 / (num_drop + lr));
    }
  }
  idx_drop_.clear();
  return num_drop;
}

void GBTree::Slice(bst_layer_t begin, bst_layer_t end, bst_layer_t step, GradientBooster* out,
                   bool* out_of_bound) const {
  CHECK(out);

  auto p_gbtree = dynamic_cast<GBTree*>(out);
  CHECK(p_gbtree);
  GBTreeModel& out_model = p_gbtree->model_;
  CHECK(this->model_.learner_model_state->Initialized());

  end = end == 0 ? model_.BoostedRounds() : end;
  CHECK_GE(step, 1);
  CHECK_NE(end, begin) << "Empty slice is not allowed.";

  if (step > (end - begin)) {
    *out_of_bound = true;
    return;
  }

  auto& out_indptr = out_model.iteration_indptr;
  TreesOneGroup& out_trees = out_model.trees;
  auto& out_tree_info = out_model.tree_info.HostVector();

  auto const& in_tree_info = this->model_.tree_info.ConstHostVector();

  bst_layer_t n_layers = (end - begin) / step;
  out_indptr.resize(n_layers + 1, 0);

  if (!this->model_.trees_to_update.empty()) {
    CHECK_EQ(this->model_.trees_to_update.size(), this->model_.trees.size())
        << "Not all trees are updated, "
        << this->model_.trees_to_update.size() - this->model_.trees.size()
        << " trees remain.  Slice the model before making update if you only "
           "want to update a portion of trees.";
  }

  *out_of_bound =
      detail::SliceTrees(begin, end, step, this->model_, [&](auto in_tree_idx, auto out_l) {
        std::unique_ptr<RegTree> new_tree{this->model_.trees.at(in_tree_idx)->Copy()};
        out_trees.emplace_back(std::move(new_tree));

        bst_group_t group = in_tree_info[in_tree_idx];
        out_tree_info.push_back(group);

        out_model.iteration_indptr[out_l + 1]++;
      });

  std::partial_sum(out_indptr.cbegin(), out_indptr.cend(), out_indptr.begin());
  CHECK_EQ(out_model.iteration_indptr.front(), 0);

  out_model.param.num_trees = out_model.trees.size();
  out_model.param.num_parallel_tree = model_.param.num_parallel_tree;
  out_model.Cats()->Copy(this->ctx_, *this->model_.Cats());

  p_gbtree->dparam_ = this->dparam_;
  p_gbtree->idx_drop_.clear();
  p_gbtree->model_.weight_drop.clear();
  if (!this->model_.weight_drop.empty()) {
    detail::SliceTrees(begin, end, step, model_, [&](auto in_tree_idx, auto const&) {
      p_gbtree->model_.weight_drop.push_back(this->model_.weight_drop.at(in_tree_idx));
    });
  }
}

std::vector<bst_ulong> GBTree::PredictBatch(std::shared_ptr<DMatrix> p_fmat,
                                            HostDeviceVector<float>* out_preds, bool is_training,
                                            bst_layer_t layer_begin, bst_layer_t layer_end) {
  auto cache = prediction_cache_.Cache(p_fmat, ctx_->Device());
  auto const* tree_weights_override = static_cast<std::vector<float> const*>(nullptr);
  auto dropped_weights = this->DropTrees(is_training);
  if (!dropped_weights.empty()) {
    tree_weights_override = &dropped_weights;
  }

  // Unweighted prediction can reuse a cached prefix of the model output by tracking how many
  // boosting iterations have already been accumulated in `cache->version`.
  //
  // Weighted prediction is used by DART and does not participate in this cache, since tree
  // weights can change the accumulated output independently of the cached unweighted prefix.
  if (layer_end == 0) {
    layer_end = this->BoostedRounds();
  }

  auto cache_version = cache->version;
  // We can preserve the cache only when:
  // - prediction is unweighted
  // - prediction starts from iteration 0, so the result is a cacheable prefix
  auto preserve_cache = tree_weights_override == nullptr && model_.TreeWeights() == nullptr &&
                        p_fmat->Info().base_margin_.Empty() && layer_begin == 0;
  // We can reuse the existing cached prefix only when:
  // - the result itself is cacheable
  // - the requested range does not move backwards past the cached version
  auto reuse_cache = preserve_cache && layer_end >= static_cast<bst_layer_t>(cache_version);
  // Initialize output when:
  // - the cached prefix cannot be reused, or
  // - the cache is valid but still empty
  auto initialize_output = !reuse_cache || cache_version == 0;
  auto prediction_begin = reuse_cache ? cache_version : layer_begin;

  if (!reuse_cache) {
    cache->version = 0;
    cache_version = 0;
  }

  if (cache->predictions.Size() == 0 && p_fmat->Info().num_row_ != 0) {
    CHECK_EQ(cache->version, 0);
  }

  if (initialize_output) {
    // cache->Size() can be non-zero as it's initialized here before any
    // tree is built at the 0^th iterator.
    predictor::InitOutPredictions(ctx_, p_fmat->Info(), &cache->predictions, model_);
  }

  auto [tree_begin, tree_end] = detail::LayerToTree(model_, prediction_begin, layer_end);
  CHECK_LE(tree_end, model_.trees.size()) << "Invalid number of trees.";
  if (tree_end > tree_begin) {
    common::DispatchKernel<predictor::PredictBatchKernel>(ctx_, p_fmat.get(), &cache->predictions,
                                                          model_, tree_begin, tree_end,
                                                          tree_weights_override);
  }

  if (!preserve_cache) {
    cache->version = 0;
  } else {
    cache->Update(layer_end - cache_version);
  }

  out_preds->SetDevice(ctx_->Device());
  out_preds->Resize(cache->predictions.Size());
  out_preds->Copy(cache->predictions);
  return {p_fmat->Info().num_row_, model_.learner_model_state->OutputLength()};
}

std::vector<bst_ulong> GBTree::InplacePredict(std::shared_ptr<DMatrix> p_m, float missing,
                                              HostDeviceVector<float>* out_preds,
                                              bst_layer_t layer_begin,
                                              bst_layer_t layer_end) const {
  auto [tree_begin, tree_end] = detail::LayerToTree(model_, layer_begin, layer_end);
  CHECK_LE(tree_end, model_.trees.size()) << "Invalid number of trees.";
  if (p_m->Ctx()->Device() != this->ctx_->Device()) {
    error::MismatchedDevices(this->ctx_, p_m->Ctx());
    auto proxy = std::dynamic_pointer_cast<data::DMatrixProxy>(p_m);
    CHECK(proxy) << error::InplacePredictProxy();
    auto p_fmat = data::CreateDMatrixFromProxy(ctx_, proxy, missing);
    predictor::InitOutPredictions(ctx_, p_fmat->Info(), out_preds, model_);
    if (tree_end > tree_begin) {
      common::DispatchKernel<predictor::PredictBatchKernel>(ctx_, p_fmat.get(), out_preds, model_,
                                                            tree_begin, tree_end, nullptr);
    }
    return {p_m->Info().num_row_, model_.learner_model_state->OutputLength()};
  }

  bool known_type = common::DispatchKernel<predictor::InplacePredictKernel>(
      ctx_, p_m, model_, missing, out_preds, tree_begin, tree_end);
  if (!known_type) {
    auto proxy = std::dynamic_pointer_cast<data::DMatrixProxy>(p_m);
    CHECK(proxy) << error::InplacePredictProxy();
    LOG(FATAL) << "Unknown data type for inplace prediction:" << proxy->Adapter().type().name();
  }
  return {p_m->Info().num_row_, model_.learner_model_state->OutputLength()};
}

// register the objective functions
DMLC_REGISTER_PARAMETER(GBTreeModelParam);
DMLC_REGISTER_PARAMETER(GBTreeTrainParam);
DMLC_REGISTER_PARAMETER(DartTrainParam);

XGBOOST_REGISTER_GBM(GBTree, "gbtree")
    .describe("Tree booster, gradient boosted trees.")
    .set_body([](LearnerModelState const* booster_config, Context const* ctx) {
      auto* p = new GBTree{booster_config, ctx};
      return p;
    });
}  // namespace xgboost::gbm
