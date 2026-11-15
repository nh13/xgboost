/**
 * Copyright 2019-2026, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>
#include <xgboost/host_device_vector.h>  // for HostDeviceVector
#include <xgboost/json.h>                // for Json, Object
#include <xgboost/learner.h>             // for Learner
#include <xgboost/logging.h>             // for ConsoleLogger
#include <xgboost/tree_updater.h>        // for TreeUpdater, XGBOOST_REGISTER_TREE_UPDATER

#include <algorithm>  // for fill
#include <cstdint>    // for int32_t, uint32_t
#include <limits>     // for numeric_limits
#include <memory>     // for shared_ptr
#include <optional>   // for optional
#include <set>        // for set
#include <string>     // for string, to_string
#include <thread>     // for this_thread
#include <tuple>      // for ignore
#include <utility>    // for pair
#include <vector>     // for vector

#if !defined(_WIN32)
#include <sys/wait.h>  // for waitpid, WIFEXITED
#include <unistd.h>    // for fork, alarm, _exit
#endif                 // !defined(_WIN32)

#include "../../../src/data/proxy_dmatrix.h"  // for DMatrixProxy
#include "../../../src/gbm/gbtree.h"
#include "../filesystem.h"  // TemporaryDirectory
#include "../helpers.h"
#include "xgboost/base.h"

namespace xgboost {
TEST(GBTree, SelectTreeMethod) {
  size_t constexpr kCols = 10;

  Context ctx;
  LearnerModelState mparam{MakeMP(kCols, .5, 1)};

  std::unique_ptr<GradientBooster> p_gbm{GradientBooster::Create("gbtree", &ctx, &mparam)};
  auto& gbtree = dynamic_cast<gbm::GBTree&>(*p_gbm);

  // Test if `tree_method` can be set
  Args args{{"tree_method", "approx"}};
  gbtree.Configure({args.cbegin(), args.cend()});

  gbtree.Configure(args);
  auto const& tparam = gbtree.GetTrainParam();
  gbtree.Configure({{"tree_method", "approx"}});
  ASSERT_EQ(tparam.updater_seq, "grow_histmaker");
  gbtree.Configure({{"tree_method", "exact"}});
  ASSERT_EQ(tparam.updater_seq, "grow_colmaker,prune");
  gbtree.Configure({{"tree_method", "hist"}});
  ASSERT_EQ(tparam.updater_seq, "grow_quantile_histmaker");
  gbtree.Configure({{"booster", "dart"}, {"tree_method", "hist"}});
  ASSERT_EQ(tparam.updater_seq, "grow_quantile_histmaker");

#ifdef XGBOOST_USE_CUDA
  ctx.UpdateAllowUnknown(Args{{"device", "cuda"}});
  gbtree.Configure({{"tree_method", "hist"}});
  ASSERT_EQ(tparam.updater_seq, "grow_gpu_hist");
  gbtree.Configure({{"booster", "dart"}, {"tree_method", "hist"}});
  ASSERT_EQ(tparam.updater_seq, "grow_gpu_hist");
#endif  // XGBOOST_USE_CUDA
}

TEST(GBTree, PredictionCache) {
  size_t constexpr kRows = 100, kCols = 10;
  Context ctx;
  LearnerModelState mparam{MakeMP(kCols, .5, 1)};

  std::unique_ptr<GradientBooster> p_gbm{GradientBooster::Create("gbtree", &ctx, &mparam)};
  auto& gbtree = dynamic_cast<gbm::GBTree&>(*p_gbm);

  gbtree.Configure({{"tree_method", "hist"}});
  auto p_m = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix();

  GradientContainer gpair = GenerateRandomGradients(&ctx, kRows, 1);

  HostDeviceVector<float> out_predictions;
  gbtree.DoBoost(p_m, &gpair, nullptr);

  ASSERT_EQ(gbtree.PredictBatch(p_m, &out_predictions, false, 0, 0),
            (std::vector<bst_ulong>{kRows, 1}));
  ASSERT_EQ(1, gbtree.PredictionCache(p_m.get()).version);
  ASSERT_EQ(out_predictions.Size(), kRows);
  std::vector<float> first_iter = out_predictions.HostVector();
  // Add 1 more boosted round
  gbtree.DoBoost(p_m, &gpair, nullptr);
  ASSERT_EQ(gbtree.PredictBatch(p_m, &out_predictions, false, 0, 0),
            (std::vector<bst_ulong>{kRows, 1}));
  ASSERT_EQ(2, gbtree.PredictionCache(p_m.get()).version);
  // Update the cache for all rounds
  ASSERT_EQ(gbtree.PredictBatch(p_m, &out_predictions, false, 1, 2),
            (std::vector<bst_ulong>{kRows, 1}));
  ASSERT_EQ(0, gbtree.PredictionCache(p_m.get()).version);
  ASSERT_EQ(gbtree.PredictBatch(p_m, &out_predictions, false, 0, 0),
            (std::vector<bst_ulong>{kRows, 1}));
  ASSERT_EQ(2, gbtree.PredictionCache(p_m.get()).version);

  gbtree.DoBoost(p_m, &gpair, nullptr);
  // drop the cache.
  ASSERT_EQ(gbtree.PredictBatch(p_m, &out_predictions, false, 1, 2),
            (std::vector<bst_ulong>{kRows, 1}));
  ASSERT_EQ(0, gbtree.PredictionCache(p_m.get()).version);
  // half open set [1, 3)
  ASSERT_EQ(gbtree.PredictBatch(p_m, &out_predictions, false, 1, 3),
            (std::vector<bst_ulong>{kRows, 1}));
  ASSERT_EQ(0, gbtree.PredictionCache(p_m.get()).version);
  // iteration end
  ASSERT_EQ(gbtree.PredictBatch(p_m, &out_predictions, false, 0, 2),
            (std::vector<bst_ulong>{kRows, 1}));
  ASSERT_EQ(2, gbtree.PredictionCache(p_m.get()).version);
  // restart the cache when end iteration is smaller than cache version
  ASSERT_EQ(gbtree.PredictBatch(p_m, &out_predictions, false, 0, 1),
            (std::vector<bst_ulong>{kRows, 1}));
  ASSERT_EQ(1, gbtree.PredictionCache(p_m.get()).version);
  ASSERT_EQ(out_predictions.HostVector(), first_iter);
}

TEST(GBTree, PredictionCacheWithBaseMargin) {
  size_t constexpr kRows = 16, kCols = 4;
  Context ctx;
  LearnerModelState mparam{MakeMP(kCols, .5, 1)};

  std::unique_ptr<GradientBooster> p_gbm{GradientBooster::Create("gbtree", &ctx, &mparam)};
  auto& gbtree = dynamic_cast<gbm::GBTree&>(*p_gbm);

  gbtree.Configure({{"tree_method", "hist"}});
  auto p_m = RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix();

  HostDeviceVector<float> out_predictions;
  p_m->Info().base_margin_.Reshape(kRows, 1);
  p_m->Info().base_margin_.Data()->HostVector().assign(kRows, 1.0f);
  gbtree.PredictBatch(p_m, &out_predictions, false, 0, 0);
  auto first = out_predictions.HostVector();
  ASSERT_EQ(0, gbtree.PredictionCache(p_m.get()).version);

  p_m->Info().base_margin_.Data()->HostVector().assign(kRows, 2.0f);
  gbtree.PredictBatch(p_m, &out_predictions, false, 0, 0);
  auto second = out_predictions.HostVector();
  ASSERT_EQ(0, gbtree.PredictionCache(p_m.get()).version);

  for (size_t i = 0; i < kRows; ++i) {
    ASSERT_NEAR(second[i] - first[i], 1.0f, kRtEps);
  }
}

TEST(GBTree, WrongUpdater) {
  size_t constexpr kRows = 17;
  size_t constexpr kCols = 15;

  auto p_dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  p_dmat->Info().labels.Reshape(kRows);

  auto learner = std::unique_ptr<Learner>(Learner::Create({p_dmat}));
  // Hist can not be used for updating tree.
  learner->Configure(Args{{"tree_method", "hist"}, {"process_type", "update"}});
  ASSERT_THROW(learner->UpdateOneIter(0, p_dmat), dmlc::Error);
  // Prune can not be used for learning new tree.
  ASSERT_THROW(learner->Configure(Args{{"tree_method", "prune"}, {"process_type", "default"}}),
               dmlc::Error);
}

#ifdef XGBOOST_USE_CUDA
TEST(GBTree, TrainingPredictionDevice) {
  // Ordinary training updates predictions from leaf positions without copying raw data.
  // Resuming training uses the configured device to predict the existing model.
  std::size_t constexpr kRows = 17, kCols = 15;

  auto p_dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();

  auto const& data = (*(p_dmat->GetBatches<SparsePage>().begin())).data;
  p_dmat->Info().labels.Reshape(kRows);

  auto learner = std::unique_ptr<Learner>(Learner::Create({p_dmat}));
  learner->Configure(Args{{"tree_method", "hist"}, {"device", "cuda"}});
  for (size_t i = 0; i < 4; ++i) {
    learner->UpdateOneIter(i, p_dmat);
  }
  ASSERT_TRUE(data.HostCanWrite());

  common::TemporaryDirectory tempdir;
  const std::string fname = tempdir.Str() + "/model_param.bst";
  {
    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname.c_str(), "w"));
    learner->Save(fo.get());
  }
  // a new learner
  learner = std::unique_ptr<Learner>(Learner::Create({p_dmat}));
  {
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname.c_str(), "r"));
    learner->Load(fi.get());
  }
  learner->Configure(Args{{"tree_method", "hist"}, {"device", "cuda"}});
  for (size_t i = 0; i < 4; ++i) {
    learner->UpdateOneIter(i, p_dmat);
  }
  ASSERT_FALSE(data.HostCanWrite());
  ASSERT_TRUE(data.DeviceCanRead());

  // another new learner
  learner = std::unique_ptr<Learner>(Learner::Create({p_dmat}));
  learner->Configure(Args{{"tree_method", "hist"}, {"device", "cuda"}});
  for (size_t i = 0; i < 4; ++i) {
    learner->UpdateOneIter(i, p_dmat);
  }
  // data is not pulled back into host
  ASSERT_FALSE(data.HostCanWrite());
}

TEST(GBTree, ChooseTreeMethod) {
  bst_idx_t n_samples{128};
  bst_feature_t n_features{64};
  auto Xy = RandomDataGenerator{n_samples, n_features, 0.5f}.GenerateDMatrix(true);

  auto with_update = [&](std::optional<std::string> device,
                         std::optional<std::string> tree_method) {
    auto learner = std::unique_ptr<Learner>(Learner::Create({Xy}));
    if (tree_method.has_value()) {
      learner->Configure({{"tree_method", tree_method.value()}});
    }
    if (device.has_value()) {
      auto const& d = device.value();
      learner->Configure({{"device", d}});
    }
    learner->Configure();
    for (std::int32_t i = 0; i < 3; ++i) {
      learner->UpdateOneIter(0, Xy);
    }
    Json config{Object{}};
    learner->SaveConfig(&config);
    auto updater = config["learner"]["gradient_booster"]["updater"];
    CHECK(!IsA<Null>(updater));
    return updater;
  };

  auto with_boost = [&](std::optional<std::string> device, std::optional<std::string> tree_method) {
    auto learner = std::unique_ptr<Learner>(Learner::Create({Xy}));
    if (tree_method.has_value()) {
      learner->Configure({{"tree_method", tree_method.value()}});
    }
    if (device.has_value()) {
      auto const& d = device.value();
      learner->Configure({{"device", d}});
    }
    learner->Configure();
    Context ctx;
    for (std::int32_t i = 0; i < 3; ++i) {
      GradientContainer gpair = GenerateRandomGradients(&ctx, Xy->Info().num_row_, 1);
      learner->BoostOneIter(0, Xy, &gpair);
    }

    Json config{Object{}};
    learner->SaveConfig(&config);
    auto updater = config["learner"]["gradient_booster"]["updater"];
    return updater;
  };

  // |        | hist    | approx | exact | NA  |
  // |--------+---------+--------+-------+-----|
  // | CUDA:0 | GPU     | GPU    | Err   | GPU |
  // | CPU    | CPU     | GPU    | CPU   | CPU |
  // |--------+---------+--------+-------+-----|
  // | NA     | CPU     | CPU    | CPU   | CPU |
  //
  // - CPU: Run on CPU.
  // - GPU: Run on CUDA.
  // - Err: Not feasible.
  // - NA:  Parameter is not specified.
  std::map<std::pair<std::optional<std::string>, std::optional<std::string>>, std::string>
      expectation{
          // hist
          {{"hist", "cpu"}, "grow_quantile_histmaker"},
          {{"hist", "cuda"}, "grow_gpu_hist"},
          {{"hist", "cuda:0"}, "grow_gpu_hist"},
          {{"hist", std::nullopt}, "grow_quantile_histmaker"},
          // approx
          {{"approx", "cpu"}, "grow_histmaker"},
          {{"approx", "cuda"}, "grow_gpu_approx"},
          {{"approx", "cuda:0"}, "grow_gpu_approx"},
          {{"approx", std::nullopt}, "grow_histmaker"},
          // exact
          {{"exact", "cpu"}, "grow_colmaker,prune"},
          {{"exact", "cuda"}, "err"},
          {{"exact", "cuda:0"}, "err"},
          {{"exact", std::nullopt}, "grow_colmaker,prune"},
          // NA
          {{std::nullopt, "cpu"}, "grow_quantile_histmaker"},
          {{std::nullopt, "cuda"}, "grow_gpu_hist"},
          {{std::nullopt, "cuda:0"}, "grow_gpu_hist"},
          {{std::nullopt, std::nullopt}, "grow_quantile_histmaker"},
      };

  auto run_test = [&](auto fn) {
    for (auto const& kv : expectation) {
      auto device = kv.first.second;
      auto tm = kv.first.first;

      if (kv.second == "err") {
        ASSERT_THROW(
            { fn(device, tm); }, dmlc::Error)
            << " device:" << device.value_or("NA") << " tm:" << tm.value_or("NA");
        continue;
      }
      auto up = fn(device, tm);
      auto ups = get<Array const>(up);
      auto exp_names = common::Split(kv.second, ',');
      ASSERT_EQ(exp_names.size(), ups.size());
      for (std::size_t i = 0; i < exp_names.size(); ++i) {
        ASSERT_EQ(get<String const>(ups[i]["name"]), exp_names[i])
            << " device:" << device.value_or("NA") << " tm:" << tm.value_or("NA");
      }
    }
  };

  run_test(with_update);
  run_test(with_boost);
}
#endif  // XGBOOST_USE_CUDA

// Some other parts of test are in `Tree.JsonIO'.
TEST(GBTree, JsonIO) {
  size_t constexpr kRows = 16, kCols = 16;

  Context ctx;
  LearnerModelState mparam{MakeMP(kCols, .5, 1)};

  std::unique_ptr<GradientBooster> gbm{
      CreateTrainedGBM("gbtree", Args{{"tree_method", "exact"}, {"default_direction", "left"}},
                       kRows, kCols, &mparam, &ctx)};

  Json model{Object()};
  model["model"] = Object();
  auto j_model = model["model"];

  model["config"] = Object();
  auto j_config = model["config"];

  gbm->SaveModel(&j_model);
  gbm->SaveConfig(&j_config);

  std::string model_str;
  Json::Dump(model, &model_str);

  model = Json::Load({model_str.c_str(), model_str.size()});
  j_model = model["model"];
  j_config = model["config"];
  ASSERT_EQ(get<String>(j_model["name"]), "gbtree");

  auto gbtree_model = j_model["model"];
  ASSERT_EQ(get<Array>(gbtree_model["trees"]).size(), 1ul);
  ASSERT_EQ(get<Integer>(get<Object>(get<Array>(gbtree_model["trees"]).front()).at("id")), 0);
  ASSERT_EQ(get<Array>(gbtree_model["tree_info"]).size(), 1ul);
  auto j_train_param = j_config["gbtree_model_param"];
  ASSERT_EQ(get<String>(j_train_param["num_parallel_tree"]), "1");

  auto check_config = [](Json j_up_config) {
    auto colmaker = get<Array const>(j_up_config).front();
    auto pruner = get<Array const>(j_up_config).back();
    ASSERT_EQ(get<String const>(colmaker["name"]), "grow_colmaker");
    ASSERT_EQ(get<String const>(pruner["name"]), "prune");
    ASSERT_EQ(get<String const>(colmaker["colmaker_train_param"]["default_direction"]), "left");
  };
  check_config(j_config["updater"]);

  std::unique_ptr<GradientBooster> loaded(gbm::GBTree::Create("gbtree", &ctx, &mparam));
  loaded->LoadModel(j_model);
  loaded->LoadConfig(j_config);

  // roundtrip test
  Json j_config_rt{Object{}};
  loaded->SaveConfig(&j_config_rt);
  check_config(j_config_rt["updater"]);
}

TEST(Dart, JsonIO) {
  size_t constexpr kRows = 16, kCols = 16;

  Context ctx;
  LearnerModelState mparam{MakeMP(kCols, .5, 1)};

  std::unique_ptr<GradientBooster> gbm{
      CreateTrainedGBM("dart", Args{}, kRows, kCols, &mparam, &ctx)};

  Json model{Object()};
  model["model"] = Object();
  auto& j_model = model["model"];
  model["config"] = Object();

  auto& j_param = model["config"];

  gbm->SaveModel(&j_model);
  gbm->SaveConfig(&j_param);

  std::string model_str;
  Json::Dump(model, &model_str);

  model = Json::Load({model_str.c_str(), model_str.size()});

  ASSERT_EQ(get<String>(model["model"]["name"]), "gbtree") << model;
  ASSERT_EQ(get<String>(model["config"]["name"]), "gbtree");
}

TEST(GBTree, LoadLegacyDartJson) {
  size_t constexpr kRows = 16, kCols = 16;

  Context ctx;
  LearnerModelState mparam{MakeMP(kCols, .5, 1)};

  std::unique_ptr<GradientBooster> gbm{
      CreateTrainedGBM("gbtree", Args{{"rate_drop", "0.5"}}, kRows, kCols, &mparam, &ctx)};

  Json model{Object{}};
  Json config{Object{}};
  gbm->SaveModel(&model);
  gbm->SaveConfig(&config);

  Json legacy_model{Object{}};
  legacy_model["name"] = String{"dart"};
  legacy_model["gbtree"] = model;
  legacy_model["weight_drop"] = model["model"]["weight_drop"];
  get<Object>(legacy_model["gbtree"]["model"]).erase("weight_drop");

  Json legacy_config{Object{}};
  legacy_config["name"] = String{"dart"};
  legacy_config["gbtree"] = config;
  legacy_config["dart_train_param"] = config["dart_train_param"];
  get<Object>(legacy_config["gbtree"]).erase("dart_train_param");

  std::unique_ptr<GradientBooster> loaded{GradientBooster::Create("dart", &ctx, &mparam)};
  loaded->LoadModel(legacy_model);
  loaded->LoadConfig(legacy_config);

  Json canonical_model{Object{}};
  Json canonical_config{Object{}};
  loaded->SaveModel(&canonical_model);
  loaded->SaveConfig(&canonical_config);

  ASSERT_EQ(get<String>(canonical_model["name"]), "gbtree");
  ASSERT_EQ(get<String>(canonical_config["name"]), "gbtree");
  ASSERT_NE(get<Array>(canonical_model["model"]["weight_drop"]).size(), 0ul);
  ASSERT_TRUE(IsA<Object>(canonical_config["dart_train_param"]));
}

TEST(GBTree, DropoutJsonIO) {
  size_t constexpr kRows = 16, kCols = 16;

  Context ctx;
  LearnerModelState mparam{MakeMP(kCols, .5, 1)};

  std::unique_ptr<GradientBooster> gbm{
      CreateTrainedGBM("gbtree", Args{{"rate_drop", "0.5"}}, kRows, kCols, &mparam, &ctx)};

  Json model{Object()};
  model["model"] = Object();
  auto& j_model = model["model"];
  model["config"] = Object();
  auto& j_param = model["config"];

  gbm->SaveModel(&j_model);
  gbm->SaveConfig(&j_param);

  ASSERT_EQ(get<String>(model["model"]["name"]), "gbtree") << model;
  ASSERT_EQ(get<String>(model["config"]["name"]), "gbtree");
  ASSERT_NE(get<Array>(model["model"]["model"]["weight_drop"]).size(), 0ul);
  ASSERT_TRUE(IsA<Object>(model["config"]["dart_train_param"]));

  std::string malformed_str = Json::Dump(j_model);
  auto malformed_model = Json::Load(StringView{malformed_str});
  auto n_trees = get<Array const>(malformed_model["model"]["trees"]).size();
  std::vector<Json> wrong_weights(n_trees + 1);
  for (auto& weight : wrong_weights) {
    weight = Number{1.0};
  }
  malformed_model["model"]["weight_drop"] = Array{std::move(wrong_weights)};
  std::unique_ptr<GradientBooster> invalid{GradientBooster::Create("gbtree", &ctx, &mparam)};
  EXPECT_ANY_THROW(invalid->LoadModel(malformed_model));

  auto legacy_model = j_model;
  legacy_model["weight_drop"] = legacy_model["model"]["weight_drop"];
  get<Object>(legacy_model["model"]).erase("weight_drop");
  std::unique_ptr<GradientBooster> loaded{GradientBooster::Create("gbtree", &ctx, &mparam)};
  loaded->LoadModel(legacy_model);
  Json canonical_model{Object{}};
  loaded->SaveModel(&canonical_model);
  ASSERT_NE(get<Array>(canonical_model["model"]["weight_drop"]).size(), 0ul);
}

namespace {
class Dart : public testing::TestWithParam<char const*> {
 public:
  void Run(std::string device) {
    size_t constexpr kRows = 16, kCols = 10;

    HostDeviceVector<float> data;
    Context ctx;
    if (device == "GPU") {
      ctx = MakeCUDACtx(0);
    }
    auto rng = RandomDataGenerator(kRows, kCols, 0).Device(ctx.Device());
    auto array_str = rng.GenerateArrayInterface(&data);
    auto p_mat = GetDMatrixFromData(data.HostVector(), kRows, kCols);

    std::vector<bst_float> labels(kRows);
    for (size_t i = 0; i < kRows; ++i) {
      labels[i] = i % 2;
    }
    p_mat->SetInfo("label", Make1dInterfaceTest(labels.data(), kRows));

    auto learner = std::unique_ptr<Learner>(Learner::Create({p_mat}));
    learner->Configure({{"booster", "dart"}});
    learner->Configure({{"rate_drop", "0.5"}});
    learner->Configure();

    for (size_t i = 0; i < 16; ++i) {
      learner->UpdateOneIter(i, p_mat);
    }
    learner->Configure({{"device", ctx.DeviceName()}});

    HostDeviceVector<float> predts_training;
    learner->Predict(p_mat, false, &predts_training, 0, 0, true);

    HostDeviceVector<float>* inplace_predts;
    std::shared_ptr<data::DMatrixProxy> x{new data::DMatrixProxy{}};
    if (ctx.IsCUDA()) {
      x->SetCudaArray(array_str.c_str());
    } else {
      x->SetArray(array_str.c_str());
    }
    learner->InplacePredict(x, PredictionType::kValue, std::numeric_limits<float>::quiet_NaN(),
                            &inplace_predts, 0, 0);
    CHECK(inplace_predts);

    HostDeviceVector<float> predts_inference;
    learner->Predict(p_mat, false, &predts_inference, 0, 0, false);

    auto const& h_predts_training = predts_training.ConstHostVector();
    auto const& h_predts_inference = predts_inference.ConstHostVector();
    auto const& h_inplace_predts = inplace_predts->HostVector();
    ASSERT_EQ(h_predts_training.size(), h_predts_inference.size());
    ASSERT_EQ(h_inplace_predts.size(), h_predts_inference.size());
    for (size_t i = 0; i < predts_inference.Size(); ++i) {
      // Inference doesn't drop tree.
      ASSERT_GT(std::abs(h_predts_training[i] - h_predts_inference[i]), kRtEps * 10);
      // Inplace prediction is inference.
      ASSERT_LT(h_inplace_predts[i] - h_predts_inference[i], kRtEps / 10);
    }
  }
};
}  // anonymous namespace

TEST_P(Dart, Prediction) { this->Run(GetParam()); }

#if defined(XGBOOST_USE_CUDA)
INSTANTIATE_TEST_SUITE_P(PredictorTypes, Dart, testing::Values("CPU", "GPU"));
#else
INSTANTIATE_TEST_SUITE_P(PredictorTypes, Dart, testing::Values("CPU"));
#endif  // defined(XGBOOST_USE_CUDA)

std::pair<Json, Json> TestModelSlice(std::string booster) {
  size_t constexpr kRows = 1000, kCols = 100, kForest = 2, kClasses = 3;
  auto m = RandomDataGenerator{kRows, kCols, 0}.Classes(kClasses).GenerateDMatrix(true);

  int32_t kIters = 10;
  std::unique_ptr<Learner> learner{Learner::Create({m})};
  Args args{{"booster", booster},
            {"tree_method", "hist"},
            {"num_parallel_tree", std::to_string(kForest)},
            {"num_class", std::to_string(kClasses)},
            {"subsample", "0.5"},
            {"max_depth", "2"}};
  if (booster == "dart") {
    args.emplace_back("rate_drop", "0.5");
  }
  learner->Configure(args);

  for (auto i = 0; i < kIters; ++i) {
    learner->UpdateOneIter(i, m);
  }

  Json model{Object()};
  Json config{Object()};
  learner->SaveModel(&model);
  learner->SaveConfig(&config);
  bool out_of_bound = false;

  size_t constexpr kSliceStart = 2, kSliceEnd = 8, kStep = 3;
  std::unique_ptr<Learner> sliced{learner->Slice(kSliceStart, kSliceEnd, kStep, &out_of_bound)};
  Json sliced_model{Object()};
  sliced->SaveModel(&sliced_model);

  auto get_gbtree = [](Json const& model) -> Json const& {
    auto const& booster = model["learner"]["gradient_booster"];
    auto const& obj = get<Object const>(booster);
    auto it = obj.find("model");
    if (it != obj.cend() && IsA<Object>(it->second)) {
      return booster;
    }
    return obj.at("gbtree");
  };

  auto get_gbtree_config = [](Json& model) -> Json& {
    auto& booster = model["learner"]["gradient_booster"];
    auto& obj = get<Object>(booster);
    auto it = obj.find("gbtree_model_param");
    if (it != obj.cend() && IsA<Object>(it->second)) {
      return booster;
    }
    return obj.at("gbtree");
  };

  auto get_shape = [&](Json const& model) {
    auto const& gbtree = get_gbtree(model);
    return get<Object const>(gbtree["model"]["gbtree_model_param"]);
  };

  auto const& model_shape = get_shape(sliced_model);
  CHECK_EQ(get<String const>(model_shape.at("num_trees")), std::to_string(2 * kClasses * kForest));

  Json sliced_config{Object()};
  sliced->SaveConfig(&sliced_config);
  // Only num trees is changed
  auto& gradient_booster = get_gbtree_config(sliced_config);
  gradient_booster["gbtree_model_param"]["num_trees"] = String("60");
  CHECK_EQ(sliced_config, config);

  auto get_trees = [&](Json const& model) {
    auto const& gbtree = get_gbtree(model);
    return get<Array const>(gbtree["model"]["trees"]);
  };

  auto get_info = [&](Json const& model) {
    auto const& gbtree = get_gbtree(model);
    return get<Array const>(gbtree["model"]["tree_info"]);
  };

  auto const& sliced_trees = get_trees(sliced_model);
  CHECK_EQ(sliced_trees.size(), 2 * kClasses * kForest);

  auto constexpr kLayerSize = kClasses * kForest;
  auto const& sliced_info = get_info(sliced_model);

  for (size_t layer = 0; layer < 2; ++layer) {
    for (size_t j = 0; j < kClasses; ++j) {
      for (size_t k = 0; k < kForest; ++k) {
        auto idx = layer * kLayerSize + j * kForest + k;
        auto const& group = get<Integer const>(sliced_info.at(idx));
        CHECK_EQ(static_cast<size_t>(group), j);
      }
    }
  }

  auto const& trees = get_trees(model);

  // Sliced layers are [2, 5]
  auto begin = kLayerSize * kSliceStart;
  auto end = begin + kLayerSize;
  auto j = 0;
  for (size_t i = begin; i < end; ++i) {
    Json tree = trees[i];
    tree["id"] = Integer(0);  // id is different, we set it to 0 to allow comparison.
    auto sliced_tree = sliced_trees[j];
    sliced_tree["id"] = Integer(0);
    CHECK_EQ(tree, sliced_tree);
    j++;
  }

  begin = kLayerSize * (kSliceStart + kStep);
  end = begin + kLayerSize;
  for (size_t i = begin; i < end; ++i) {
    Json tree = trees[i];
    tree["id"] = Integer(0);
    auto sliced_tree = sliced_trees[j];
    sliced_tree["id"] = Integer(0);
    CHECK_EQ(tree, sliced_tree);
    j++;
  }

  // CHECK sliced model doesn't have dependency on the old one
  learner.reset();
  CHECK_EQ(sliced->GetNumFeature(), kCols);

  return std::make_pair(model, sliced_model);
}

TEST(GBTree, Slice) { TestModelSlice("gbtree"); }

TEST(Dart, Slice) {
  auto [model, sliced_model] = TestModelSlice("dart");
  auto const& weights =
      get<Array const>(model["learner"]["gradient_booster"]["model"]["weight_drop"]);
  auto const& trees = get<Array const>(model["learner"]["gradient_booster"]["model"]["trees"]);
  ASSERT_EQ(weights.size(), trees.size());
}

TEST(GBTree, FeatureScore) {
  size_t n_samples = 1000, n_features = 10, n_classes = 4;
  auto m = RandomDataGenerator{n_samples, n_features, 0.5}.Classes(n_classes).GenerateDMatrix(true);

  std::unique_ptr<Learner> learner{Learner::Create({m})};
  learner->Configure({{"num_class", std::to_string(n_classes)}});

  learner->Configure();
  for (size_t i = 0; i < 2; ++i) {
    learner->UpdateOneIter(i, m);
  }

  std::vector<bst_feature_t> features_weight;
  std::vector<float> scores_weight;
  learner->CalcFeatureScore("weight", {}, &features_weight, &scores_weight);
  ASSERT_EQ(features_weight.size(), scores_weight.size());
  ASSERT_LE(features_weight.size(), learner->GetNumFeature());
  ASSERT_TRUE(std::is_sorted(features_weight.begin(), features_weight.end()));

  auto test_eq = [&learner, &scores_weight](std::string type) {
    std::vector<bst_feature_t> features;
    std::vector<float> scores;
    learner->CalcFeatureScore(type, {}, &features, &scores);

    std::vector<bst_feature_t> features_total;
    std::vector<float> scores_total;
    learner->CalcFeatureScore("total_" + type, {}, &features_total, &scores_total);

    for (size_t i = 0; i < scores_weight.size(); ++i) {
      ASSERT_LE(RelError(scores_total[i] / scores[i], scores_weight[i]), kRtEps);
    }
  };

  test_eq("gain");
  test_eq("cover");
}

TEST(GBTree, PredictRange) {
  size_t n_samples = 1000, n_features = 10, n_classes = 4;
  auto m = RandomDataGenerator{n_samples, n_features, 0.5}.Classes(n_classes).GenerateDMatrix(true);

  std::unique_ptr<Learner> learner{Learner::Create({m})};
  learner->Configure({{"num_class", std::to_string(n_classes)}});

  learner->Configure();
  for (size_t i = 0; i < 2; ++i) {
    learner->UpdateOneIter(i, m);
  }
  HostDeviceVector<float> out_predt;
  ASSERT_THROW(learner->Predict(m, false, &out_predt, 0, 3), dmlc::Error);

  auto m_1 =
      RandomDataGenerator{n_samples, n_features, 0.5}.Classes(n_classes).GenerateDMatrix(true);
  HostDeviceVector<float> out_predt_full;
  learner->Predict(m_1, false, &out_predt_full, 0, 0);
  ASSERT_TRUE(std::equal(out_predt.HostVector().begin(), out_predt.HostVector().end(),
                         out_predt_full.HostVector().begin()));

  {
    // inplace predict
    HostDeviceVector<float> raw_storage;
    auto raw = RandomDataGenerator{n_samples, n_features, 0.5}.GenerateArrayInterface(&raw_storage);
    std::shared_ptr<data::DMatrixProxy> x{new data::DMatrixProxy{}};
    x->SetArray(raw.data());

    HostDeviceVector<float>* out_predt;
    learner->InplacePredict(x, PredictionType::kValue, std::numeric_limits<float>::quiet_NaN(),
                            &out_predt, 0, 2);
    auto h_out_predt = out_predt->HostVector();
    learner->InplacePredict(x, PredictionType::kValue, std::numeric_limits<float>::quiet_NaN(),
                            &out_predt, 0, 0);
    auto h_out_predt_full = out_predt->HostVector();

    ASSERT_TRUE(std::equal(h_out_predt.begin(), h_out_predt.end(), h_out_predt_full.begin()));
    // Out of range.
    ASSERT_THROW(learner->InplacePredict(x, PredictionType::kValue,
                                         std::numeric_limits<float>::quiet_NaN(), &out_predt, 0, 3),
                 dmlc::Error);
  }
}

TEST(GBTree, InplacePredictionError) {
  std::size_t n_samples{2048}, n_features{32};

  auto test_ext_err = [&](std::string booster, Context const* ctx) {
    std::shared_ptr<DMatrix> p_fmat =
        RandomDataGenerator{n_samples, n_features, 0.5f}.Batches(2).GenerateSparsePageDMatrix(
            "cache", true);
    std::unique_ptr<Learner> learner{Learner::Create({p_fmat})};
    learner->Configure(Args{{"booster", booster}, {"device", ctx->DeviceName()}});
    learner->Configure();
    for (std::int32_t i = 0; i < 3; ++i) {
      learner->UpdateOneIter(i, p_fmat);
    }
    HostDeviceVector<float>* out_predt;
    ASSERT_THROW(
        {
          learner->InplacePredict(p_fmat, PredictionType::kValue,
                                  std::numeric_limits<float>::quiet_NaN(), &out_predt, 0, 0);
        },
        dmlc::Error);
  };

  {
    Context ctx;
    test_ext_err("gbtree", &ctx);
    test_ext_err("dart", &ctx);
  }

#if defined(XGBOOST_USE_CUDA)
  {
    auto ctx = MakeCUDACtx(0);
    test_ext_err("gbtree", &ctx);
    test_ext_err("dart", &ctx);
  }
#endif  // defined(XGBOOST_USE_CUDA)

  auto test_qdm_err = [&](std::string booster, Context const* ctx) {
    std::shared_ptr<DMatrix> p_fmat;
    bst_bin_t max_bins = 16;
    auto rng =
        RandomDataGenerator{n_samples, n_features, 0.5f}.Device(ctx->Device()).Bins(max_bins);
    if (ctx->IsCPU()) {
      p_fmat = rng.GenerateQuantileDMatrix(true);
    } else {
#if defined(XGBOOST_USE_CUDA)
      p_fmat = rng.Device(ctx->Device()).GenerateQuantileDMatrix(true);
#else
      CHECK(p_fmat);
#endif  // defined(XGBOOST_USE_CUDA)
    }
    std::unique_ptr<Learner> learner{Learner::Create({p_fmat})};
    learner->Configure(Args{{"booster", booster},
                            {"max_bin", std::to_string(max_bins)},
                            {"device", ctx->DeviceName()}});
    learner->Configure();
    for (std::int32_t i = 0; i < 3; ++i) {
      learner->UpdateOneIter(i, p_fmat);
    }
    HostDeviceVector<float>* out_predt;
    ASSERT_THROW(
        {
          learner->InplacePredict(p_fmat, PredictionType::kValue,
                                  std::numeric_limits<float>::quiet_NaN(), &out_predt, 0, 0);
        },
        dmlc::Error);
  };

  {
    Context ctx;
    test_qdm_err("gbtree", &ctx);
    test_qdm_err("dart", &ctx);
  }

#if defined(XGBOOST_USE_CUDA)
  {
    auto ctx = MakeCUDACtx(0);
    test_qdm_err("gbtree", &ctx);
    test_qdm_err("dart", &ctx);
  }
#endif  // defined(XGBOOST_USE_CUDA)
}

namespace {
bst_idx_t constexpr kTargetRows = 512;
bst_feature_t constexpr kTargetCols = 8;
bst_target_t constexpr kTargets = 3;

// Whether BoostTargets can build the targets at the same time with n_threads threads.
bool ConcurrentTargets(std::int32_t n_threads) {
#if defined(XGBOOST_USE_THREADPOOL)
  return false;
#else
  return AllThreadsForTest() >= n_threads;
#endif  // defined(XGBOOST_USE_THREADPOOL)
}

struct TargetTreesResult {
  std::string model;
  std::vector<std::int32_t> worker_threads;
};

// Boosts kTargets scalar-leaf targets for four iterations directly on a GBTree. With deep, the
// last two iterations run from a deeper stack frame filled with garbage.
TargetTreesResult TrainTargets(std::shared_ptr<DMatrix> p_fmat, std::int32_t n_threads,
                               bool parallel, Args const& extra = {}, std::uint32_t seed = 0,
                               bool deep = false);

#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
void DoBoostFromDeeperFrame(gbm::GBTree* gbtree, std::shared_ptr<DMatrix> p_fmat,
                            GradientContainer* gpair) {
  std::size_t constexpr kBytes = 1 << 16;
  char volatile garbage[kBytes];  // NOLINT
  for (std::size_t i = 0; i < kBytes; ++i) {
    garbage[i] = static_cast<char>(0xFF);
  }
  gbtree->DoBoost(p_fmat, gpair, nullptr);
  garbage[0] = garbage[1];
}

std::string DumpModel(gbm::GBTree const& gbtree) {
  Json model{Object{}};
  gbtree.SaveModel(&model);
  std::string str;
  Json::Dump(model, &str);
  return str;
}

TargetTreesResult TrainTargets(std::shared_ptr<DMatrix> p_fmat, std::int32_t n_threads,
                               bool parallel, Args const& extra, std::uint32_t seed, bool deep) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"nthread", std::to_string(n_threads)}});
  ctx.Rng().seed(seed);
  LearnerModelState mparam{MakeMP(kTargetCols, .5, kTargets)};
  std::unique_ptr<GradientBooster> p_gbm{GradientBooster::Create("gbtree", &ctx, &mparam)};
  auto& gbtree = dynamic_cast<gbm::GBTree&>(*p_gbm);
  Args args{{"tree_method", "hist"}, {"parallel_target_trees", parallel ? "true" : "false"}};
  args.insert(args.end(), extra.cbegin(), extra.cend());
  gbtree.Configure(args);
  auto gpair = GenerateRandomGradients(&ctx, kTargetRows, kTargets);
  for (std::int32_t i = 0; i < 4; ++i) {
    if (deep && i >= 2) {
      DoBoostFromDeeperFrame(&gbtree, p_fmat, &gpair);
    } else {
      gbtree.DoBoost(p_fmat, &gpair, nullptr);
    }
  }
  return {DumpModel(gbtree), gbtree.TargetWorkerThreadsForTest()};
}

// The trees of one target, in the order they were built.
std::vector<std::string> TargetTrees(std::string const& model, bst_target_t gid) {
  auto jmodel = Json::Load(StringView{model});
  auto const& trees = get<Array const>(jmodel["model"]["trees"]);
  auto const& tree_info = get<Array const>(jmodel["model"]["tree_info"]);
  std::vector<std::string> out;
  for (std::size_t i = 0; i < trees.size(); ++i) {
    if (get<Integer const>(tree_info[i]) == static_cast<Integer::Int>(gid)) {
      std::string str;
      Json::Dump(trees[i], &str);
      out.push_back(str);
    }
  }
  return out;
}

std::shared_ptr<DMatrix> TargetQuantileDMatrix() {
  return RandomDataGenerator{kTargetRows, kTargetCols, 0.2}.Bins(256).GenerateQuantileDMatrix(
      false);
}

// Runs fn and returns what it logs at the warning level.
template <typename Fn>
std::string CaptureWarnings(Fn&& fn) {
  auto const verbosity = static_cast<std::int32_t>(ConsoleLogger::GlobalVerbosity());
  ConsoleLogger::Configure({{"verbosity", "1"}});
  testing::internal::CaptureStderr();
  try {
    fn();
  } catch (...) {
    std::ignore = testing::internal::GetCapturedStderr();
    ConsoleLogger::Configure({{"verbosity", std::to_string(verbosity)}});
    throw;
  }
  auto output = testing::internal::GetCapturedStderr();
  ConsoleLogger::Configure({{"verbosity", std::to_string(verbosity)}});
  return output;
}

std::size_t CountOf(std::string const& haystack, std::string const& needle) {
  std::size_t n{0};
  for (auto pos = haystack.find(needle); pos != std::string::npos;
       pos = haystack.find(needle, pos + 1)) {
    ++n;
  }
  return n;
}

char const kFallback[] = "`parallel_target_trees` builds the targets one after another";

// Builds no trees, and throws when it runs on a thread other than the one that created it, so
// only the workers on the pool's threads throw.
std::thread::id test_updater_thread;
class ThrowOffCallingThread : public TreeUpdater {
 public:
  explicit ThrowOffCallingThread(Context const* ctx) : TreeUpdater{ctx} {}
  std::set<std::string> Configure(Args const&) override { return {}; }
  void LoadConfig(Json const&) override {}
  void SaveConfig(Json*) const override {}
  [[nodiscard]] char const* Name() const override { return "test_throw_off_calling_thread"; }
  [[nodiscard]] bool PrepareConcurrentBuild(tree::TrainParam const*, DMatrix*) override {
    return true;
  }
  void Update(tree::TrainParam const*, GradientContainer*, DMatrix*,
              common::Span<HostDeviceVector<bst_node_t>>, std::vector<RegTree*> const&) override {
    if (std::this_thread::get_id() != test_updater_thread) {
      LOG(FATAL) << "Thrown off the calling thread.";
    }
  }
};

XGBOOST_REGISTER_TREE_UPDATER(ThrowOffCallingThread, "test_throw_off_calling_thread")
    .describe("Test-only updater that throws on the threads of BoostTargets' pool.")
    .set_body([](Context const* ctx, ObjInfo const*) { return new ThrowOffCallingThread{ctx}; });
}  // namespace

TEST(GBTree, ParallelTargetTrees) {
  if (!ConcurrentTargets(3)) {
    GTEST_SKIP() << "Requires three threads and an OpenMP build.";
  }
  // With one thread per worker the trees match a serial build: a DMatrix gets its gradient
  // index before the first concurrent build, a QuantileDMatrix has it from the start.
  std::vector<std::int32_t> const kThree{1, 1, 1};
  for (auto const& p_fmat : {RandomDataGenerator{kTargetRows, kTargetCols, 0.2}.GenerateDMatrix(),
                             TargetQuantileDMatrix()}) {
    auto expected = TrainTargets(p_fmat, 1, false);
    ASSERT_TRUE(expected.worker_threads.empty());
    // Three threads give each target a worker; two make the first worker build two targets.
    for (auto const& [n_threads, worker_threads] :
         {std::pair{3, std::vector<std::int32_t>{1, 1, 1}},
          std::pair{2, std::vector<std::int32_t>{1, 1}}}) {
      auto result = TrainTargets(p_fmat, n_threads, true);
      ASSERT_EQ(result.worker_threads, worker_threads);
      ASSERT_EQ(result.model, expected.model);
    }
    // Two trees per target and iteration.
    Args forest{{"num_parallel_tree", "2"}};
    auto forest_result = TrainTargets(p_fmat, 3, true, forest);
    ASSERT_EQ(forest_result.worker_threads, kThree);
    ASSERT_EQ(forest_result.model, TrainTargets(p_fmat, 1, false, forest).model);
    // The updaters keep the training parameter's address across iterations.
    auto deep = TrainTargets(p_fmat, 3, true, {}, 0, true);
    ASSERT_EQ(deep.worker_threads, kThree);
    ASSERT_EQ(deep.model, expected.model);
  }
}

TEST(GBTree, ParallelTargetTreesUnevenThreads) {
  if (!ConcurrentTargets(4)) {
    GTEST_SKIP() << "Requires four threads and an OpenMP build.";
  }
  // With four threads the first worker has two. Each target's trees match a serial build with
  // the same number of threads.
  auto p_fmat = TargetQuantileDMatrix();
  auto result = TrainTargets(p_fmat, 4, true);
  ASSERT_EQ(result.worker_threads, (std::vector<std::int32_t>{2, 1, 1}));
  auto two_threads = TrainTargets(p_fmat, 2, false).model;
  auto one_thread = TrainTargets(p_fmat, 1, false).model;
  ASSERT_EQ(TargetTrees(result.model, 0), TargetTrees(two_threads, 0));
  for (bst_target_t gid = 1; gid < kTargets; ++gid) {
    ASSERT_EQ(TargetTrees(result.model, gid), TargetTrees(one_thread, gid));
  }
}

TEST(GBTree, ParallelTargetTreesSampling) {
  if (!ConcurrentTargets(3)) {
    GTEST_SKIP() << "Requires three threads and an OpenMP build.";
  }
  // Each target samples from its own seed, so the model doesn't depend on the number of
  // workers, or on whether the targets are built at the same time at all (one thread).
  auto p_fmat = TargetQuantileDMatrix();
  Args sampling{{"subsample", "0.8"}, {"colsample_bynode", "0.5"}};
  auto serial = TrainTargets(p_fmat, 1, true, sampling);
  ASSERT_TRUE(serial.worker_threads.empty());
  for (auto const& [n_threads, worker_threads] : {std::pair{3, std::vector<std::int32_t>{1, 1, 1}},
                                                  std::pair{2, std::vector<std::int32_t>{1, 1}}}) {
    auto result = TrainTargets(p_fmat, n_threads, true, sampling);
    ASSERT_EQ(result.worker_threads, worker_threads);
    ASSERT_EQ(result.model, serial.model);
  }
  // The seeds come from the booster's generator.
  ASSERT_NE(TrainTargets(p_fmat, 3, true, sampling, 1).model, serial.model);
}

TEST(GBTree, ParallelTargetTreesWorkers) {
  if (!ConcurrentTargets(4)) {
    GTEST_SKIP() << "Requires four threads and an OpenMP build.";
  }
  // DoBoost updates the training prediction cache from each target's leaf positions.
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"nthread", "4"}});
  LearnerModelState mparam{MakeMP(kTargetCols, .5, kTargets)};
  std::unique_ptr<GradientBooster> p_gbm{GradientBooster::Create("gbtree", &ctx, &mparam)};
  auto& gbtree = dynamic_cast<gbm::GBTree&>(*p_gbm);
  Args args{{"tree_method", "hist"}, {"parallel_target_trees", "true"}};
  gbtree.Configure(args);

  auto p_fmat = RandomDataGenerator{kTargetRows, kTargetCols, 0.2}.GenerateDMatrix();
  auto p_copy = RandomDataGenerator{kTargetRows, kTargetCols, 0.2}.GenerateDMatrix();
  auto gpair = GenerateRandomGradients(&ctx, kTargetRows, kTargets);
  std::int32_t version{0};
  auto boost = [&] {
    gbtree.DoBoost(p_fmat, &gpair, nullptr);
    ++version;
    auto const& cache = gbtree.PredictionCache(p_fmat.get());
    ASSERT_EQ(cache.version, version);
    HostDeviceVector<float> fresh;
    gbtree.PredictBatch(p_copy, &fresh, false, 0, 0);
    auto const& h_cached = cache.predictions.ConstHostVector();
    auto const& h_fresh = fresh.ConstHostVector();
    ASSERT_EQ(h_cached.size(), kTargetRows * kTargets);
    ASSERT_EQ(h_cached.size(), h_fresh.size());
    for (std::size_t i = 0; i < h_cached.size(); ++i) {
      ASSERT_NEAR(h_cached[i], h_fresh[i], 1e-6);
    }
  };
  for (std::int32_t i = 0; i < 2; ++i) {
    boost();
    ASSERT_EQ(gbtree.TargetWorkerThreadsForTest(), (std::vector<std::int32_t>{2, 1, 1}));
  }
  ASSERT_EQ(gbtree.TargetWorkerBuildsForTest(), 1);

  // A different thread split rebuilds the workers, even with the same number of workers.
  for (auto const& [n_threads, worker_threads] : {std::pair{3, std::vector<std::int32_t>{1, 1, 1}},
                                                  std::pair{2, std::vector<std::int32_t>{1, 1}}}) {
    ctx.UpdateAllowUnknown(Args{{"nthread", std::to_string(n_threads)}});
    boost();
    ASSERT_EQ(gbtree.TargetWorkerThreadsForTest(), worker_threads);
  }
  ASSERT_EQ(gbtree.TargetWorkerBuildsForTest(), 3);

  // Configuring the same parameters again, as a learning rate schedule does every iteration,
  // keeps the workers. A change to the updaters' configuration rebuilds them.
  gbtree.Configure(args);
  boost();
  ASSERT_EQ(gbtree.TargetWorkerBuildsForTest(), 3);
  gbtree.Configure(Args{{"max_cached_hist_node", "1024"}});
  boost();
  ASSERT_EQ(gbtree.TargetWorkerBuildsForTest(), 4);

  // LoadConfig drops the workers, and the next iteration recreates them.
  Json config{Object{}};
  gbtree.SaveConfig(&config);
  gbtree.LoadConfig(config);
  ASSERT_TRUE(gbtree.TargetWorkerThreadsForTest().empty());
  boost();
  ASSERT_EQ(gbtree.TargetWorkerThreadsForTest(), (std::vector<std::int32_t>{1, 1}));
  ASSERT_EQ(gbtree.TargetWorkerBuildsForTest(), 5);

  // Building the targets one after another frees the workers.
  ctx.UpdateAllowUnknown(Args{{"nthread", "1"}});
  std::ignore = CaptureWarnings([&] { boost(); });
  ASSERT_TRUE(gbtree.TargetWorkerThreadsForTest().empty());
}

TEST(GBTree, ParallelTargetTreesFallback) {
  if (!ConcurrentTargets(3)) {
    GTEST_SKIP() << "Requires three threads and an OpenMP build.";
  }
  // Unsupported setups build the targets one after another, each logging its own reason.
  common::TemporaryDirectory tmpdir;
  auto p_extmem =
      RandomDataGenerator{kTargetRows, kTargetCols, 0.2}.Batches(2).GenerateSparsePageDMatrix(
          (tmpdir.Path() / "cache").string(), true);
  auto p_fmat = RandomDataGenerator{kTargetRows, kTargetCols, 0.2}.GenerateDMatrix();
  auto check = [](auto train, std::string const& reason) {
    TargetTreesResult result;
    auto output = CaptureWarnings([&] { result = train(); });
    ASSERT_TRUE(result.worker_threads.empty()) << reason;
    ASSERT_EQ(CountOf(output, std::string(kFallback) + ": " + reason), 1) << output;
  };
  check([&] { return TrainTargets(p_extmem, 3, true); },
        "the `grow_quantile_histmaker` updater doesn't support it with this data");
  check([&] { return TrainTargets(p_fmat, 3, true, {{"tree_method", "approx"}}); },
        "the `grow_histmaker` updater doesn't support it with this data");
  check([&] { return TrainTargets(p_fmat, 1, true); }, "it needs more than one thread");
}

#if defined(XGBOOST_USE_THREADPOOL)
TEST(GBTree, ParallelTargetTreesThreadPoolFallback) {
  // The workers' loops would share the one global pool, so the targets are built one after
  // another.
  auto p_fmat = TargetQuantileDMatrix();
  TargetTreesResult result;
  auto output = CaptureWarnings([&] { result = TrainTargets(p_fmat, 3, true); });
  ASSERT_TRUE(result.worker_threads.empty());
  ASSERT_EQ(CountOf(output, std::string(kFallback) +
                                ": this build runs parallel loops on one shared thread pool"),
            1)
      << output;
}
#endif  // defined(XGBOOST_USE_THREADPOOL)

TEST(GBTree, ParallelTargetTreesFallbackWarning) {
  if (!ConcurrentTargets(3)) {
    GTEST_SKIP() << "Requires three threads and an OpenMP build.";
  }
  // The fallback warns once per reason, so a learning rate schedule, which configures the
  // booster every iteration, doesn't repeat it.
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"nthread", "3"}});
  LearnerModelState mparam{MakeMP(kTargetCols, .5, kTargets)};
  std::unique_ptr<GradientBooster> p_gbm{GradientBooster::Create("gbtree", &ctx, &mparam)};
  Args args{{"tree_method", "approx"}, {"parallel_target_trees", "true"}};
  p_gbm->Configure(args);
  auto p_fmat = RandomDataGenerator{kTargetRows, kTargetCols, 0.2}.GenerateDMatrix();
  auto gpair = GenerateRandomGradients(&ctx, kTargetRows, kTargets);
  auto n_warnings = [&](std::int32_t n_iters) {
    auto output = CaptureWarnings([&] {
      for (std::int32_t i = 0; i < n_iters; ++i) {
        p_gbm->Configure(args);
        p_gbm->DoBoost(p_fmat, &gpair, nullptr);
      }
    });
    return CountOf(output, kFallback);
  };
  EXPECT_EQ(n_warnings(2), 1);
  EXPECT_EQ(n_warnings(1), 0);
  // A different reason is logged.
  ctx.UpdateAllowUnknown(Args{{"nthread", "1"}});
  EXPECT_EQ(n_warnings(1), 1);
}

TEST(GBTree, ParallelTargetTreesUpdateProcess) {
  if (!ConcurrentTargets(3)) {
    GTEST_SKIP() << "Requires three threads and an OpenMP build.";
  }
  // Updating existing trees builds the targets one after another, with the same result.
  auto p_fmat = RandomDataGenerator{kTargetRows, kTargetCols, 0.2}.GenerateDMatrix();
  auto refresh = [&](bool parallel) {
    Context ctx;
    ctx.UpdateAllowUnknown(Args{{"nthread", "3"}});
    LearnerModelState mparam{MakeMP(kTargetCols, .5, kTargets)};
    std::unique_ptr<GradientBooster> p_gbm{GradientBooster::Create("gbtree", &ctx, &mparam)};
    auto& gbtree = dynamic_cast<gbm::GBTree&>(*p_gbm);
    gbtree.Configure(Args{{"tree_method", "hist"}});
    auto gpair = GenerateRandomGradients(&ctx, kTargetRows, kTargets);
    for (std::int32_t i = 0; i < 2; ++i) {
      gbtree.DoBoost(p_fmat, &gpair, nullptr);
    }
    gbtree.Configure(Args{{"process_type", "update"},
                          {"updater", "refresh"},
                          {"parallel_target_trees", parallel ? "true" : "false"}});
    std::string output = CaptureWarnings([&] { gbtree.DoBoost(p_fmat, &gpair, nullptr); });
    EXPECT_TRUE(gbtree.TargetWorkerThreadsForTest().empty());
    if (parallel) {
      EXPECT_EQ(
          CountOf(output, std::string(kFallback) + ": it supports only `process_type=default`"), 1)
          << output;
    }
    return DumpModel(gbtree);
  };
  ASSERT_EQ(refresh(true), refresh(false));
}

TEST(GBTree, ParallelTargetTreesError) {
  if (!ConcurrentTargets(3)) {
    GTEST_SKIP() << "Requires three threads and an OpenMP build.";
  }
  // An error in the updaters reaches the caller. More constraints than features fail in every
  // worker.
  auto p_fmat = TargetQuantileDMatrix();
  ASSERT_THROW(TrainTargets(p_fmat, 3, true, {{"monotone_constraints", "(1,1,1,1,1,1,1,1,1)"}}),
               dmlc::Error);
  // An error only on the pool's threads reaches the caller too.
  test_updater_thread = std::this_thread::get_id();
  ASSERT_THROW(TrainTargets(p_fmat, 3, true, {{"updater", "test_throw_off_calling_thread"}}),
               dmlc::Error);
}

#if !defined(_WIN32)
TEST(GBTree, ParallelTargetTreesFork) {
  if (!ConcurrentTargets(3)) {
    GTEST_SKIP() << "Requires three threads and an OpenMP build.";
  }
  // A forked child has none of the pool's threads. Training, reloading the configuration and
  // freeing the booster in the child must not wait for or join them.
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"nthread", "3"}});
  LearnerModelState mparam{MakeMP(kTargetCols, .5, kTargets)};
  std::unique_ptr<GradientBooster> p_gbm{GradientBooster::Create("gbtree", &ctx, &mparam)};
  auto& gbtree = dynamic_cast<gbm::GBTree&>(*p_gbm);
  gbtree.Configure(Args{{"tree_method", "hist"}, {"parallel_target_trees", "true"}});
  auto p_fmat = TargetQuantileDMatrix();
  auto gpair = GenerateRandomGradients(&ctx, kTargetRows, kTargets);
  gbtree.DoBoost(p_fmat, &gpair, nullptr);
  ASSERT_EQ(gbtree.TargetWorkerBuildsForTest(), 1);

  std::vector<std::string> actions{"load_config", "free"};
#if defined(__APPLE__)
  // LLVM's OpenMP runtime supports parallel regions in a forked child; GNU's may not.
  actions.emplace_back("boost");
#endif  // defined(__APPLE__)
  for (auto const& action : actions) {
    auto pid = fork();
    ASSERT_NE(pid, -1);
    if (pid == 0) {
      alarm(60);  // A child that waits on the parent's threads is killed rather than hanging.
      std::int32_t rc{0};
      try {
        if (action == "load_config") {
          Json config{Object{}};
          gbtree.SaveConfig(&config);
          gbtree.LoadConfig(config);
        } else if (action == "free") {
          p_gbm.reset();
        } else {
          gbtree.DoBoost(p_fmat, &gpair, nullptr);
          rc = gbtree.TargetWorkerBuildsForTest() == 2 ? 0 : 3;
        }
      } catch (...) {
        rc = 2;
      }
      _exit(rc);
    }
    int status{0};
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status)) << action << ": killed by signal " << WTERMSIG(status);
    EXPECT_EQ(WEXITSTATUS(status), 0) << action;
  }
}
#endif  // !defined(_WIN32)
}  // namespace xgboost
