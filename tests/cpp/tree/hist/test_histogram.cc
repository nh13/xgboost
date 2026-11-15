/**
 * Copyright 2018-2026 by Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>                // for bst_node_t, bst_bin_t, Gradient...
#include <xgboost/context.h>             // for Context
#include <xgboost/data.h>                // for BatchIterator, BatchSet, DMatrix
#include <xgboost/host_device_vector.h>  // for HostDeviceVector
#include <xgboost/linalg.h>              // for MakeTensorView
#include <xgboost/logging.h>             // for Error, LogCheck_EQ, LogCheck_LT
#include <xgboost/span.h>                // for Span, operator!=
#include <xgboost/tree_model.h>          // for RegTree

#include <algorithm>  // for max
#include <cmath>      // for ldexp, nextafter
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t, uint32_t
#include <iterator>   // for back_inserter
#include <limits>     // for numeric_limits
#include <memory>     // for shared_ptr, allocator, unique_ptr
#include <numeric>    // for iota, accumulate
#include <vector>     // for vector

#include "../../../../src/collective/communicator-inl.h"  // for GetRank, GetWorldSize
#include "../../../../src/common/hist_util.h"             // for GHistRow, HistogramCuts, Sketch...
#include "../../../../src/common/ref_resource_view.h"     // for RefResourceView
#include "../../../../src/common/row_set.h"               // for RowSetCollection
#include "../../../../src/common/threading_utils.h"       // for BlockedSpace2d
#include "../../../../src/data/gradient_index.h"          // for GHistIndexMatrix
#include "../../../../src/tree/common_row_partitioner.h"  // for CommonRowPartitioner
#include "../../../../src/tree/driver.h"                  // for Driver
#include "../../../../src/tree/hist/expand_entry.h"       // for CPUExpandEntry
#include "../../../../src/tree/hist/hist_cache.h"         // for BoundedHistCollection
#include "../../../../src/tree/hist/hist_param.h"         // for HistMakerTrainParam
#include "../../../../src/tree/hist/histogram.h"          // for HistogramBuilder
#include "../../../../src/tree/tree_view.h"               // for ScalarTreeView
#include "../../categorical_helpers.h"                    // for OneHotEncodeFeature
#include "../../collective/test_worker.h"                 // for TestDistributedGlobal
#include "../../helpers.h"                                // for RandomDataGenerator, GenerateRa...

namespace xgboost::tree {
namespace {
void InitRowPartitionForTest(common::RowSetCollection *row_set, size_t n_samples,
                             size_t base_rowid = 0) {
  auto &row_indices = *row_set->Data();
  row_indices.resize(n_samples);
  std::iota(row_indices.begin(), row_indices.end(), base_rowid);
  row_set->Init();
}
}  // anonymous namespace

void TestAddHistRows(bool is_distributed) {
  Context ctx;
  std::vector<bst_node_t> nodes_to_build;
  std::vector<bst_node_t> nodes_to_sub;

  size_t constexpr kNRows = 8, kNCols = 16;
  int32_t constexpr kMaxBins = 4;
  auto p_fmat = RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
  auto const &gmat =
      *(p_fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{kMaxBins, 0.5}).begin());

  RegTree tree;

  tree.Expand({SplitInfo{0, 0, 0, false}, {0, 0}, {0, 0}, {0, 0}, 0});
  tree.Expand({SplitInfo{tree[0].LeftChild(), 0, 0, false}, {0, 0}, {0, 0}, {0, 0}, 0});
  tree.Expand({SplitInfo{tree[0].RightChild(), 0, 0, false}, {0, 0}, {0, 0}, {0, 0}, 0});
  nodes_to_build.emplace_back(3);
  nodes_to_build.emplace_back(4);
  nodes_to_sub.emplace_back(5);
  nodes_to_sub.emplace_back(6);

  HistMakerTrainParam hist_param;
  HistogramBuilder histogram_builder;
  histogram_builder.Reset(&ctx, gmat.cut.TotalBins(), {kMaxBins, 0.5}, is_distributed, &hist_param);
  histogram_builder.AddHistRows(tree.HostScView(), &nodes_to_build, &nodes_to_sub, false);

  for (bst_node_t const &nidx : nodes_to_build) {
    ASSERT_TRUE(histogram_builder.Histogram().HistogramExists(nidx));
  }
  for (bst_node_t const &nidx : nodes_to_sub) {
    ASSERT_TRUE(histogram_builder.Histogram().HistogramExists(nidx));
  }
}

namespace {
void TestHistogramMinChildWeight(bool limit, bool is_distributed) {
  Context ctx;
  ctx.nthread = 2;
  auto dmat = GetDMatrixFromData({0.0f, 1.0f}, 2, 1);
  BatchParam batch{2, 0.5};
  auto const &gmat = *dmat->GetBatches<GHistIndexMatrix>(&ctx, batch).begin();
  TrainParam param;
  param.Init(Args{{"min_child_weight", "1"}});
  HistMakerTrainParam hist_param;
  if (limit) {
    hist_param.Init(Args{{"max_cached_hist_node", "1"}});
  }
  auto world = static_cast<float>(collective::GetWorldSize());
  for (float left_hess : {1.5f, std::nextafter(2.0f, 0.0f), 2.0f}) {
    for (float right_hess : {1.5f, std::nextafter(2.0f, 0.0f), 2.0f}) {
      SCOPED_TRACE(left_hess);
      SCOPED_TRACE(right_hess);
      RegTree tree;
      std::vector<CommonRowPartitioner> partitioners;
      partitioners.emplace_back(&ctx, 2, 0);
      std::vector<GradientPair> gpair{{-1.0f / world, left_hess / world},
                                      {1.0f / world, right_hess / world}};
      auto view = linalg::MakeTensorView(&ctx, common::Span<GradientPair const>{gpair}, 2, 1);
      MultiHistogramBuilder builder;
      builder.Reset(&ctx, gmat.cut.TotalBins(), 1, batch, is_distributed, &hist_param);
      CPUExpandEntry best{0, 0};
      builder.BuildRootHist(dmat.get(), tree.HostScView(), partitioners, view, best, batch);

      auto split_cond = gmat.cut.Values().front();
      GradientPairPrecise left{-1.0, left_hess}, right{1.0, right_hess};
      best.split.Update(1.0f, 0, split_cond, false, false, linalg::MakeVec(&left, 1),
                        linalg::MakeVec(&right, 1));
      tree.Expand({{0, 0, split_cond, false},
                   {0.0, left_hess + right_hess},
                   {0.0, left_hess},
                   {0.0, right_hess},
                   1.0});
      std::vector<CPUExpandEntry> candidates{best};
      partitioners.front().UpdatePosition(&ctx, gmat, candidates, tree.HostScView());
      Driver<CPUExpandEntry> driver{param};
      if (driver.IsChildValid(best, left_hess, right_hess)) {
        builder.BuildHistLeftRight(&ctx, dmat.get(), tree.HostScView(), partitioners, candidates,
                                   view, batch);
      }
      auto const &hist = builder.Histogram(0);
      bool can_split = left_hess >= 2.0f || right_hess >= 2.0f;
      EXPECT_EQ(hist.HistogramExists(1), can_split);
      EXPECT_EQ(hist.HistogramExists(2), can_split);
      if (!can_split) {
        // Skipping the entire batch must not allocate histograms or evict the parent.
        EXPECT_TRUE(hist.HistogramExists(0));
        continue;
      }
      EXPECT_EQ(hist.HistogramExists(0), !limit);
      // Even an unsplittable sibling is retained for subtraction if the other child
      // can split. With a full cache, both histograms are instead built directly.
      EXPECT_EQ(hist[1][0], GradientPairPrecise(-1.0, left_hess));
      EXPECT_EQ(hist[1][1], GradientPairPrecise());
      EXPECT_EQ(hist[2][0], GradientPairPrecise());
      EXPECT_EQ(hist[2][1], GradientPairPrecise(1.0, right_hess));
    }
  }
}
}  // namespace

TEST(CPUHistogram, MinChildWeight) {
  TestHistogramMinChildWeight(false, false);
  TestHistogramMinChildWeight(true, false);
}

TEST(CPUHistogram, DistributedMinChildWeight) {
  collective::TestDistributedGlobal(2, [] {
    TestHistogramMinChildWeight(false, true);
    TestHistogramMinChildWeight(true, true);
  });
}

namespace {
// Trees in which the root and its left child, node 1, are both split, with scalar leaves and with
// vector leaves.
std::unique_ptr<RegTree> MakeTwoSplitTree() {
  auto p_tree = std::make_unique<RegTree>();
  p_tree->Expand({{0, 0, 0.5f, false}, {0.0, 10.0}, {0.0, 1.0}, {0.0, 9.0}, 1.0});
  p_tree->Expand({{1, 0, 0.25f, false}, {0.0, 1.0}, {0.0, 0.9}, {0.0, 0.1}, 1.0});
  return p_tree;
}

std::unique_ptr<RegTree> MakeTwoSplitTree(bst_target_t n_targets) {
  Context ctx;
  auto p_tree = std::make_unique<RegTree>(n_targets, 2);
  std::vector<float> weight(n_targets, 1.0f);
  common::Span<float const> w{weight};
  p_tree->SetRoot(linalg::MakeVec(weight.data(), n_targets), 10.0f);
  p_tree->Expand(&ctx, {{{0, 0, 0.5f, false}, {w, 10.0}, {w, 1.0}, {w, 9.0}, 1.0f}});
  p_tree->Expand(&ctx, {{{1, 0, 0.25f, false}, {w, 1.0}, {w, 0.9}, {w, 0.1}, 1.0f}});
  return p_tree;
}

template <typename TreeView>
void TestAssignNodes(TreeView const &view, bst_target_t n_targets) {
  // Row counts and sums of Hessian disagree in every check below, so each checks one rule.
  auto make_candidate = [&](bst_node_t nidx, double left_hess, double right_hess) {
    std::vector<GradientPairPrecise> left(n_targets, {0.0, left_hess});
    std::vector<GradientPairPrecise> right(n_targets, {0.0, right_hess});
    CPUExpandEntry c{nidx, nidx == 0 ? 0 : 1};
    c.split.Update(1.0f, 0, 0.5f, false, false, linalg::MakeVec(left.data(), n_targets),
                   linalg::MakeVec(right.data(), n_targets));
    return c;
  };
  // The root's left child has less Hessian, node 1's right child does.
  auto root = make_candidate(0, 1.0, 9.0), node1 = make_candidate(1, 0.9, 0.1);
  auto check = [&](std::vector<CPUExpandEntry> const &candidates,
                   std::vector<ChildRows> const &n_rows, std::vector<bst_node_t> const &expected) {
    std::vector<bst_node_t> build(candidates.size()), sub(candidates.size());
    AssignNodes(view, candidates, n_rows, build, sub);
    ASSERT_EQ(build, expected);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      auto nidx = candidates[i].nid;
      ASSERT_EQ(sub[i],
                build[i] == view.LeftChild(nidx) ? view.RightChild(nidx) : view.LeftChild(nidx));
    }
  };
  auto left1 = view.LeftChild(1), right1 = view.RightChild(1);
  check({root}, {{10, 3}}, {2});      // The right child has fewer rows.
  check({root}, {}, {1});             // The left child has less Hessian.
  check({node1}, {{3, 3}}, {left1});  // Ties build the left child.
  // Each candidate uses its own counts.
  check({root, node1}, {{10, 3}, {2, 8}}, {2, left1});
  check({root, node1}, {{3, 10}, {8, 2}}, {1, right1});
  check({root, node1}, {}, {1, right1});
  // The counts must cover every candidate.
  std::vector<bst_node_t> build(1), sub(1);
  EXPECT_THROW(AssignNodes(view, {root}, std::vector<ChildRows>{{10, 3}, {2, 8}}, build, sub),
               dmlc::Error);
}
}  // namespace

TEST(CPUHistogram, AssignNodes) {
  auto p_tree = MakeTwoSplitTree();
  TestAssignNodes(p_tree->HostScView(), 1);
  bst_target_t constexpr kTargets = 3;
  auto p_mt_tree = MakeTwoSplitTree(kTargets);
  TestAssignNodes(p_mt_tree->HostMtView(), kTargets);
}

TEST(CPUHistogram, CountChildRows) {
  // Two pages: in the first alone the root's right child has fewer rows, in the sum the left one.
  struct Partition {
    std::size_t n;
    [[nodiscard]] std::size_t Size() const { return n; }
  };
  struct Partitioner {
    std::vector<Partition> partitions;
    [[nodiscard]] auto const &Partitions() const { return partitions; }
  };
  std::vector<Partitioner> partitioners{{{{8}, {7}, {1}, {4}, {3}}}, {{{9}, {0}, {9}, {0}, {0}}}};
  std::vector<CPUExpandEntry> candidates{{0, 0}, {1, 1}};
  std::vector<ChildRows> expected{{7, 10}, {4, 3}};
  auto p_tree = MakeTwoSplitTree();
  ASSERT_EQ(cpu_impl::CountChildRows(p_tree->HostScView(), partitioners, candidates), expected);
  auto p_mt_tree = MakeTwoSplitTree(3);
  ASSERT_EQ(cpu_impl::CountChildRows(p_mt_tree->HostMtView(), partitioners, candidates), expected);
}

namespace {
// Three rows split two to one, and a constant second feature puts all of them in one bin. The two
// rows on one side have a Hessian of 1 each, the lone row on the other side 2^60, so in the shared
// bin the parent's sum 2 + 2^60 rounds to 2^60. Building the lone row's child, which has fewer
// rows, makes the other child's bin 2^60 - 2^60 = 0, the rounding error that subtraction inherits;
// building the child with less Hessian gives exactly 2. The distributed run gives every worker
// all three rows, with gradients divided by the number of workers, and keeps the Hessian rule.
void TestBuildChild(bool is_distributed, bool fewer_left) {
  Context ctx;
  float const lone = fewer_left ? 0.0f : 1.0f, pair = fewer_left ? 1.0f : 0.0f;
  auto dmat = GetDMatrixFromData(
      {fewer_left ? lone : pair, 0.0f, pair, 0.0f, fewer_left ? pair : lone, 0.0f}, 3, 2);
  BatchParam batch{2, 0.5};
  auto const &gmat = *dmat->GetBatches<GHistIndexMatrix>(&ctx, batch).begin();
  HistMakerTrainParam hist_param;
  float const big = std::ldexp(1.0f, 60);
  auto world = static_cast<float>(collective::GetWorldSize());
  // The lone row is row 0 when it goes left, row 2 otherwise.
  std::vector<GradientPair> gpair(3, GradientPair{-1.0f / world, 1.0f / world});
  gpair[fewer_left ? 0 : 2] = GradientPair{1.0f / world, big / world};
  auto view = linalg::MakeTensorView(&ctx, common::Span<GradientPair const>{gpair}, 3, 1);

  RegTree tree;
  std::vector<CommonRowPartitioner> partitioners;
  partitioners.emplace_back(&ctx, 3, 0);
  MultiHistogramBuilder builder;
  builder.Reset(&ctx, gmat.cut.TotalBins(), 1, batch, is_distributed, &hist_param);
  CPUExpandEntry best{0, 0};
  builder.BuildRootHist(dmat.get(), tree.HostScView(), partitioners, view, best, batch);

  auto split_cond = gmat.cut.Values().front();
  GradientPairPrecise few{1.0, big}, many{-2.0, 2.0};
  auto const &left = fewer_left ? few : many;
  auto const &right = fewer_left ? many : few;
  best.split.Update(1.0f, 0, split_cond, false, false, linalg::MakeVec(&left, 1),
                    linalg::MakeVec(&right, 1));
  tree.Expand({{0, 0, split_cond, false},
               {0.0, 2.0 + big},
               {0.0, left.GetHess()},
               {0.0, right.GetHess()},
               1.0});
  std::vector<CPUExpandEntry> candidates{best};
  partitioners.front().UpdatePosition(&ctx, gmat, candidates, tree.HostScView());
  bst_node_t const few_nidx = fewer_left ? 1 : 2, many_nidx = fewer_left ? 2 : 1;
  ASSERT_EQ(partitioners.front().Partitions()[few_nidx].Size(), 1);
  ASSERT_EQ(partitioners.front().Partitions()[many_nidx].Size(), 2);
  builder.BuildHistLeftRight(&ctx, dmat.get(), tree.HostScView(), partitioners, candidates, view,
                             batch);

  auto const &hist = builder.Histogram(0);
  auto shared_bin = gmat.cut.Ptrs()[1];  // The only bin of the second feature.
  ASSERT_EQ(gmat.cut.Ptrs()[2] - shared_bin, 1);
  ASSERT_EQ(hist[0][shared_bin].GetHess(), static_cast<double>(big));
  ASSERT_EQ(hist[few_nidx][shared_bin].GetHess(), static_cast<double>(big));
  // 0 is the rounding error, not the true sum of 2: it shows that the child with fewer rows was
  // built and the other one subtracted.
  ASSERT_EQ(hist[many_nidx][shared_bin].GetHess(), is_distributed ? 2.0 : 0.0);
}
}  // namespace

TEST(CPUHistogram, BuildChildWithFewerRows) {
  TestBuildChild(false, false);
  TestBuildChild(false, true);
}

TEST(CPUHistogram, DistributedBuildChildWithLessHessian) {
  collective::TestDistributedGlobal(2, [] {
    TestBuildChild(true, false);
    TestBuildChild(true, true);
  });
}

TEST(CPUHistogram, CacheGrowth) {
  BoundedHistCollection cache;
  bst_bin_t constexpr kBins = 3;
  std::size_t constexpr kLimit = 7;
  cache.Reset(kBins, kLimit);
  std::size_t allocations = 0;
  for (bst_node_t n = 0; n < static_cast<bst_node_t>(kLimit); ++n) {
    auto previous_capacity = cache.Capacity();
    cache.AllocateHistograms(std::vector<bst_node_t>{n});
    allocations += cache.Capacity() != previous_capacity;
    ASSERT_EQ(cache.Size(), (n + 1) * kBins);
    ASSERT_GE(cache.Capacity(), cache.Size());
    ASSERT_LE(cache.Capacity(), kLimit * kBins);
    for (auto &bin : cache[n]) {
      bin = GradientPairPrecise{static_cast<double>(n), static_cast<double>(n + 1)};
    }
    // Growth must preserve every previously allocated histogram.
    for (bst_node_t old = 0; old <= n; ++old) {
      for (auto const &bin : cache[old]) {
        ASSERT_EQ(bin.GetGrad(), old);
        ASSERT_EQ(bin.GetHess(), old + 1);
      }
    }
  }
  ASSERT_EQ(allocations, 6);  // Bin capacities: 3, 6, 9, 13, 19, then the limit of 21.
  std::vector<bst_node_t> next{7};
  ASSERT_FALSE(cache.CanHost(next, {}));
  auto capacity = cache.Capacity();
  cache.Clear(true);
  ASSERT_TRUE(cache.HasExceeded());
  ASSERT_FALSE(cache.HistogramExists(0));
  cache.AllocateHistograms(next);
  ASSERT_EQ(cache.Size(), kBins);
  ASSERT_EQ(cache.Capacity(), capacity);

  // A batch may exceed the cache limit, but must not reserve additional spare capacity.
  cache.Reset(kBins, 2);
  std::vector<bst_node_t> batch(9);
  std::iota(batch.begin(), batch.end(), 0);
  ASSERT_FALSE(cache.CanHost(batch, {}));
  cache.AllocateHistograms(batch);
  ASSERT_EQ(cache.Size(), batch.size() * kBins);
  ASSERT_EQ(cache.Capacity(), cache.Size());
  ASSERT_FALSE(cache.HasExceeded());
  ASSERT_EQ(cache[8].size(), kBins);
}

TEST(CPUHistogram, AddRows) {
  TestAddHistRows(true);
  TestAddHistRows(false);
}

void TestSyncHist(bool is_distributed) {
  std::size_t constexpr kNRows = 8, kNCols = 16;
  bst_bin_t constexpr kMaxBins = 4;
  Context ctx;

  std::vector<bst_bin_t> nodes_for_explicit_hist_build;
  std::vector<bst_bin_t> nodes_for_subtraction_trick;
  RegTree tree;

  auto p_fmat = RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
  auto const &gmat =
      *(p_fmat->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{kMaxBins, 0.5}).begin());

  HistogramBuilder histogram;
  uint32_t total_bins = gmat.cut.Ptrs().back();
  HistMakerTrainParam hist_param;
  histogram.Reset(&ctx, total_bins, {kMaxBins, 0.5}, is_distributed, &hist_param);

  common::RowSetCollection row_set_collection;
  {
    row_set_collection.Clear();
    std::vector<bst_idx_t> &row_indices = *row_set_collection.Data();
    row_indices.resize(kNRows);
    std::iota(row_indices.begin(), row_indices.end(), 0);
    row_set_collection.Init();
  }

  // level 0
  nodes_for_explicit_hist_build.emplace_back(0);
  histogram.AddHistRows(tree.HostScView(), &nodes_for_explicit_hist_build,
                        &nodes_for_subtraction_trick, false);

  tree.Expand({SplitInfo{0, 0, 0, false}, {0, 0}, {0, 0}, {0, 0}, 0});
  nodes_for_explicit_hist_build.clear();
  nodes_for_subtraction_trick.clear();

  // level 1
  nodes_for_explicit_hist_build.emplace_back(tree[0].LeftChild());
  nodes_for_subtraction_trick.emplace_back(tree[0].RightChild());

  histogram.AddHistRows(tree.HostScView(), &nodes_for_explicit_hist_build,
                        &nodes_for_subtraction_trick, false);

  tree.Expand({SplitInfo{tree[0].LeftChild(), 0, 0, false}, {0, 0}, {0, 0}, {0, 0}, 0});
  tree.Expand({SplitInfo{tree[0].RightChild(), 0, 0, false}, {0, 0}, {0, 0}, {0, 0}, 0});

  nodes_for_explicit_hist_build.clear();
  nodes_for_subtraction_trick.clear();
  // level 2
  nodes_for_explicit_hist_build.emplace_back(3);
  nodes_for_subtraction_trick.emplace_back(4);
  nodes_for_explicit_hist_build.emplace_back(5);
  nodes_for_subtraction_trick.emplace_back(6);

  histogram.AddHistRows(tree.HostScView(), &nodes_for_explicit_hist_build,
                        &nodes_for_subtraction_trick, false);

  const size_t n_nodes = nodes_for_explicit_hist_build.size();
  ASSERT_EQ(n_nodes, 2ul);
  row_set_collection.AddSplit(0, tree[0].LeftChild(), tree[0].RightChild(), 4, 4);
  row_set_collection.AddSplit(1, tree[1].LeftChild(), tree[1].RightChild(), 2, 2);
  row_set_collection.AddSplit(2, tree[2].LeftChild(), tree[2].RightChild(), 2, 2);

  common::BlockedSpace2d space(
      n_nodes,
      [&](std::size_t nidx_in_set) {
        bst_node_t nidx = nodes_for_explicit_hist_build[nidx_in_set];
        return row_set_collection[nidx].Size();
      },
      256);

  std::vector<common::GHistRow> target_hists(n_nodes);
  for (size_t i = 0; i < nodes_for_explicit_hist_build.size(); ++i) {
    bst_node_t nidx = nodes_for_explicit_hist_build[i];
    target_hists[i] = histogram.Histogram()[nidx];
  }

  // set values to specific nodes hist
  std::vector<size_t> n_ids = {1, 2};
  for (size_t i : n_ids) {
    auto this_hist = histogram.Histogram()[i];
    double *p_hist = reinterpret_cast<double *>(this_hist.data());
    for (size_t bin_id = 0; bin_id < 2 * total_bins; ++bin_id) {
      p_hist[bin_id] = 2 * bin_id;
    }
  }
  n_ids[0] = 3;
  n_ids[1] = 5;
  for (size_t i : n_ids) {
    auto this_hist = histogram.Histogram()[i];
    double *p_hist = reinterpret_cast<double *>(this_hist.data());
    for (size_t bin_id = 0; bin_id < 2 * total_bins; ++bin_id) {
      p_hist[bin_id] = bin_id;
    }
  }

  histogram.Buffer().Reset(1, n_nodes, space, target_hists);
  // sync hist
  histogram.SyncHistogram(&ctx, tree.HostScView(), nodes_for_explicit_hist_build,
                          nodes_for_subtraction_trick);

  using GHistRowT = common::GHistRow;
  auto check_hist = [](const GHistRowT parent, const GHistRowT left, const GHistRowT right,
                       size_t begin, size_t end) {
    const double *p_parent = reinterpret_cast<const double *>(parent.data());
    const double *p_left = reinterpret_cast<const double *>(left.data());
    const double *p_right = reinterpret_cast<const double *>(right.data());
    for (size_t i = 2 * begin; i < 2 * end; ++i) {
      ASSERT_EQ(p_parent[i], p_left[i] + p_right[i]);
    }
  };
  size_t node_id = 0;
  for (auto const &nidx : nodes_for_explicit_hist_build) {
    auto this_hist = histogram.Histogram()[nidx];
    const size_t parent_id = tree[nidx].Parent();
    const size_t subtraction_node_id = nodes_for_subtraction_trick[node_id];
    auto parent_hist = histogram.Histogram()[parent_id];
    auto sibling_hist = histogram.Histogram()[subtraction_node_id];

    check_hist(parent_hist, this_hist, sibling_hist, 0, total_bins);
    ++node_id;
  }
  node_id = 0;
  for (auto const &nidx : nodes_for_subtraction_trick) {
    auto this_hist = histogram.Histogram()[nidx];
    const size_t parent_id = tree[nidx].Parent();
    const size_t subtraction_node_id = nodes_for_explicit_hist_build[node_id];
    auto parent_hist = histogram.Histogram()[parent_id];
    auto sibling_hist = histogram.Histogram()[subtraction_node_id];

    check_hist(parent_hist, this_hist, sibling_hist, 0, total_bins);
    ++node_id;
  }
}

TEST(CPUHistogram, SyncHist) {
  TestSyncHist(true);
  TestSyncHist(false);
}

void TestBuildHistogram(Context const *ctx, bool is_distributed, bool force_read_by_column) {
  size_t constexpr kNRows = 8, kNCols = 16;
  int32_t constexpr kMaxBins = 4;
  auto p_fmat = RandomDataGenerator(kNRows, kNCols, 0.8).Seed(3).GenerateDMatrix();
  auto const &gmat =
      *(p_fmat->GetBatches<GHistIndexMatrix>(ctx, BatchParam{kMaxBins, 0.5}).begin());
  uint32_t total_bins = gmat.cut.Ptrs().back();

  static double constexpr kEps = 1e-6;
  std::vector<GradientPair> gpair = {{0.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f},
                                     {0.27f, 0.28f}, {0.27f, 0.29f}, {0.37f, 0.39f},
                                     {0.47f, 0.49f}, {0.57f, 0.59f}};

  bst_node_t nid = 0;
  HistogramBuilder histogram;
  HistMakerTrainParam hist_param;
  histogram.Reset(ctx, total_bins, {kMaxBins, 0.5}, is_distributed, &hist_param);

  RegTree tree;

  common::RowSetCollection row_set_collection;
  row_set_collection.Clear();
  std::vector<bst_idx_t> &row_indices = *row_set_collection.Data();
  row_indices.resize(kNRows);
  std::iota(row_indices.begin(), row_indices.end(), 0);
  row_set_collection.Init();

  CPUExpandEntry node{RegTree::kRoot, tree.GetDepth(0)};
  std::vector<bst_node_t> nodes_to_build{node.nid};
  std::vector<bst_node_t> dummy_sub;

  histogram.AddHistRows(tree.HostScView(), &nodes_to_build, &dummy_sub, false);
  common::BlockedSpace2d space{
      1, [&](std::size_t nidx_in_set) { return row_set_collection[nidx_in_set].Size(); }, 256};
  for (auto const &gidx : p_fmat->GetBatches<GHistIndexMatrix>(ctx, {kMaxBins, 0.5})) {
    histogram.BuildHist(0, space, gidx, row_set_collection, nodes_to_build,
                        linalg::MakeTensorView(ctx, gpair, gpair.size()), force_read_by_column);
  }
  histogram.SyncHistogram(ctx, tree.HostScView(), nodes_to_build, {});

  // Check if number of histogram bins is correct
  ASSERT_EQ(histogram.Histogram()[nid].size(), gmat.cut.Ptrs().back());
  std::vector<GradientPairPrecise> histogram_expected(histogram.Histogram()[nid].size());

  // Compute the correct histogram (histogram_expected)
  CHECK_EQ(gpair.size(), kNRows);
  for (size_t rid = 0; rid < kNRows; ++rid) {
    const size_t ibegin = gmat.row_ptr[rid];
    const size_t iend = gmat.row_ptr[rid + 1];
    for (size_t i = ibegin; i < iend; ++i) {
      const size_t bin_id = gmat.index[i];
      histogram_expected[bin_id] += GradientPairPrecise(gpair[rid]);
    }
  }

  // Now validate the computed histogram returned by BuildHist
  for (size_t i = 0; i < histogram.Histogram()[nid].size(); ++i) {
    GradientPairPrecise sol = histogram_expected[i];
    ASSERT_NEAR(sol.GetGrad(), histogram.Histogram()[nid][i].GetGrad(), kEps);
    ASSERT_NEAR(sol.GetHess(), histogram.Histogram()[nid][i].GetHess(), kEps);
  }
}

TEST(CPUHistogram, BuildHist) {
  Context ctx;
  TestBuildHistogram(&ctx, true, false);
  TestBuildHistogram(&ctx, false, false);
  TestBuildHistogram(&ctx, true, true);
  TestBuildHistogram(&ctx, false, true);
}

namespace {
template <typename GradientSumT>
void ValidateCategoricalHistogram(size_t n_categories, common::Span<GradientSumT> onehot,
                                  common::Span<GradientSumT> cat) {
  auto cat_sum = std::accumulate(cat.cbegin(), cat.cend(), GradientPairPrecise{});
  for (size_t c = 0; c < n_categories; ++c) {
    auto zero = onehot[c * 2];
    auto one = onehot[c * 2 + 1];

    auto chosen = cat[c];
    auto not_chosen = cat_sum - chosen;

    ASSERT_LE(RelError(zero.GetGrad(), not_chosen.GetGrad()), kRtEps);
    ASSERT_LE(RelError(zero.GetHess(), not_chosen.GetHess()), kRtEps);

    ASSERT_LE(RelError(one.GetGrad(), chosen.GetGrad()), kRtEps);
    ASSERT_LE(RelError(one.GetHess(), chosen.GetHess()), kRtEps);
  }
}

void TestHistogramCategorical(size_t n_categories, bool force_read_by_column) {
  size_t constexpr kRows = 340;
  bst_bin_t constexpr kBins = 256;
  auto x = GenerateRandomCategoricalSingleColumn(kRows, n_categories);
  auto cat_m = GetDMatrixFromData(x, kRows, 1);
  cat_m->Info().feature_types.HostVector().push_back(FeatureType::kCategorical);
  Context ctx;

  BatchParam batch_param{0, kBins};

  RegTree tree;
  CPUExpandEntry node{RegTree::kRoot, tree.GetDepth(RegTree::kRoot)};
  std::vector<bst_node_t> nodes_to_build;
  nodes_to_build.push_back(node.nid);

  auto gpair = GenerateRandomGradients(kRows, 0, 2);

  common::RowSetCollection row_set_collection;
  row_set_collection.Clear();
  std::vector<bst_idx_t> &row_indices = *row_set_collection.Data();
  row_indices.resize(kRows);
  std::iota(row_indices.begin(), row_indices.end(), 0);
  row_set_collection.Init();
  HistMakerTrainParam hist_param;
  std::vector<bst_node_t> dummy_sub;

  common::BlockedSpace2d space{
      1, [&](std::size_t nidx_in_set) { return row_set_collection[nidx_in_set].Size(); }, 256};

  /**
   * Generate hist with cat data.
   */
  HistogramBuilder cat_hist;
  for (auto const &gidx : cat_m->GetBatches<GHistIndexMatrix>(&ctx, {kBins, 0.5})) {
    auto total_bins = gidx.cut.TotalBins();
    cat_hist.Reset(&ctx, total_bins, {kBins, 0.5}, false, &hist_param);
    cat_hist.AddHistRows(tree.HostScView(), &nodes_to_build, &dummy_sub, false);
    cat_hist.BuildHist(0, space, gidx, row_set_collection, nodes_to_build,
                       linalg::MakeTensorView(&ctx, gpair.ConstHostSpan(), gpair.Size()),
                       force_read_by_column);
  }
  cat_hist.SyncHistogram(&ctx, tree.HostScView(), nodes_to_build, {});

  /**
   * Generate hist with one hot encoded data.
   */
  auto x_encoded = OneHotEncodeFeature(x, n_categories);
  auto encode_m = GetDMatrixFromData(x_encoded, kRows, n_categories);
  HistogramBuilder onehot_hist;
  for (auto const &gidx : encode_m->GetBatches<GHistIndexMatrix>(&ctx, {kBins, 0.5})) {
    auto total_bins = gidx.cut.TotalBins();
    onehot_hist.Reset(&ctx, total_bins, {kBins, 0.5}, false, &hist_param);
    onehot_hist.AddHistRows(tree.HostScView(), &nodes_to_build, &dummy_sub, false);
    onehot_hist.BuildHist(0, space, gidx, row_set_collection, nodes_to_build,
                          linalg::MakeTensorView(&ctx, gpair.ConstHostSpan(), gpair.Size()),
                          force_read_by_column);
  }
  onehot_hist.SyncHistogram(&ctx, tree.HostScView(), nodes_to_build, {});

  auto cat = cat_hist.Histogram()[0];
  auto onehot = onehot_hist.Histogram()[0];
  ValidateCategoricalHistogram(n_categories, onehot, cat);
}
}  // anonymous namespace

TEST(CPUHistogram, Categorical) {
  for (size_t n_categories = 2; n_categories < 8; ++n_categories) {
    TestHistogramCategorical(n_categories, false);
  }
  for (size_t n_categories = 2; n_categories < 8; ++n_categories) {
    TestHistogramCategorical(n_categories, true);
  }
}
namespace {
void TestHistogramExternalMemory(Context const *ctx, BatchParam batch_param, bool is_approx,
                                 bool force_read_by_column) {
  size_t constexpr kEntries = 1 << 16;
  auto m =
      RandomDataGenerator{kEntries / 8, 8, 0.0f}.Batches(4).GenerateSparsePageDMatrix("temp", true);

  std::vector<float> hess(m->Info().num_row_, 1.0);
  if (is_approx) {
    batch_param.hess = hess;
  }

  std::vector<bst_idx_t> partition_size(1, 0);
  bst_bin_t total_bins{0};
  bst_idx_t n_samples{0};

  auto gpair = GenerateRandomGradients(m->Info().num_row_, 0.0, 1.0);
  auto const &h_gpair = gpair.HostVector();

  RegTree tree;
  std::vector<bst_node_t> nodes{RegTree::kRoot};
  common::BlockedSpace2d space{
      1, [&](std::size_t nidx_in_set) { return partition_size.at(nidx_in_set); }, 256};

  common::GHistRow multi_page;
  HistogramBuilder multi_build;
  HistMakerTrainParam hist_param;
  std::vector<bst_node_t> dummy_sub;
  {
    /**
     * Multi page
     */
    std::vector<common::RowSetCollection> rows_set;
    for (auto const &page : m->GetBatches<GHistIndexMatrix>(ctx, batch_param)) {
      CHECK_LT(page.base_rowid, m->Info().num_row_);
      auto n_rows_in_node = page.Size();
      partition_size[0] = std::max(partition_size[0], n_rows_in_node);
      total_bins = page.cut.TotalBins();
      n_samples += n_rows_in_node;

      rows_set.emplace_back();
      InitRowPartitionForTest(&rows_set.back(), n_rows_in_node, page.base_rowid);
    }
    ASSERT_EQ(n_samples, m->Info().num_row_);

    multi_build.Reset(ctx, total_bins, batch_param, false, &hist_param);
    multi_build.AddHistRows(tree.HostScView(), &nodes, &dummy_sub, false);
    std::size_t page_idx{0};
    for (auto const &page : m->GetBatches<GHistIndexMatrix>(ctx, batch_param)) {
      multi_build.BuildHist(page_idx, space, page, rows_set[page_idx], nodes,
                            linalg::MakeTensorView(ctx, h_gpair, h_gpair.size()),
                            force_read_by_column);
      ++page_idx;
    }
    multi_build.SyncHistogram(ctx, tree.HostScView(), nodes, {});

    multi_page = multi_build.Histogram()[RegTree::kRoot];
  }

  HistogramBuilder single_build;
  common::GHistRow single_page;
  {
    /**
     * Single page
     */
    common::RowSetCollection row_set_collection;
    InitRowPartitionForTest(&row_set_collection, n_samples);

    single_build.Reset(ctx, total_bins, batch_param, false, &hist_param);
    SparsePage concat;
    std::vector<float> hess(m->Info().num_row_, 1.0f);
    for (auto const &page : m->GetBatches<SparsePage>()) {
      concat.Push(page);
    }

    auto cut = common::SketchOnDMatrix(ctx, m.get(), batch_param.max_bin, false, hess);
    GHistIndexMatrix gmat(ctx, concat, {}, cut, batch_param.max_bin, false,
                          std::numeric_limits<double>::quiet_NaN());

    single_build.AddHistRows(tree.HostScView(), &nodes, &dummy_sub, false);
    single_build.BuildHist(0, space, gmat, row_set_collection, nodes,
                           linalg::MakeTensorView(ctx, h_gpair, h_gpair.size()),
                           force_read_by_column);
    single_build.SyncHistogram(ctx, tree.HostScView(), nodes, {});

    single_page = single_build.Histogram()[RegTree::kRoot];
  }

  for (size_t i = 0; i < single_page.size(); ++i) {
    ASSERT_NEAR(single_page[i].GetGrad(), multi_page[i].GetGrad(), kRtEps);
    ASSERT_NEAR(single_page[i].GetHess(), multi_page[i].GetHess(), kRtEps);
  }
}
}  // anonymous namespace

TEST(CPUHistogram, ExternalMemory) {
  int32_t constexpr kBins = 256;
  Context ctx;

  TestHistogramExternalMemory(&ctx, BatchParam{kBins, common::Span<float>{}, false}, true, false);
  TestHistogramExternalMemory(&ctx, BatchParam{kBins, common::Span<float>{}, false}, true, true);

  float sparse_thresh{0.5};
  TestHistogramExternalMemory(&ctx, {kBins, sparse_thresh}, false, false);
  TestHistogramExternalMemory(&ctx, {kBins, sparse_thresh}, false, true);
  sparse_thresh = std::numeric_limits<float>::quiet_NaN();
  TestHistogramExternalMemory(&ctx, {kBins, sparse_thresh}, false, false);
  TestHistogramExternalMemory(&ctx, {kBins, sparse_thresh}, false, true);
}

namespace {
class OverflowTest : public ::testing::TestWithParam<bool> {
 public:
  std::vector<GradientPairPrecise> TestOverflow(bool limit, bool is_distributed) {
    bst_bin_t constexpr kBins = 256;
    Context ctx;
    HistMakerTrainParam hist_param;
    if (limit) {
      hist_param.Init(Args{{"max_cached_hist_node", "1"}});
    }

    auto Xy = RandomDataGenerator{8192, 16, 0.5}.Bins(kBins).GenerateQuantileDMatrix(true);

    double sparse_thresh{TrainParam::DftSparseThreshold()};
    auto batch = BatchParam{kBins, sparse_thresh};
    bst_bin_t n_total_bins{0};
    float split_cond{0};
    for (auto const &page : Xy->GetBatches<GHistIndexMatrix>(&ctx, batch)) {
      n_total_bins = page.cut.TotalBins();
      // use a cut point in the second column for split
      split_cond = page.cut.Values()[kBins + kBins / 2];
    }

    RegTree tree;
    MultiHistogramBuilder hist_builder;

    hist_builder.Reset(&ctx, n_total_bins, tree.NumTargets(), batch, is_distributed, &hist_param);

    std::vector<CommonRowPartitioner> partitioners;
    partitioners.emplace_back(&ctx, Xy->Info().num_row_, /*base_rowid=*/0);

    auto gpair = GenerateRandomGradients(Xy->Info().num_row_, 0.0, 1.0);

    CPUExpandEntry best;
    hist_builder.BuildRootHist(Xy.get(), tree.HostScView(), partitioners,
                               linalg::MakeTensorView(&ctx, gpair.ConstHostSpan(), gpair.Size(), 1),
                               best, batch);

    GradientPairPrecise sum{1.0, 1.0};
    best.split.Update(1.0f, 1, split_cond, false, false, linalg::MakeVec(&sum, 1),
                      linalg::MakeVec(&sum, 1));
    tree.Expand({{best.nid, best.split.SplitIndex(), best.split.split_value, false},
                 {2.0f, 2.0f},
                 {1.0f, best.split.left_sum[0].GetHess()},
                 {1.0f, best.split.right_sum[0].GetHess()},
                 best.GetLossChange()});

    std::vector<CPUExpandEntry> valid_candidates{best};
    for (auto const &page : Xy->GetBatches<GHistIndexMatrix>(&ctx, batch)) {
      partitioners.front().UpdatePosition(&ctx, page, valid_candidates, tree.HostScView());
    }
    CHECK_NE(partitioners.front()[tree.LeftChild(best.nid)].Size(), 0);
    CHECK_NE(partitioners.front()[tree.RightChild(best.nid)].Size(), 0);

    hist_builder.BuildHistLeftRight(
        &ctx, Xy.get(), tree.HostScView(), partitioners, valid_candidates,
        linalg::MakeTensorView(&ctx, gpair.ConstHostSpan(), gpair.Size(), 1), batch);

    if (limit) {
      CHECK(!hist_builder.Histogram(0).HistogramExists(best.nid));
    } else {
      CHECK(hist_builder.Histogram(0).HistogramExists(best.nid));
    }

    std::vector<GradientPairPrecise> result;
    auto hist = hist_builder.Histogram(0)[tree.LeftChild(best.nid)];
    std::copy(hist.cbegin(), hist.cend(), std::back_inserter(result));
    hist = hist_builder.Histogram(0)[tree.RightChild(best.nid)];
    std::copy(hist.cbegin(), hist.cend(), std::back_inserter(result));

    return result;
  }

  void RunTest() {
    auto param = GetParam();
    auto res0 = this->TestOverflow(false, param);
    auto res1 = this->TestOverflow(true, param);
    ASSERT_EQ(res0, res1);
  }
};

auto MakeParamsForTest() { return std::vector<bool>{true, false}; }
}  // anonymous namespace

TEST_P(OverflowTest, Overflow) { this->RunTest(); }

INSTANTIATE_TEST_SUITE_P(CPUHistogram, OverflowTest, ::testing::ValuesIn(MakeParamsForTest()));
}  // namespace xgboost::tree
