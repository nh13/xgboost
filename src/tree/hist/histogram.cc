/**
 * Copyright 2023-2026, XGBoost Contributors
 */
#include "histogram.h"

#include <cstddef>  // for size_t
#include <numeric>  // for accumulate
#include <utility>  // for swap
#include <vector>   // for vector

#include "../../common/transform_iterator.h"  // for MakeIndexTransformIter
#include "../tree_view.h"                     // for ScalarTreeView, MultiTargetTreeView
#include "expand_entry.h"                     // for CPUExpandEntry
#include "xgboost/logging.h"                  // for CHECK_EQ
#include "xgboost/span.h"                     // for Span
#include "xgboost/tree_model.h"               // for RegTree

namespace xgboost::tree {
template <typename TreeView>
void AssignNodes(TreeView const &tree, std::vector<CPUExpandEntry> const &candidates,
                 common::Span<ChildRows const> n_rows, common::Span<bst_node_t> nodes_to_build,
                 common::Span<bst_node_t> nodes_to_sub) {
  CHECK_EQ(nodes_to_build.size(), candidates.size());
  CHECK(n_rows.empty() || n_rows.size() == candidates.size())
      << "Need the row counts of the children of every candidate.";
  auto sum_hess = [](auto const &sums) {
    auto it = common::MakeIndexTransformIter([&](auto i) { return sums[i].GetHess(); });
    return std::accumulate(it, it + sums.size(), .0);
  };
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    auto const &c = candidates[i];
    auto build_nidx = tree.LeftChild(c.nid);
    auto subtract_nidx = tree.RightChild(c.nid);
    auto build_right = n_rows.empty() ? sum_hess(c.split.right_sum) < sum_hess(c.split.left_sum)
                                      : n_rows[i][1] < n_rows[i][0];
    if (build_right) {
      std::swap(build_nidx, subtract_nidx);
    }
    nodes_to_build[i] = build_nidx;
    nodes_to_sub[i] = subtract_nidx;
  }
}

template void AssignNodes(ScalarTreeView const &tree, std::vector<CPUExpandEntry> const &candidates,
                          common::Span<ChildRows const> n_rows,
                          common::Span<bst_node_t> nodes_to_build,
                          common::Span<bst_node_t> nodes_to_sub);
template void AssignNodes(MultiTargetTreeView const &tree,
                          std::vector<CPUExpandEntry> const &candidates,
                          common::Span<ChildRows const> n_rows,
                          common::Span<bst_node_t> nodes_to_build,
                          common::Span<bst_node_t> nodes_to_sub);
}  // namespace xgboost::tree
