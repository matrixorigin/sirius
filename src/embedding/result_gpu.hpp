/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "embedding/result.hpp"
#include "embedding/terminal_admission.hpp"
#include "op/sirius_physical_operator.hpp"

namespace sirius::embedding {
// Owns at most the admission limit of GPU publication cursors. A separate
// publisher thread parks on host capacity; GPU executor workers never wait.
class result_publisher {
 public:
  result_publisher(std::shared_ptr<native_result>,
                   std::vector<owned_column> schema,
                   std::size_t limit,
                   std::stop_token,
                   clock::time_point,
                   std::function<void(std::exception_ptr)> failed);
  ~result_publisher();
  std::shared_ptr<terminal_admission> admission() const;
  void submit(op::operator_data const&, rmm::cuda_stream_view, std::shared_ptr<terminal_ticket>);
  void finish();
  void stop();

 private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

class native_result_sink final : public op::sirius_physical_operator {
 public:
  native_result_sink(duckdb::vector<sirius::logical_type> types,
                     std::size_t cardinality,
                     std::shared_ptr<result_publisher> publisher);
  bool is_sink() const override { return true; }
  void build_pipelines(pipeline::sirius_pipeline&, pipeline::sirius_meta_pipeline&) override;
  std::unique_ptr<op::operator_data> execute(op::operator_data const&,
                                             rmm::cuda_stream_view) override;
  std::shared_ptr<terminal_admission> terminal_admission_control() const override;
  void sink_admitted(op::operator_data const&,
                     rmm::cuda_stream_view,
                     std::shared_ptr<terminal_ticket>) override;

 private:
  std::shared_ptr<result_publisher> publisher_;
};
}  // namespace sirius::embedding
