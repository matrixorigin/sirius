/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "op/scan/mo_native_scan_task.hpp"

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace sirius::op::scan {
namespace {

template <typename T>
void write_scalar(std::uint8_t*& output, T value)
{
  std::memcpy(output, &value, sizeof(value));
  output += sizeof(value);
}

struct flattened_constant_plan {
  std::size_t element_size = 0;
  std::size_t data_size    = 0;
  std::size_t area_size    = 0;
  std::size_t null_size    = 0;
  std::size_t encoded_size = 0;
  bool variable            = false;
  bool big_variable        = false;
  bool is_null             = false;
  tae::Varlena source{};
};

flattened_constant_plan plan_flatten_constant(const offload::mo_native_column_view& column,
                                              std::uint32_t rows)
{
  const auto fixed        = tae::MOTypeFixedSize(static_cast<tae::MOTypeOid>(column.type.oid));
  const auto element_size = static_cast<std::size_t>(fixed < 0 ? tae::VARLENA_SIZE : fixed);
  if (!column.data.empty() && column.data.size() != element_size) {
    throw std::invalid_argument("constant MO vector has an invalid physical value size");
  }
  if (rows != 0 && element_size > std::numeric_limits<std::uint32_t>::max() / rows) {
    throw std::overflow_error("flattened MO vector exceeds uint32");
  }
  const auto data_size = static_cast<std::size_t>(rows) * element_size;
  if (data_size > offload::max_expanded_native_batch_bytes) {
    throw std::overflow_error("flattened MO vector exceeds the GPU scan expansion bound");
  }

  flattened_constant_plan plan;
  plan.element_size = element_size;
  plan.data_size    = data_size;
  plan.variable     = fixed < 0;
  plan.is_null      = column.data.empty() || column.is_null(0);

  if (fixed < 0 && !column.data.empty()) {
    std::memcpy(&plan.source, column.data.data(), sizeof(plan.source));
    if (!plan.source.is_inline()) {
      if (plan.source.big_offset() > column.area.size() ||
          plan.source.big_length() > column.area.size() - plan.source.big_offset() ||
          (rows != 0 &&
           plan.source.big_length() > std::numeric_limits<std::uint32_t>::max() / rows) ||
          static_cast<std::size_t>(rows) * plan.source.big_length() >
            offload::max_expanded_native_batch_bytes) {
        throw std::invalid_argument("constant MO varlena exceeds its area or the flattened format");
      }
      plan.big_variable = true;
      plan.area_size    = static_cast<std::size_t>(rows) * plan.source.big_length();
    }
  }
  if (!plan.big_variable) { plan.area_size = column.area.size(); }
  if (plan.area_size > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("flattened MO vector area exceeds uint32");
  }

  if (plan.is_null) {
    const auto word_count = (static_cast<std::uint64_t>(rows) + 63U) / 64U;
    plan.null_size        = 24U + word_count * sizeof(std::uint64_t);
  }
  plan.encoded_size = 1U + sizeof(tae::MOType) + sizeof(std::uint32_t) * 4U + plan.data_size +
                      plan.area_size + plan.null_size + 1U;
  if (plan.encoded_size > offload::max_expanded_native_batch_bytes) {
    throw std::overflow_error("flattened MO vector exceeds the GPU scan expansion bound");
  }
  return plan;
}

void write_flattened_constant(std::uint8_t*& output,
                              const offload::mo_native_column_view& column,
                              std::uint32_t rows,
                              const flattened_constant_plan& plan)
{
  *output++ = 0;
  std::memcpy(output, &column.type, sizeof(column.type));
  output += sizeof(column.type);
  write_scalar(output, rows);
  write_scalar(output, static_cast<std::uint32_t>(plan.data_size));

  for (std::uint32_t row = 0; row < rows; ++row) {
    if (plan.big_variable) {
      auto value             = plan.source;
      const auto area_offset = static_cast<std::uint32_t>(row * plan.source.big_length());
      std::memcpy(value.data + 4, &area_offset, sizeof(area_offset));
      std::memcpy(output, &value, sizeof(value));
      output += sizeof(value);
    } else if (column.data.empty()) {
      std::memset(output, 0, plan.element_size);
      output += plan.element_size;
    } else {
      std::memcpy(output, column.data.data(), plan.element_size);
      output += plan.element_size;
    }
  }

  write_scalar(output, static_cast<std::uint32_t>(plan.area_size));
  if (plan.big_variable) {
    auto source = column.area.substr(plan.source.big_offset(), plan.source.big_length());
    for (std::uint32_t row = 0; row < rows; ++row) {
      std::memcpy(output, source.data(), source.size());
      output += source.size();
    }
  } else if (!column.area.empty()) {
    std::memcpy(output, column.area.data(), column.area.size());
    output += column.area.size();
  }

  write_scalar(output, static_cast<std::uint32_t>(plan.null_size));
  if (plan.null_size != 0) {
    const auto word_count = (static_cast<std::uint64_t>(rows) + 63U) / 64U;
    write_scalar(output, static_cast<std::int64_t>(rows));
    write_scalar(output, static_cast<std::uint64_t>(rows));
    write_scalar(output, word_count * sizeof(std::uint64_t));
    for (std::uint64_t word = 0; word < word_count; ++word) {
      const auto remaining = static_cast<std::uint64_t>(rows) - word * 64U;
      const auto bits      = remaining >= 64U ? std::numeric_limits<std::uint64_t>::max()
                                              : ((std::uint64_t{1} << remaining) - 1U);
      write_scalar(output, bits);
    }
  }
  *output++ = 0;
}

class mo_native_input_lease final : public host_tae_input_lease {
 public:
  mo_native_input_lease(std::weak_ptr<mo_native_scan_task_global_state> state,
                        std::unique_ptr<cucascade::memory::reservation> reservation)
    : _state(std::move(state)), _reservation(std::move(reservation))
  {
  }

  ~mo_native_input_lease() override
  {
    _reservation.reset();
    if (_h2d_complete) {
      if (auto state = _state.lock()) { state->release_after_h2d(); }
    }
  }

  void mark_h2d_complete() noexcept override { _h2d_complete = true; }

 private:
  std::weak_ptr<mo_native_scan_task_global_state> _state;
  std::unique_ptr<cucascade::memory::reservation> _reservation;
  bool _h2d_complete = false;
};

}  // namespace

mo_native_scan_task_global_state::mo_native_scan_task_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  sirius_physical_gpu_mo_scan* scan_op,
  cucascade::memory::memory_space* host_memory_space)
  : sirius_pipeline_task_global_state(std::move(pipeline)),
    _scan_op(scan_op),
    _host_memory_space(host_memory_space)
{
  if (!_scan_op || !_host_memory_space) {
    throw std::invalid_argument(
      "GPU MO scan global state requires an operator and host memory space");
  }
}

bool mo_native_scan_task_global_state::try_claim_task() noexcept
{
  bool expected = false;
  return !_scan_op->exhausted.load(std::memory_order_acquire) &&
         _scan_op->task_active.compare_exchange_strong(
           expected, true, std::memory_order_acq_rel, std::memory_order_acquire);
}

void mo_native_scan_task_global_state::finish_eof() noexcept
{
  _scan_op->exhausted.store(true, std::memory_order_release);
  _scan_op->task_active.store(false, std::memory_order_release);
}

void mo_native_scan_task_global_state::acknowledge(std::uint64_t sequence) noexcept
{
  _scan_op->source->mark_consumed(sequence);
  SIRIUS_LOG_DEBUG("[mo_native_scan] acknowledged input sequence {}", sequence);
}

void mo_native_scan_task_global_state::release_after_h2d() noexcept
{ _scan_op->task_active.store(false, std::memory_order_release); }

mo_native_scan_task::mo_native_scan_task(
  std::uint64_t task_id,
  cucascade::shared_data_repository* data_repo,
  std::unique_ptr<mo_native_scan_task_local_state> local_state,
  std::shared_ptr<mo_native_scan_task_global_state> global_state)
  : pipeline::sirius_pipeline_itask(std::move(local_state), std::move(global_state)),
    _task_id(task_id),
    _data_repo(data_repo)
{
}

mo_native_scan_task::~mo_native_scan_task()
{
  if (_global_state) {
    auto& state = _global_state->cast<mo_native_scan_task_global_state>();
    if (auto pipeline = state.get_pipeline()) { pipeline->mark_task_completed(); }
  }
}

void mo_native_scan_task::execute(rmm::cuda_stream_view stream)
{
  if (auto output = compute_task(stream); output) { publish_output(*output, stream); }
}

std::unique_ptr<operator_data> mo_native_scan_task::compute_task(rmm::cuda_stream_view /*stream*/)
{
  auto shared_state = std::static_pointer_cast<mo_native_scan_task_global_state>(_global_state);
  auto& state       = *shared_state;
  auto& op          = state.get_operator();

  auto input = op.source->next_batch();
  if (!input) {
    state.finish_eof();
    return nullptr;
  }
  if (input->rows() == 0 || input->columns().empty() ||
      input->rows() > static_cast<std::uint64_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::invalid_argument("GPU MO scan received an empty or oversized native batch");
  }
  SIRIUS_LOG_DEBUG("[mo_native_scan] received input sequence {} with {} rows and {} columns",
                   input->sequence(),
                   input->rows(),
                   input->columns().size());

  struct input_column_plan {
    const offload::mo_native_column_view* column;
    std::uint16_t output_column;
    std::size_t encoded_size;
    std::uint32_t null_count;
    std::optional<flattened_constant_plan> constant;
  };
  std::vector<input_column_plan> plans;
  plans.reserve(op.source_column_ids.size());
  std::size_t bytes = 0;
  for (std::size_t output_column = 0; output_column < op.source_column_ids.size();
       ++output_column) {
    const auto source_column = op.source_column_ids[output_column];
    if (source_column >= input->columns().size()) {
      throw std::invalid_argument("GPU MO scan projection exceeds the native batch width");
    }
    auto const& column = input->columns()[source_column];
    input_column_plan plan{&column, static_cast<std::uint16_t>(output_column), 0, 0, std::nullopt};
    if (column.vector_class == 0) {
      plan.encoded_size = column.encoded.size();
      plan.null_count   = static_cast<std::uint32_t>(column.null_count);
    } else if (column.vector_class == 1) {
      plan.constant     = plan_flatten_constant(column, static_cast<std::uint32_t>(input->rows()));
      plan.encoded_size = plan.constant->encoded_size;
      plan.null_count   = plan.constant->is_null ? static_cast<std::uint32_t>(input->rows()) : 0;
    } else {
      throw std::invalid_argument("GPU MO scan received an unsupported MatrixOne vector class");
    }
    if (plan.encoded_size > offload::max_expanded_native_batch_bytes - bytes) {
      throw std::overflow_error("MO native batch exceeds the GPU scan expansion bound");
    }
    bytes += plan.encoded_size;
    plans.push_back(std::move(plan));
  }

  pinned_host_buffer host_data(bytes);
  std::vector<host_tae_representation::column_chunk_info> chunks;
  chunks.reserve(plans.size());
  std::size_t offset = 0;
  for (auto& item : plans) {
    auto* output = host_data.data() + offset;
    if (item.constant) {
      write_flattened_constant(
        output, *item.column, static_cast<std::uint32_t>(input->rows()), *item.constant);
      if (static_cast<std::size_t>(output - (host_data.data() + offset)) != item.encoded_size) {
        throw std::runtime_error("flattened MO vector size does not match its preflight");
      }
    } else {
      std::memcpy(output, item.column->encoded.data(), item.encoded_size);
    }
    host_tae_representation::column_chunk_info chunk{};
    chunk.column_idx         = item.output_column;
    chunk.type_oid           = static_cast<tae::MOTypeOid>(item.column->type.oid);
    chunk.width              = item.column->type.width;
    chunk.scale              = item.column->type.scale;
    chunk.extent             = tae::Extent{0,
                                           0,
                                           static_cast<std::uint32_t>(item.encoded_size),
                                           static_cast<std::uint32_t>(item.encoded_size)};
    chunk.null_cnt           = item.null_count;
    chunk.row_count          = static_cast<std::uint32_t>(input->rows());
    chunk.pinned_offset      = offset;
    chunk.pinned_length      = item.encoded_size;
    chunk.vector_header_size = 1 + sizeof(tae::MOType) + 4 + 4;
    chunks.push_back(chunk);
    offset += item.encoded_size;
  }

  auto& local_state = _local_state->cast<mo_native_scan_task_local_state>();
  auto reservation  = local_state.release_reservation();
  if (!reservation || reservation->size() < offload::mo_native_scan_reservation_bytes()) {
    throw std::runtime_error("GPU MO scan lost its host reservation before publication");
  }
  auto lease = std::make_unique<mo_native_input_lease>(shared_state, std::move(reservation));
  auto representation = std::make_unique<host_tae_representation>(state.get_host_memory_space(),
                                                                  std::move(host_data),
                                                                  std::move(chunks),
                                                                  input->rows(),
                                                                  bytes,
                                                                  bytes,
                                                                  nullptr,
                                                                  std::vector<std::size_t>{},
                                                                  std::move(lease));
  auto batch =
    std::make_shared<cucascade::data_batch>(get_next_batch_id(), std::move(representation));
  // Sirius now owns the complete frame. This permits exactly one subsequent
  // frame to occupy the sidecar's prefetch slot; the source claim remains held
  // until this representation completes H2D.
  const auto sequence = input->sequence();
  input.reset();
  state.acknowledge(sequence);
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(batch)});
}

void mo_native_scan_task::publish_output(operator_data& output_data,
                                         rmm::cuda_stream_view /*stream*/)
{
  auto& output = dynamic_cast<pipelineable_operator_data&>(output_data);
  for (auto& batch : output.release_data_batches()) {
    _data_repo->add_data_batch(std::move(batch));
  }
}

std::size_t mo_native_scan_task::get_estimated_reservation_size() const
{
  // Reserve for one maximally expanded frame, not merely its 4 MiB wire form.
  // Ownership moves into the published representation until H2D completes.
  return _local_state->cast<mo_native_scan_task_local_state>().get_task_consumption_basis();
}

std::vector<sirius_physical_operator*> mo_native_scan_task::get_output_consumers()
{
  std::vector<sirius_physical_operator*> result;
  auto& op = _global_state->cast<mo_native_scan_task_global_state>().get_operator();
  for (auto& port : op.get_next_port_after_sink()) {
    result.push_back(port.next_operator);
  }
  return result;
}

}  // namespace sirius::op::scan
