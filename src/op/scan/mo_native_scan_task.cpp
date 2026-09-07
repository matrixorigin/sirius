/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "op/scan/mo_native_scan_task.hpp"

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_operator.hpp"

#include <bit>
#include <chrono>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace sirius::op::scan {
namespace {

struct host_buffer_writer {
  pinned_host_buffer& buffer;
  std::size_t offset;
  void write(void const* data, std::size_t bytes)
  {
    buffer.copy_from(offset, data, bytes);
    offset += bytes;
  }
};

template <typename T>
void write_scalar(host_buffer_writer& output, T value)
{
  output.write(&value, sizeof(value));
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

  if (fixed < 0 && !plan.is_null) {
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
  if (!plan.big_variable && !plan.is_null) { plan.area_size = column.area.size(); }
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

void write_flattened_constant(host_buffer_writer& output,
                              const offload::mo_native_column_view& column,
                              std::uint32_t rows,
                              const flattened_constant_plan& plan)
{
  write_scalar(output, std::uint8_t{0});
  output.write(&column.type, sizeof(column.type));
  write_scalar(output, rows);
  write_scalar(output, static_cast<std::uint32_t>(plan.data_size));

  for (std::uint32_t row = 0; row < rows; ++row) {
    if (plan.big_variable) {
      auto value             = plan.source;
      const auto area_offset = static_cast<std::uint32_t>(row * plan.source.big_length());
      std::memcpy(value.data + 4, &area_offset, sizeof(area_offset));
      output.write(&value, sizeof(value));
    } else if (plan.is_null) {
      const std::uint8_t zero[tae::VARLENA_SIZE]{};
      if (plan.element_size > sizeof(zero)) {
        throw std::invalid_argument("unsupported MO constant element size");
      }
      output.write(zero, plan.element_size);
    } else {
      output.write(column.data.data(), plan.element_size);
    }
  }

  write_scalar(output, static_cast<std::uint32_t>(plan.area_size));
  if (plan.big_variable) {
    auto source = column.area.substr(plan.source.big_offset(), plan.source.big_length());
    for (std::uint32_t row = 0; row < rows; ++row) {
      output.write(source.data(), source.size());
    }
  } else if (plan.area_size != 0) {
    output.write(column.area.data(), column.area.size());
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
  write_scalar(output, std::uint8_t{0});
}

struct input_column_plan {
  const offload::mo_native_column_view* column;
  std::uint16_t output_column;
  std::size_t encoded_size;
  std::uint32_t null_count;
  std::optional<flattened_constant_plan> constant;
};

struct native_frame_plan {
  std::shared_ptr<offload::mo_native_batch> input;
  std::vector<input_column_plan> columns;
  std::size_t bytes = 0;
};

native_frame_plan plan_native_frame(std::shared_ptr<offload::mo_native_batch> input,
                                    const std::vector<std::size_t>& source_column_ids)
{
  if (!input || input->rows() == 0 || input->columns().empty() ||
      input->rows() > static_cast<std::uint64_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::invalid_argument("GPU MO scan received an empty or oversized native batch");
  }

  native_frame_plan result;
  result.input = std::move(input);
  result.columns.reserve(source_column_ids.size());
  for (std::size_t output_column = 0; output_column < source_column_ids.size(); ++output_column) {
    const auto source_column = source_column_ids[output_column];
    if (source_column >= result.input->columns().size()) {
      throw std::invalid_argument("GPU MO scan projection exceeds the native batch width");
    }
    auto const& column = result.input->columns()[source_column];
    input_column_plan plan{&column, static_cast<std::uint16_t>(output_column), 0, 0, std::nullopt};
    if (column.vector_class == 0) {
      plan.encoded_size = column.encoded.size();
      plan.null_count   = static_cast<std::uint32_t>(column.null_count);
    } else if (column.vector_class == 1) {
      plan.constant =
        plan_flatten_constant(column, static_cast<std::uint32_t>(result.input->rows()));
      plan.encoded_size = plan.constant->encoded_size;
      plan.null_count =
        plan.constant->is_null ? static_cast<std::uint32_t>(result.input->rows()) : 0;
    } else {
      throw std::invalid_argument("GPU MO scan received an unsupported MatrixOne vector class");
    }
    if (plan.encoded_size > offload::max_expanded_native_batch_bytes - result.bytes) {
      throw std::overflow_error("MO native batch exceeds the GPU scan expansion bound");
    }
    result.bytes += plan.encoded_size;
    result.columns.push_back(std::move(plan));
  }
  return result;
}

void append_native_frame(native_frame_plan& plan,
                         pinned_host_buffer& host_data,
                         std::size_t& offset,
                         std::vector<host_tae_representation::column_chunk_info>& chunks)
{
  if (plan.bytes > host_data.capacity() - offset) {
    throw std::logic_error("MO native source batch exceeds its host buffer capacity");
  }
  for (auto& item : plan.columns) {
    host_buffer_writer output{host_data, offset};
    if (item.constant) {
      write_flattened_constant(
        output, *item.column, static_cast<std::uint32_t>(plan.input->rows()), *item.constant);
      if (output.offset - offset != item.encoded_size) {
        throw std::runtime_error("flattened MO vector size does not match its preflight");
      }
    } else {
      output.write(item.column->encoded.data(), item.encoded_size);
      if (item.null_count != 0 &&
          tae::MOTypeFixedSize(static_cast<tae::MOTypeOid>(item.column->type.oid)) < 0) {
        // MO may keep stale descriptors in NULL string slots. GPU varlena
        // decoding reads every slot, so replace only those descriptors with
        // empty values while preserving the bitmap and the source batch.
        const tae::Varlena empty{};
        auto words = item.column->null_words;
        for (std::size_t word_offset = 0; word_offset + sizeof(std::uint64_t) <= words.size();
             word_offset += sizeof(std::uint64_t)) {
          const auto first_row = word_offset / sizeof(std::uint64_t) * 64U;
          if (first_row >= plan.input->rows()) { break; }
          std::uint64_t bits;
          std::memcpy(&bits, words.data() + word_offset, sizeof(bits));
          while (bits != 0) {
            const auto row = first_row + std::countr_zero(bits);
            if (row >= plan.input->rows()) { break; }
            host_data.copy_from(
              offset + 1 + sizeof(tae::MOType) + 8 + row * sizeof(empty), &empty, sizeof(empty));
            bits &= bits - 1;
          }
        }
      }
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
    chunk.row_count          = static_cast<std::uint32_t>(plan.input->rows());
    chunk.pinned_offset      = offset;
    chunk.pinned_length      = item.encoded_size;
    chunk.vector_header_size = 1 + sizeof(tae::MOType) + 4 + 4;
    chunks.push_back(chunk);
    offset += item.encoded_size;
  }
}

class mo_native_input_lease final : public host_tae_input_lease {
 public:
  explicit mo_native_input_lease(std::weak_ptr<mo_native_scan_task_global_state> state)
    : _state(std::move(state))
  {
  }

  ~mo_native_input_lease() override
  {
    if (_h2d_complete) {
      if (auto state = _state.lock()) { state->release_after_h2d(); }
    }
  }

  void mark_h2d_complete() noexcept override { _h2d_complete = true; }

 private:
  std::weak_ptr<mo_native_scan_task_global_state> _state;
  bool _h2d_complete = false;
};

}  // namespace

mo_native_scan_task_global_state::mo_native_scan_task_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  sirius_physical_gpu_mo_scan* scan_op,
  cucascade::memory::memory_space* host_memory_space,
  std::size_t source_batch_target_bytes,
  std::size_t source_batch_capacity_bytes)
  : sirius_pipeline_task_global_state(std::move(pipeline)),
    _scan_op(scan_op),
    _host_memory_space(host_memory_space),
    _source_batch_target_bytes(source_batch_target_bytes),
    _source_batch_capacity_bytes(source_batch_capacity_bytes)
{
  if (!_scan_op || !_host_memory_space) {
    throw std::invalid_argument(
      "GPU MO scan global state requires an operator and host memory space");
  }
  if (_source_batch_target_bytes == 0 || _source_batch_capacity_bytes == 0 ||
      _source_batch_target_bytes > _source_batch_capacity_bytes ||
      _source_batch_capacity_bytes > offload::max_expanded_native_batch_bytes) {
    throw std::invalid_argument("GPU MO scan source batch limits are invalid");
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

bool mo_native_scan_task_global_state::acknowledge(std::uint64_t sequence)
{
  if (!_scan_op->source->mark_consumed(sequence)) { return false; }
  SIRIUS_LOG_DEBUG("[mo_native_scan] acknowledged input sequence {}", sequence);
  return true;
}

void mo_native_scan_task_global_state::release_after_h2d() noexcept
{
  _scan_op->task_active.store(false, std::memory_order_release);
}

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
  const auto target_bytes   = state.get_source_batch_target_bytes();
  const auto capacity_bytes = state.get_source_batch_capacity_bytes();

  std::optional<pinned_host_buffer> host_data;
  std::vector<host_tae_representation::column_chunk_info> chunks;
  std::size_t staged_bytes  = 0;
  std::size_t staged_frames = 0;
  std::uint64_t total_rows  = 0;
#ifdef SIRIUS_PROFILE
  const auto profile_start = std::chrono::steady_clock::now();
  double wait_ms = 0, allocation_ms = 0, copy_ms = 0;
  auto profile_elapsed = [](auto start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
  };
#endif

  while (staged_frames < offload::max_native_frames_per_source_task &&
         staged_bytes < target_bytes) {
#ifdef SIRIUS_PROFILE
    const auto wait_start = std::chrono::steady_clock::now();
#endif
    auto next = op.source->next_batch();
#ifdef SIRIUS_PROFILE
    wait_ms += profile_elapsed(wait_start);
#endif
    if (next.status == offload::mo_native_batch_source_status::NOT_NEEDED) {
      state.finish_eof();
      return nullptr;
    }
    if (next.status == offload::mo_native_batch_source_status::END_OF_STREAM) {
      state.finish_eof();
      break;
    }
    if (!next.batch) {
      throw std::runtime_error("GPU MO scan source returned an empty batch result");
    }

    auto plan = plan_native_frame(std::move(next.batch), op.source_column_ids);
    SIRIUS_LOG_DEBUG("[mo_native_scan] received input sequence {} with {} rows and {} columns",
                     plan.input->sequence(),
                     plan.input->rows(),
                     plan.input->columns().size());

    const auto max_rows = static_cast<std::uint64_t>(std::numeric_limits<cudf::size_type>::max());
    if (staged_frames != 0 && (plan.bytes > capacity_bytes - staged_bytes ||
                               plan.input->rows() > max_rows - total_rows)) {
      break;
    }
    if (plan.bytes > capacity_bytes || plan.input->rows() > max_rows) {
      throw std::overflow_error("MO native frame exceeds the source batch hard bound");
    }
#ifdef SIRIUS_PROFILE
    const auto allocation_start = std::chrono::steady_clock::now();
#endif
    if (!host_data) {
      auto* pool =
        state.get_host_memory_space()->get_memory_resource_of<cucascade::memory::Tier::HOST>();
      if (!pool) { throw std::runtime_error("GPU MO scan requires a host buffer pool"); }
      auto& local_state = _local_state->cast<mo_native_scan_task_local_state>();
      host_data.emplace(capacity_bytes, *pool, local_state.release_reservation());
    }
#ifdef SIRIUS_PROFILE
    allocation_ms += profile_elapsed(allocation_start);
    const auto copy_start = std::chrono::steady_clock::now();
#endif
    append_native_frame(plan, *host_data, staged_bytes, chunks);
#ifdef SIRIUS_PROFILE
    copy_ms += profile_elapsed(copy_start);
#endif
    total_rows += plan.input->rows();
    ++staged_frames;

    const auto sequence = plan.input->sequence();
    if (!state.acknowledge(sequence)) {
      state.finish_eof();
      return nullptr;
    }
  }

  if (staged_frames == 0) { return nullptr; }
  host_data->set_logical_size(staged_bytes);
#ifdef SIRIUS_PROFILE
  SIRIUS_LOG_INFO(
    "[mo_native_source_profile] frames={} bytes={} wait_ms={:.3f} allocation_ms={:.3f} "
    "copy_ms={:.3f} total_ms={:.3f}",
    staged_frames,
    staged_bytes,
    wait_ms,
    allocation_ms,
    copy_ms,
    profile_elapsed(profile_start));
#endif

  auto lease          = std::make_unique<mo_native_input_lease>(shared_state);
  auto representation = std::make_unique<host_tae_representation>(state.get_host_memory_space(),
                                                                  std::move(*host_data),
                                                                  std::move(chunks),
                                                                  total_rows,
                                                                  staged_bytes,
                                                                  staged_bytes,
                                                                  nullptr,
                                                                  std::vector<std::size_t>{},
                                                                  std::move(lease));
  auto batch =
    std::make_shared<cucascade::data_batch>(get_next_batch_id(), std::move(representation));
  SIRIUS_LOG_DEBUG(
    "[mo_native_scan] produced source batch with {} frames, {} rows, and {} expanded bytes",
    staged_frames,
    total_rows,
    staged_bytes);
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
  // Reserve the complete bounded source batch, not merely one 4 MiB wire frame.
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
