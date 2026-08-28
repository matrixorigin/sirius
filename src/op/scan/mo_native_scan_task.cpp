/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "op/scan/mo_native_scan_task.hpp"

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace sirius::op::scan {
namespace {

template <typename T>
void append_scalar(std::string& output, T value)
{
  output.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::string flatten_constant(const offload::mo_native_column_view& column, std::uint32_t rows)
{
  const auto fixed        = tae::MOTypeFixedSize(static_cast<tae::MOTypeOid>(column.type.oid));
  const auto element_size = static_cast<std::size_t>(fixed < 0 ? tae::VARLENA_SIZE : fixed);
  if (!column.data.empty() && column.data.size() != element_size) {
    throw std::invalid_argument("constant MO vector has an invalid physical value size");
  }
  if (rows != 0 && element_size > std::numeric_limits<std::uint32_t>::max() / rows) {
    throw std::overflow_error("flattened MO vector exceeds uint32");
  }
  if (static_cast<std::size_t>(rows) * element_size > offload::max_expanded_native_batch_bytes) {
    throw std::overflow_error("flattened MO vector exceeds the GPU scan expansion bound");
  }

  std::string result;
  result.push_back('\0');
  result.append(reinterpret_cast<const char*>(&column.type), sizeof(column.type));
  append_scalar(result, rows);
  const auto data_size = static_cast<std::uint32_t>(element_size * rows);
  append_scalar(result, data_size);
  std::string flattened_area;
  if (fixed < 0 && !column.data.empty()) {
    tae::Varlena source{};
    std::memcpy(&source, column.data.data(), sizeof(source));
    if (!source.is_inline()) {
      if (source.big_offset() > column.area.size() ||
          source.big_length() > column.area.size() - source.big_offset() ||
          (rows != 0 && source.big_length() > std::numeric_limits<std::uint32_t>::max() / rows) ||
          static_cast<std::size_t>(rows) * source.big_length() >
            offload::max_expanded_native_batch_bytes) {
        throw std::invalid_argument("constant MO varlena exceeds its area or the flattened format");
      }
      for (std::uint32_t row = 0; row < rows; ++row) {
        auto value             = source;
        const auto area_offset = static_cast<std::uint32_t>(flattened_area.size());
        std::memcpy(value.data + 4, &area_offset, sizeof(area_offset));
        result.append(reinterpret_cast<const char*>(&value), sizeof(value));
        flattened_area.append(column.area.substr(source.big_offset(), source.big_length()));
      }
    } else {
      for (std::uint32_t row = 0; row < rows; ++row) {
        result.append(column.data);
      }
    }
  } else {
    for (std::uint32_t row = 0; row < rows; ++row) {
      if (column.data.empty())
        result.append(element_size, '\0');
      else
        result.append(column.data);
    }
  }
  const auto& area = flattened_area.empty() ? column.area : std::string_view(flattened_area);
  append_scalar(result, static_cast<std::uint32_t>(area.size()));
  result.append(area);

  const bool is_null = column.data.empty() || column.is_null(0);
  if (!is_null) {
    append_scalar(result, std::uint32_t{0});
  } else {
    const auto word_count = (rows + 63U) / 64U;
    const auto null_size  = static_cast<std::uint32_t>(24U + word_count * sizeof(std::uint64_t));
    append_scalar(result, null_size);
    append_scalar(result, static_cast<std::int64_t>(rows));
    append_scalar(result, static_cast<std::uint64_t>(rows));
    append_scalar(result, static_cast<std::uint64_t>(word_count * sizeof(std::uint64_t)));
    for (std::uint32_t word = 0; word < word_count; ++word) {
      const auto remaining = rows - word * 64U;
      const auto bits      = remaining >= 64U ? std::numeric_limits<std::uint64_t>::max()
                                              : ((std::uint64_t{1} << remaining) - 1U);
      append_scalar(result, bits);
    }
  }
  result.push_back('\0');
  return result;
}

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

void mo_native_scan_task_global_state::release_after_publish() noexcept
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

  struct staged_chunk {
    std::uint16_t output_column;
    tae::MOType type;
    std::uint32_t rows;
    std::uint32_t null_count;
    std::string encoded;
  };
  std::vector<staged_chunk> staged;
  std::size_t bytes      = 0;
  std::size_t total_rows = 0;
  bool eof               = false;
  while (bytes < offload::target_staged_native_batch_bytes) {
    auto input = op.source->next_batch();
    if (!input) {
      eof = true;
      break;
    }
    if (input->rows() == 0 || input->columns().empty() ||
        input->rows() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("GPU MO scan received an empty or oversized native batch");
    }
    SIRIUS_LOG_DEBUG("[mo_native_scan] received input sequence {} with {} rows and {} columns",
                     input->sequence(),
                     input->rows(),
                     input->columns().size());

    std::vector<staged_chunk> input_chunks;
    input_chunks.reserve(op.source_column_ids.size());
    std::size_t input_bytes = 0;
    for (std::size_t output_column = 0; output_column < op.source_column_ids.size();
         ++output_column) {
      const auto source_column = op.source_column_ids[output_column];
      if (source_column >= input->columns().size()) {
        throw std::invalid_argument("GPU MO scan projection exceeds the native batch width");
      }
      auto const& column = input->columns()[source_column];
      std::string encoded;
      if (column.vector_class == 0) {
        encoded.assign(column.encoded);
      } else if (column.vector_class == 1) {
        encoded = flatten_constant(column, static_cast<std::uint32_t>(input->rows()));
      } else {
        throw std::invalid_argument("GPU MO scan received an unsupported MatrixOne vector class");
      }
      input_bytes += encoded.size();
      if (input_bytes > offload::max_expanded_native_batch_bytes ||
          bytes > offload::max_staged_native_batch_bytes - input_bytes) {
        throw std::overflow_error("MO native batch exceeds the GPU scan staging bound");
      }
      const auto null_count = column.vector_class == 1 && (column.data.empty() || column.is_null(0))
                                ? static_cast<std::uint32_t>(input->rows())
                                : static_cast<std::uint32_t>(column.null_count);
      input_chunks.push_back({static_cast<std::uint16_t>(output_column),
                              column.type,
                              static_cast<std::uint32_t>(input->rows()),
                              null_count,
                              std::move(encoded)});
    }
    if (total_rows >
        static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()) - input->rows()) {
      throw std::overflow_error("MO native staged batch exceeds the cuDF row-count bound");
    }
    total_rows += input->rows();
    bytes += input_bytes;
    staged.insert(staged.end(),
                  std::make_move_iterator(input_chunks.begin()),
                  std::make_move_iterator(input_chunks.end()));
    const auto sequence = input->sequence();
    input.reset();
    // All Flight-owned bytes are now copied into staged strings.
    state.acknowledge(sequence);
  }

  if (staged.empty()) {
    state.finish_eof();
    return nullptr;
  }

  pinned_host_buffer host_data(bytes);
  std::vector<host_tae_representation::column_chunk_info> chunks;
  chunks.reserve(staged.size());
  std::size_t offset = 0;
  for (auto& item : staged) {
    std::memcpy(host_data.data() + offset, item.encoded.data(), item.encoded.size());
    host_tae_representation::column_chunk_info chunk{};
    chunk.column_idx         = item.output_column;
    chunk.type_oid           = static_cast<tae::MOTypeOid>(item.type.oid);
    chunk.width              = item.type.width;
    chunk.scale              = item.type.scale;
    chunk.extent             = tae::Extent{0,
                               0,
                               static_cast<std::uint32_t>(item.encoded.size()),
                               static_cast<std::uint32_t>(item.encoded.size())};
    chunk.null_cnt           = item.null_count;
    chunk.row_count          = item.rows;
    chunk.pinned_offset      = offset;
    chunk.pinned_length      = item.encoded.size();
    chunk.vector_header_size = 1 + sizeof(tae::MOType) + 4 + 4;
    chunks.push_back(chunk);
    offset += item.encoded.size();
  }

  auto representation = std::make_unique<host_tae_representation>(state.get_host_memory_space(),
                                                                  std::move(host_data),
                                                                  std::move(chunks),
                                                                  total_rows,
                                                                  bytes,
                                                                  bytes,
                                                                  nullptr,
                                                                  std::vector<std::size_t>{});
  if (eof) { state.finish_eof(); }
  auto batch =
    std::make_shared<cucascade::data_batch>(get_next_batch_id(), std::move(representation));
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
  _global_state->cast<mo_native_scan_task_global_state>().release_after_publish();
}

std::size_t mo_native_scan_task::get_estimated_reservation_size() const
{
  // Reserve for the expanded/coalesced pinned-host representation, not merely
  // the compressed 4 MiB wire frame. Under-reserving here lets concurrent
  // StreamRead tasks exceed Sirius's host-memory admission limit.
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
