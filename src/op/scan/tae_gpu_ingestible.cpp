/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <data/host_tae_representation_converters.hpp>
#include <data/sirius_converter_registry.hpp>
#include <embedding/control.hpp>
#include <embedding/tae_read.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <io/sirius_datasource.hpp>
#include <log/logging.hpp>
#include <lz4.h>
#include <op/scan/owning_table_view.hpp>
#include <op/scan/scan_utils.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/scan/tae_gpu_ingestible.hpp>
#include <tae/tae_format.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::op::scan {
namespace {

using embedding::failure;

std::size_t checked_add(std::size_t a, std::size_t b)
{
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw failure(SIRIUS_RESOURCE_EXHAUSTED, "TAE descriptor size overflow");
  return a + b;
}

struct embedded_object_metadata {
  tae::ObjectMeta object;
  bool crc{false};
  std::size_t rows{0};
};

template <typename T>
T metadata_value(std::span<const std::uint8_t> bytes, std::size_t offset)
{
  if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
    throw std::runtime_error("TAE metadata field is truncated");
  T value;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

// Validate counts and additions before the general parser allocates vectors.
// In particular block_count * BI_POS_LEN must not wrap uint32_t.
void validate_embedded_metadata(std::span<const std::uint8_t> bytes)
{
  if (bytes.size() < tae::META_V3_HEADER_LEN)
    throw std::runtime_error("TAE metadata header is truncated");
  if (metadata_value<std::uint16_t>(bytes, tae::MV3_DATA_COUNT_OFF) == 0) return;
  auto const data_offset = metadata_value<std::uint32_t>(bytes, tae::MV3_DATA_OFFSET_OFF);
  if (data_offset > bytes.size()) throw std::runtime_error("TAE metadata offset is invalid");
  auto const data         = bytes.subspan(data_offset);
  auto const columns      = metadata_value<std::uint16_t>(data, tae::BH_META_COL_CNT_OFF);
  auto const index_offset = tae::BLOCK_HEADER_SIZE + std::size_t(columns) * tae::COL_META_LEN;
  auto const blocks       = metadata_value<std::uint32_t>(data, index_offset);
  auto const index_begin  = index_offset + tae::BI_BLOCK_COUNT_LEN;
  if (blocks > (data.size() - index_begin) / tae::BI_POS_LEN)
    throw std::runtime_error("TAE metadata block index is invalid");
  // Include a second copy's worth for allocator rounding / vector owners.
  auto parsed_bytes =
    checked_add(sizeof(tae::ObjectMeta), 2 * std::size_t(blocks) * sizeof(tae::BlockInfo));
  if (parsed_bytes > embedding::tae_metadata_blob_limit)
    throw failure(SIRIUS_RESOURCE_EXHAUSTED, "TAE parsed metadata exceeds 8 MiB");
  for (std::size_t i = 0; i < blocks; ++i) {
    auto const offset = metadata_value<std::uint32_t>(data, index_begin + i * tae::BI_POS_LEN);
    auto const length = metadata_value<std::uint32_t>(data, index_begin + i * tae::BI_POS_LEN + 4);
    if (offset > data.size() || length > data.size() - offset || length < tae::BLOCK_HEADER_SIZE)
      throw std::runtime_error("TAE metadata block is truncated");
    auto const block = data.subspan(offset, length);
    auto const count = metadata_value<std::uint16_t>(block, tae::BH_META_COL_CNT_OFF);
    if (count > (length - tae::BLOCK_HEADER_SIZE) / tae::COL_META_LEN)
      throw std::runtime_error("TAE metadata columns are truncated");
    // Block offsets need not be disjoint in hostile input. Charge the full
    // repeated expansion BEFORE ParseMetadata resizes any block/column vector.
    parsed_bytes = checked_add(parsed_bytes, 2 * std::size_t(count) * sizeof(tae::ColumnMetaInfo));
    if (parsed_bytes > embedding::tae_metadata_blob_limit)
      throw failure(SIRIUS_RESOURCE_EXHAUSTED, "TAE parsed metadata exceeds 8 MiB");
  }
}

std::vector<std::uint8_t> bounded_metadata_read(io::sirius_datasource& source,
                                                bool crc,
                                                std::uint64_t offset,
                                                std::size_t length,
                                                std::stop_token stop)
{
  // At most two 8 MiB serialized buffers plus parsed metadata coexist on the
  // single metadata worker, independently of the 32 MiB immutable LRU.
  if (length > embedding::tae_metadata_blob_limit)
    throw failure(SIRIUS_RESOURCE_EXHAUSTED, "TAE object metadata exceeds 8 MiB");
  std::vector<std::uint8_t> bytes(length);
  if (stop.stop_requested()) throw failure(SIRIUS_CANCELLED, "TAE metadata scan stopped");
  if (!crc) {
    if (offset > source.size() || length > source.size() - offset ||
        source.host_read(offset, length, bytes.data()) != length)
      throw std::runtime_error("short TAE metadata read");
    return bytes;
  }
  std::array<std::uint8_t, 64u << 10> scratch;
  embedding::tae_read_at read = [&](auto off, auto count, auto* destination) {
    return source.host_read(off, count, destination);
  };
  std::size_t done = 0;
  while (done < length) {
    if (stop.stop_requested()) throw failure(SIRIUS_CANCELLED, "TAE metadata scan stopped");
    auto const n =
      embedding::read_tae_slice(read, source.size(), crc, offset + done, length - done, scratch);
    std::memcpy(bytes.data() + done, scratch.data(), n);
    done += n;
  }
  return bytes;
}

void upload_embedded_payload(tae_scan_info const& info,
                             void* device,
                             std::size_t bytes,
                             rmm::cuda_stream_view stream)
{
  if (!info.work_permit ||
      info.admitted_bytes.load(std::memory_order_acquire) < info.gpu_peak_bytes ||
      !info.gpu_peak_bytes || bytes != info.compressed_bytes)
    throw failure(SIRIUS_INVALID_STATE, "TAE payload I/O requires full GPU admission");
  if (info.stop.stop_requested()) throw failure(SIRIUS_CANCELLED, "TAE scan stopped");
  if (!info.staging) {
    info.staging = std::make_shared<pinned_host_buffer>(info.work_permit->staging_bytes(), true);
    info.work_permit->retain_staging(info.staging);
  }
  // Each attempt owns its datasource; mutable prefetch handles are never shared
  // between work units or cached as part of immutable object metadata.
  auto source = info.io_context->open_datasource(info.object_path, info.object_size);
  embedding::tae_read_at read = [&](auto off, auto count, auto* destination) {
    return source->host_read(off, count, destination);
  };
  for (auto const& chunk : info.chunks) {
    std::size_t done = 0;
    while (done < chunk.pinned_length) {
      if (info.stop.stop_requested()) throw failure(SIRIUS_CANCELLED, "TAE scan stopped");
      auto const n = embedding::read_tae_slice(read,
                                               source->size(),
                                               info.crc_wrapped,
                                               std::uint64_t(chunk.extent.offset) + done,
                                               chunk.pinned_length - done,
                                               {info.staging->data(), info.staging->size()});
      info.work_permit->record_payload_bytes(n);
      CUDF_CUDA_TRY(cudaMemcpyAsync(static_cast<std::uint8_t*>(device) + chunk.pinned_offset + done,
                                    info.staging->data(),
                                    n,
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
      // Reuse only after this DMA has retired. On a CUDA failure the descriptor
      // keeps the allocation/permit alive for task-level quiescence/quarantine.
      stream.synchronize();
      done += n;
    }
  }
}

class passthrough_coalescer final : public batch_coalescer {
 public:
  std::vector<std::unique_ptr<scan_info>> push(std::unique_ptr<scan_info> info) override
  {
    std::vector<std::unique_ptr<scan_info>> out;
    if (info) { out.push_back(std::move(info)); }
    return out;
  }

  std::vector<std::unique_ptr<scan_info>> flush() override { return {}; }
};

std::vector<std::uint8_t> decompress_metadata_lz4(const std::uint8_t* src,
                                                  std::uint32_t src_len,
                                                  std::uint32_t origin_size)
{
  std::vector<std::uint8_t> dst(origin_size);
  auto const decoded = LZ4_decompress_safe(reinterpret_cast<char const*>(src),
                                           reinterpret_cast<char*>(dst.data()),
                                           static_cast<int>(src_len),
                                           static_cast<int>(origin_size));
  if (decoded < 0 || static_cast<std::uint32_t>(decoded) != origin_size) {
    throw std::runtime_error("TAE metadata LZ4 decompression failed");
  }
  return dst;
}

void host_read_exact(io::sirius_datasource& source,
                     std::uint64_t offset,
                     std::size_t length,
                     std::uint8_t* destination)
{
  auto const read = source.host_read(offset, length, destination);
  if (read != length) { throw std::runtime_error("short TAE host read"); }
}

bool has_crc_wrapper(io::sirius_datasource& source)
{
  std::uint8_t probe[12];
  host_read_exact(source, 0, sizeof(probe), probe);

  std::uint64_t magic_at_zero{};
  std::uint64_t magic_at_four{};
  std::memcpy(&magic_at_zero, probe, sizeof(magic_at_zero));
  std::memcpy(&magic_at_four, probe + 4, sizeof(magic_at_four));
  if (magic_at_zero == tae::OBJECT_MAGIC) { return false; }
  if (magic_at_four == tae::OBJECT_MAGIC) { return true; }
  throw std::runtime_error("invalid TAE magic: neither raw nor CRC-wrapped");
}

std::vector<std::uint8_t> read_logical_bytes(io::sirius_datasource& source,
                                             std::uint64_t logical_offset,
                                             std::uint64_t length,
                                             bool crc_wrapped)
{
  if (!crc_wrapped) {
    std::vector<std::uint8_t> result(length);
    host_read_exact(source, logical_offset, length, result.data());
    return result;
  }

  auto const first_block     = logical_offset / tae::CRC_CONTENT_SIZE;
  auto const last_block      = (logical_offset + length - 1) / tae::CRC_CONTENT_SIZE;
  auto const physical_offset = first_block * tae::CRC_BLOCK_SIZE;
  auto physical_end          = (last_block + 1) * tae::CRC_BLOCK_SIZE;
  physical_end               = std::min<std::uint64_t>(physical_end, source.size());

  std::vector<std::uint8_t> raw(physical_end - physical_offset);
  host_read_exact(source, physical_offset, raw.size(), raw.data());

  std::vector<std::uint8_t> stripped;
  stripped.reserve((raw.size() / tae::CRC_BLOCK_SIZE + 1) * tae::CRC_CONTENT_SIZE);
  for (std::size_t offset = 0; offset < raw.size(); offset += tae::CRC_BLOCK_SIZE) {
    auto const remaining = raw.size() - offset;
    if (remaining <= tae::CRC_SIZE) { break; }
    auto const content = std::min<std::size_t>(tae::CRC_CONTENT_SIZE, remaining - tae::CRC_SIZE);
    stripped.insert(stripped.end(),
                    raw.begin() + static_cast<std::ptrdiff_t>(offset + tae::CRC_SIZE),
                    raw.begin() + static_cast<std::ptrdiff_t>(offset + tae::CRC_SIZE + content));
  }

  auto const content_start = first_block * tae::CRC_CONTENT_SIZE;
  auto const local_offset  = logical_offset - content_start;
  if (local_offset + length > stripped.size()) {
    throw std::runtime_error("CRC TAE read exceeds stripped content");
  }
  return {stripped.begin() + static_cast<std::ptrdiff_t>(local_offset),
          stripped.begin() + static_cast<std::ptrdiff_t>(local_offset + length)};
}

struct planned_read {
  std::uint64_t offset{};
  std::uint32_t compressed_length{};
  std::uint32_t origin_size{};
  std::uint8_t algorithm{};
  std::uint16_t column_position{};
  std::uint32_t row_count{};
  std::uint32_t null_count{};
  tae::MOTypeOid type_oid{tae::MO_T_any};
  std::int32_t width{};
  std::int32_t scale{};
};

void read_chunks_coalesced(io::sirius_datasource& source,
                           bool crc_wrapped,
                           std::vector<planned_read>& reads,
                           pinned_host_buffer& destination)
{
  std::sort(reads.begin(), reads.end(), [](planned_read const& left, planned_read const& right) {
    return left.offset < right.offset;
  });

  constexpr std::size_t max_group_reads = 32;
  std::size_t group_begin               = 0;
  std::size_t destination_offset        = 0;
  while (group_begin < reads.size()) {
    std::size_t group_end = group_begin + 1;
    while (group_end < reads.size() && group_end - group_begin < max_group_reads &&
           reads[group_end].offset ==
             reads[group_end - 1].offset + reads[group_end - 1].compressed_length) {
      ++group_end;
    }

    auto const& first      = reads[group_begin];
    auto const& last       = reads[group_end - 1];
    auto const logical_end = last.offset + last.compressed_length;
    if (!crc_wrapped) {
      host_read_exact(
        source, first.offset, logical_end - first.offset, destination.data() + destination_offset);
    } else {
      auto const first_block  = first.offset / tae::CRC_CONTENT_SIZE;
      auto const last_block   = (logical_end - 1) / tae::CRC_CONTENT_SIZE;
      auto const physical_beg = first_block * tae::CRC_BLOCK_SIZE;
      auto const physical_end =
        std::min<std::uint64_t>((last_block + 1) * tae::CRC_BLOCK_SIZE, source.size());
      std::vector<std::uint8_t> raw(physical_end - physical_beg);
      host_read_exact(source, physical_beg, raw.size(), raw.data());

      std::vector<std::uint8_t> stripped;
      stripped.reserve((raw.size() / tae::CRC_BLOCK_SIZE + 1) * tae::CRC_CONTENT_SIZE);
      for (std::size_t offset = 0; offset < raw.size(); offset += tae::CRC_BLOCK_SIZE) {
        auto const remaining = raw.size() - offset;
        if (remaining <= tae::CRC_SIZE) { break; }
        auto const content =
          std::min<std::size_t>(tae::CRC_CONTENT_SIZE, remaining - tae::CRC_SIZE);
        stripped.insert(
          stripped.end(),
          raw.begin() + static_cast<std::ptrdiff_t>(offset + tae::CRC_SIZE),
          raw.begin() + static_cast<std::ptrdiff_t>(offset + tae::CRC_SIZE + content));
      }

      auto const content_begin = first_block * tae::CRC_CONTENT_SIZE;
      std::size_t write_offset = destination_offset;
      for (std::size_t read_idx = group_begin; read_idx < group_end; ++read_idx) {
        auto const& read        = reads[read_idx];
        auto const local_offset = read.offset - content_begin;
        if (local_offset + read.compressed_length > stripped.size()) {
          throw std::runtime_error("CRC TAE coalesced read exceeds stripped content");
        }
        std::memcpy(destination.data() + write_offset,
                    stripped.data() + local_offset,
                    read.compressed_length);
        write_offset += read.compressed_length;
      }
    }

    for (std::size_t read_idx = group_begin; read_idx < group_end; ++read_idx) {
      destination_offset += reads[read_idx].compressed_length;
    }
    group_begin = group_end;
  }
}

std::vector<std::optional<std::size_t>> to_batch_positions(
  std::vector<duckdb::idx_t> const& positions)
{
  std::vector<std::optional<std::size_t>> out;
  out.reserve(positions.size());
  for (auto const position : positions) {
    if (position == static_cast<duckdb::idx_t>(-1)) {
      out.emplace_back(std::nullopt);
    } else {
      out.emplace_back(static_cast<std::size_t>(position));
    }
  }
  return out;
}

}  // namespace

void tae_ingestible_table_info::refresh_object_paths()
{
  object_paths_.clear();
  // Live sources do not participate in scan-manager cache matching. Expanding
  // a long common root into every manifest path would multiply host metadata.
  if (!bind_data || embedded_manifest) { return; }
  object_paths_.reserve(bind_data->objects.size());
  for (auto const& object : bind_data->objects) {
    object_paths_.push_back(bind_data->data_dir + "/" + object.file_path);
  }
}

tae_gpu_ingestible::tae_gpu_ingestible(std::unique_ptr<tae_ingestible_table_info> info)
  : _info(std::move(info))
{
  if (!_info || !_info->bind_data || _info->context == nullptr) {
    throw std::invalid_argument("TAE ingestible requires bind data and a client context");
  }
  if (_info->embedded_manifest && !_info->embedded_controller)
    throw failure(SIRIUS_INVALID_STATE, "embedded TAE requires a shared demand controller");
  _info->refresh_object_paths();
  _plan = build_tae_scan_plan(*_info->bind_data,
                              _info->column_ids,
                              _info->projection_ids,
                              _info->returned_types,
                              _info->output_types.size(),
                              _info->table_filters.get());

  if (_info->table_filters && !_info->table_filters->filters.empty()) {
    _filter_expression =
      op::convert_table_filters_to_expression(*_info->table_filters,
                                              _info->column_ids,
                                              _info->returned_types,
                                              to_batch_positions(_plan.batch_column_map));
  }
}

tae_gpu_ingestible::~tae_gpu_ingestible() { stop_metadata_scan(); }

void tae_scan_info::gpu_admitted(std::size_t bytes) const
{
  if (gpu_peak_bytes && bytes < gpu_peak_bytes)
    throw failure(SIRIUS_RESOURCE_EXHAUSTED, "TAE GPU reservation is below the required peak");
  admitted_bytes.store(bytes, std::memory_order_release);
}

bool tae_gpu_ingestible::live_ready() const
{
  std::lock_guard lock(_work_mutex);
  return !_metadata_stop.stop_requested() && (_ready || _live_error);
}

bool tae_gpu_ingestible::live_exhausted() const
{
  std::lock_guard lock(_work_mutex);
  return _metadata_stop.stop_requested() || (_exhausted && !_ready && !_pending && !_live_error);
}

void tae_gpu_ingestible::live_subscribe(std::shared_ptr<embedding::capacity_waker> wake)
{
  bool notify = false;
  {
    std::lock_guard lock(_work_mutex);
    _waker = wake;
    notify = _ready || _live_error || _exhausted;
  }
  // Subscription alone does not activate a scan on a later join branch.
  if (notify && wake) wake->wake();
}

void tae_gpu_ingestible::live_request()
{
  if (!is_live()) return;
  {
    std::lock_guard lock(_work_mutex);
    if (_pending || _ready || _exhausted || _live_error || _metadata_stop.stop_requested()) return;
    _demand_started = true;
    _pending        = true;
  }
  try {
    auto weak = weak_from_this();
    if (_info->embedded_controller->request([weak](auto permit) noexcept {
          if (auto source = weak.lock())
            static_cast<tae_gpu_ingestible&>(*source).produce_live(std::move(permit));
        }))
      return;
    throw failure(SIRIUS_CANCELLED, "TAE demand controller is closed");
  } catch (...) {
    std::shared_ptr<embedding::capacity_waker> wake;
    {
      std::lock_guard lock(_work_mutex);
      _pending    = false;
      _live_error = std::current_exception();
      wake        = _waker;
    }
    if (wake) wake->wake();
  }
}

void tae_gpu_ingestible::live_set_io_resolver(io::ioctx_resolver resolve)
{
  if (!resolve) throw failure(SIRIUS_INVALID_ARGUMENT, "TAE I/O resolver is empty");
  std::lock_guard lock(_work_mutex);
  if (_demand_started)
    throw failure(SIRIUS_INVALID_STATE, "TAE I/O resolver cannot change after demand");
  _info->embedded_resolve = std::move(resolve);
}

void tae_gpu_ingestible::live_stop()
{
  _metadata_stop.request_stop();
  std::unique_ptr<tae_scan_info> discarded;
  std::shared_ptr<embedding::capacity_waker> wake;
  {
    std::lock_guard lock(_work_mutex);
    discarded = std::move(_ready);
    wake      = std::move(_waker);
  }
  if (wake) wake->wake();
}

std::unique_ptr<op::operator_data> tae_gpu_ingestible::live_claim()
{
  std::unique_ptr<tae_scan_info> ready;
  {
    std::lock_guard lock(_work_mutex);
    if (_metadata_stop.stop_requested()) return {};
    if (_live_error) std::rethrow_exception(_live_error);
    ready = std::move(_ready);
  }
  if (!ready) return {};
  live_request();
  return std::make_unique<scan_operator_input>(std::move(ready));
}

void tae_gpu_ingestible::produce_live(std::shared_ptr<embedding::tae_work_permit> permit) noexcept
{
  std::unique_ptr<tae_scan_info> split;
  std::exception_ptr error;
  try {
    if (!permit) throw failure(SIRIUS_RESOURCE_EXHAUSTED, "TAE staging entitlement unavailable");
    if (!_info->embedded_resolve)
      throw failure(SIRIUS_INVALID_STATE, "TAE I/O resolver is not installed");
    if (!_metadata_stop.stop_requested()) split = plan_live_split(permit);
  } catch (...) {
    error = std::current_exception();
  }
  std::shared_ptr<embedding::capacity_waker> wake;
  {
    std::lock_guard lock(_work_mutex);
    _pending = false;
    if (!_metadata_stop.stop_requested()) {
      _exhausted  = !split && !error;
      _ready      = std::move(split);
      _live_error = std::move(error);
      wake        = _waker;
    }
  }
  if (wake) wake->wake();
}

std::unique_ptr<tae_scan_info> tae_gpu_ingestible::plan_live_split(
  std::shared_ptr<embedding::tae_work_permit> const& permit)
{
  // Cursors are touched only by this source's single outstanding metadata job.
  while (_next_object < _info->bind_data->objects.size()) {
    if (_metadata_stop.stop_requested()) return {};
    auto const& object = _info->bind_data->objects[_next_object];
    if (!object.sort_key_zm.empty() && !_plan.pushed_filters.empty() &&
        _plan.sort_column_idx >= 0 &&
        !tae::ZoneMapPassesFilters(_plan.pushed_filters,
                                   object.sort_key_zm.data(),
                                   static_cast<std::uint16_t>(_plan.sort_column_idx))) {
      ++_next_object;
      _next_block = 0;
      continue;
    }
    if (_info->bind_data->data_dir.size() >= embedding::tae_path_limit ||
        object.file_path.size() > embedding::tae_path_limit - _info->bind_data->data_dir.size() - 1)
      throw failure(SIRIUS_RESOURCE_EXHAUSTED, "TAE object path exceeds 4096 bytes");
    auto const path = _info->bind_data->data_dir + "/" + object.file_path;
    auto context    = _info->embedded_resolve(path);
    if (!context) throw std::runtime_error("TAE demand source has no I/O backend");
    auto const cache_key = path + ":" + std::to_string(object.size_bytes);
    auto metadata        = std::static_pointer_cast<const embedded_object_metadata>(
      _info->embedded_controller->cached_metadata(cache_key));
    if (!metadata) {
      auto source = context->open_datasource(path, object.size_bytes);
      auto parsed = std::make_shared<embedded_object_metadata>();
      parsed->crc = has_crc_wrapper(*source);
      auto header = bounded_metadata_read(
        *source, parsed->crc, 0, tae::HEADER_SIZE, _metadata_stop.get_token());
      auto const extent = metadata_value<tae::Extent>(header, tae::HEADER_META_EXTENT_OFF);
      if (extent.alg > 1 || extent.origin_size > embedding::tae_metadata_blob_limit)
        throw failure(SIRIUS_RESOURCE_EXHAUSTED, "TAE metadata encoding exceeds admission bounds");
      auto bytes = bounded_metadata_read(
        *source, parsed->crc, extent.offset, extent.length, _metadata_stop.get_token());
      if (extent.is_compressed())
        bytes = decompress_metadata_lz4(bytes.data(), extent.length, extent.origin_size);
      if (bytes.size() < tae::IO_ENTRY_HEADER_LEN)
        throw std::runtime_error("TAE metadata IO entry header is missing");
      auto const contents = std::span<const std::uint8_t>(bytes).subspan(tae::IO_ENTRY_HEADER_LEN);
      validate_embedded_metadata(contents);
      tae::ParseMetadata(
        contents.data(), static_cast<std::uint32_t>(contents.size()), parsed->object);
      std::size_t metadata_bytes = sizeof(*parsed) + path.size() +
                                   2 * parsed->object.blocks.capacity() * sizeof(tae::BlockInfo);
      for (auto const& block : parsed->object.blocks) {
        metadata_bytes =
          checked_add(metadata_bytes, 2 * block.columns.capacity() * sizeof(tae::ColumnMetaInfo));
        parsed->rows = checked_add(parsed->rows, block.rows);
      }
      _info->embedded_controller->cache_metadata(cache_key, parsed, metadata_bytes);
      metadata = std::move(parsed);
    }
    if (metadata->object.block_count != object.blocks)
      throw std::runtime_error("TAE manifest block count does not match object metadata");
    if (metadata->rows != object.rows)
      throw std::runtime_error("TAE manifest row count does not match object metadata");

    auto split = std::make_unique<tae_scan_info>();
    auto const chunk_limit =
      permit->chunk_limit(sizeof(host_tae_representation::column_chunk_info));
    if (_plan.projected_columns.size() > chunk_limit)
      throw failure(SIRIUS_RESOURCE_EXHAUSTED,
                    "TAE block projection exceeds work metadata entitlement");
    split->chunks.reserve(chunk_limit);
    split->work_permit = permit;
    split->io_context  = std::move(context);
    split->object_path = path;
    split->object_size = object.size_bytes;
    split->crc_wrapped = metadata->crc;
    split->stop        = _metadata_stop.get_token();
    while (_next_block < metadata->object.blocks.size()) {
      if (_metadata_stop.stop_requested()) return {};
      if (_plan.projected_columns.size() > chunk_limit - split->chunks.size()) break;
      auto const& block = metadata->object.blocks[_next_block];
      bool passes       = true;
      for (auto const sequence : _plan.filter_seqnums) {
        if (sequence >= block.columns.size())
          throw std::runtime_error("TAE object is missing a filtered column sequence");
        if (!tae::ZoneMapPassesFilters(
              _plan.pushed_filters, block.columns[sequence].zone_map, sequence)) {
          passes = false;
          break;
        }
      }
      if (!passes || !block.rows) {
        ++_next_block;
        continue;
      }
      std::size_t decoded_bytes = 0;
      for (auto const& projected : _plan.projected_columns) {
        if (projected.seqnum >= block.columns.size())
          throw std::runtime_error("TAE object is missing a projected column sequence");
        auto const& extent = block.columns[projected.seqnum].location;
        if (extent.alg > 1 || !extent.length || (extent.is_compressed() && !extent.origin_size))
          throw std::runtime_error("TAE column extent has an unsupported encoding");
        decoded_bytes =
          checked_add(decoded_bytes, extent.is_compressed() ? extent.origin_size : extent.length);
      }
      if (split->rows && (decoded_bytes > embedding::tae_target_decoded_bytes -
                                            std::min(split->uncompressed_bytes,
                                                     embedding::tae_target_decoded_bytes) ||
                          block.rows >= std::numeric_limits<std::int32_t>::max() - split->rows))
        break;
      if (block.rows >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
        throw failure(SIRIUS_RESOURCE_EXHAUSTED, "TAE block exceeds cuDF row capacity");
      for (auto const& projected : _plan.projected_columns) {
        auto const& column = block.columns[projected.seqnum];
        if (column.null_cnt > block.rows)
          throw std::runtime_error("TAE column null count exceeds block rows");
        host_tae_representation::column_chunk_info chunk{};
        chunk.column_idx        = projected.col_ids_position;
        chunk.type_oid          = projected.type_oid;
        chunk.width             = projected.width;
        chunk.scale             = projected.scale;
        chunk.extent            = column.location;
        chunk.null_cnt          = column.null_cnt;
        chunk.row_count         = block.rows;
        chunk.pinned_offset     = split->compressed_bytes;
        chunk.pinned_length     = column.location.length;
        split->compressed_bytes = checked_add(split->compressed_bytes, chunk.pinned_length);
        split->chunks.push_back(chunk);
      }
      split->rows               = checked_add(split->rows, block.rows);
      split->uncompressed_bytes = checked_add(split->uncompressed_bytes, decoded_bytes);
      ++_next_block;
      if (split->uncompressed_bytes >= embedding::tae_target_decoded_bytes) break;
    }
    if (_next_block == metadata->object.blocks.size()) {
      ++_next_object;
      _next_block = 0;
    }
    if (split->rows && !split->chunks.empty()) {
      split->gpu_peak_bytes = tae_decode_reservation_floor(split->chunks);
      return split;
    }
  }
  return {};
}

std::unique_ptr<batch_coalescer> tae_gpu_ingestible::create_batch_coalescer() const
{
  return std::make_unique<passthrough_coalescer>();
}

bool tae_gpu_ingestible::has_processed_all_metadata() const
{
  if (is_live()) return live_exhausted();
  if (_metadata_stop.stop_requested()) return true;
  std::lock_guard lock(_work_mutex);
  return _next_object >= _info->bind_data->objects.size();
}

void tae_gpu_ingestible::stop_metadata_scan() noexcept { _metadata_stop.request_stop(); }

gpu_ingestible::metadata_scan_task_t tae_gpu_ingestible::next_split_provider(
  io::ioctx_resolver resolve)
{
  if (is_live()) throw failure(SIRIUS_INVALID_STATE, "embedded TAE uses demand scheduling");
  if (!resolve) { throw std::runtime_error("TAE ingestible has no scan-manager I/O resolver"); }
  if (_metadata_stop.stop_requested()) return nullptr;
  std::size_t object_index, block_index;
  {
    std::lock_guard lock(_work_mutex);
    if (!_info->embedded_manifest) {
      if (_next_object >= _info->bind_data->objects.size()) return nullptr;
      object_index = _next_object++;
      block_index  = std::numeric_limits<std::size_t>::max();
    } else {
      while (_next_object < _info->bind_data->objects.size() &&
             _next_block >= _info->bind_data->objects[_next_object].blocks) {
        ++_next_object;
        _next_block = 0;
      }
      if (_next_object >= _info->bind_data->objects.size()) return nullptr;
      object_index = _next_object;
      block_index  = _next_block++;
    }
  }
  auto const file_path =
    _info->bind_data->data_dir + "/" + _info->bind_data->objects[object_index].file_path;
  auto io_ctx = resolve(file_path);
  return [this, object_index, block_index, io_ctx = std::move(io_ctx)] {
    return load_object(object_index, block_index, io_ctx);
  };
}

std::unique_ptr<tae_scan_info> tae_gpu_ingestible::load_object(
  std::size_t object_index,
  std::size_t block_index,
  std::shared_ptr<io::sirius_ioctx> const& io_ctx) const
{
  if (_metadata_stop.stop_requested()) return std::make_unique<tae_scan_info>();
  auto const& object   = _info->bind_data->objects.at(object_index);
  auto const file_path = _info->bind_data->data_dir + "/" + object.file_path;
  if (!io_ctx) { throw std::runtime_error("TAE ingestible received a null I/O context"); }
  if (!object.sort_key_zm.empty() && !_plan.pushed_filters.empty() && _plan.sort_column_idx >= 0 &&
      !tae::ZoneMapPassesFilters(_plan.pushed_filters,
                                 object.sort_key_zm.data(),
                                 static_cast<std::uint16_t>(_plan.sort_column_idx))) {
    return std::make_unique<tae_scan_info>();
  }

  auto source            = io_ctx->open_datasource(file_path, object.size_bytes);
  auto const crc_wrapped = has_crc_wrapper(*source);

  auto header = read_logical_bytes(*source, 0, tae::HEADER_SIZE, crc_wrapped);
  tae::Extent metadata_extent{};
  std::memcpy(
    &metadata_extent, header.data() + tae::HEADER_META_EXTENT_OFF, sizeof(metadata_extent));
  auto metadata =
    read_logical_bytes(*source, metadata_extent.offset, metadata_extent.length, crc_wrapped);
  if (metadata_extent.is_compressed()) {
    metadata =
      decompress_metadata_lz4(metadata.data(), metadata_extent.length, metadata_extent.origin_size);
  }
  if (metadata.size() <= tae::IO_ENTRY_HEADER_LEN) {
    throw std::runtime_error("TAE object metadata is shorter than its IO entry header");
  }

  tae::ObjectMeta object_metadata;
  tae::ParseMetadata(metadata.data() + tae::IO_ENTRY_HEADER_LEN,
                     static_cast<std::uint32_t>(metadata.size() - tae::IO_ENTRY_HEADER_LEN),
                     object_metadata);
  if (object_metadata.block_count != object.blocks)
    throw std::runtime_error("TAE manifest block count does not match object metadata");
  std::uint64_t object_rows = 0;
  for (auto const& block : object_metadata.blocks) {
    if (block.rows > std::numeric_limits<std::uint64_t>::max() - object_rows)
      throw std::runtime_error("TAE object row count overflow");
    object_rows += block.rows;
  }
  if (object_rows != object.rows)
    throw std::runtime_error("TAE manifest row count does not match object metadata");
  bool const whole_object = block_index == std::numeric_limits<std::size_t>::max();
  if (!whole_object && block_index >= object_metadata.block_count)
    throw std::runtime_error("TAE manifest block count exceeds object metadata");

  std::vector<planned_read> reads;
  std::size_t selected_rows = 0;
  auto const block_begin    = whole_object ? 0u : static_cast<std::uint32_t>(block_index);
  auto const block_end      = whole_object ? object_metadata.block_count : block_begin + 1;
  for (std::uint32_t current_block = block_begin; current_block < block_end; ++current_block) {
    auto const& block = object_metadata.blocks[current_block];
    bool passes       = true;
    for (auto const sequence : _plan.filter_seqnums) {
      if (sequence >= block.columns.size())
        throw std::runtime_error("TAE object is missing a filtered column sequence");
      if (!tae::ZoneMapPassesFilters(
            _plan.pushed_filters, block.columns[sequence].zone_map, sequence)) {
        passes = false;
        break;
      }
    }
    if (!passes) { continue; }

    selected_rows += block.rows;
    for (auto const& projected : _plan.projected_columns) {
      if (projected.seqnum >= block.columns.size())
        throw std::runtime_error("TAE object is missing a projected column sequence");
      auto const& column = block.columns[projected.seqnum];
      auto const& extent = column.location;
      reads.push_back({extent.offset,
                       extent.length,
                       extent.origin_size,
                       extent.alg,
                       projected.col_ids_position,
                       block.rows,
                       column.null_cnt,
                       projected.type_oid,
                       projected.width,
                       projected.scale});
    }
  }

  auto result  = std::make_unique<tae_scan_info>();
  result->rows = selected_rows;
  if (reads.empty() || selected_rows == 0) { return result; }

  for (auto const& read : reads) {
    if (read.compressed_length >
          std::numeric_limits<std::size_t>::max() - result->compressed_bytes ||
        read.origin_size > std::numeric_limits<std::size_t>::max() - result->uncompressed_bytes)
      throw sirius::embedding::failure(SIRIUS_RESOURCE_EXHAUSTED, "TAE split size overflow");
    result->compressed_bytes += read.compressed_length;
    result->uncompressed_bytes += read.origin_size;
  }
  if (_info->embedded_host_budget && result->compressed_bytes) {
    auto credit = std::make_shared<sirius::embedding::buffer_budget::lease>();
    auto status = _info->embedded_host_budget->acquire(
      result->compressed_bytes,
      _metadata_stop.get_token(),
      sirius::embedding::buffer_budget::clock::time_point::max(),
      *credit);
    if (status != SIRIUS_OK) {
      if (_metadata_stop.stop_requested()) return std::make_unique<tae_scan_info>();
      throw sirius::embedding::failure(status, "embedded TAE host staging is unavailable");
    }
    result->host_credit = std::move(credit);
  }
  auto host               = std::make_shared<pinned_host_buffer>(result->compressed_bytes);
  std::size_t host_offset = 0;
  read_chunks_coalesced(*source, crc_wrapped, reads, *host);
  result->chunks.reserve(reads.size());
  for (auto const& read : reads) {
    host_tae_representation::column_chunk_info chunk;
    chunk.column_idx    = read.column_position;
    chunk.type_oid      = read.type_oid;
    chunk.width         = read.width;
    chunk.scale         = read.scale;
    chunk.extent        = tae::Extent{read.algorithm,
                               static_cast<std::uint32_t>(read.offset),
                               read.compressed_length,
                               read.origin_size};
    chunk.null_cnt      = read.null_count;
    chunk.row_count     = read.row_count;
    chunk.pinned_offset = host_offset;
    chunk.pinned_length = read.compressed_length;
    result->chunks.push_back(chunk);
    host_offset += read.compressed_length;
  }
  result->host_data = std::move(host);
  return result;
}

std::unique_ptr<cudf::table> tae_gpu_ingestible::make_empty_table(
  const cucascade::memory::memory_space&, rmm::cuda_stream_view) const
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(_info->output_types.size());
  for (auto const& logical : _info->output_types) {
    auto const native = sirius::try_get_cudf_type(logical);
    if (!native) { throw std::runtime_error("TAE empty result has an unsupported output type"); }
    columns.push_back(cudf::make_empty_column(*native));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

filtered_table tae_gpu_ingestible::materialize_metadata_to_table(
  scan_info const& generic_info,
  const cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream,
  bool,
  std::shared_ptr<const sirius::like_multiliteral_cache>)
{
  auto const& info = dynamic_cast<tae_scan_info const&>(generic_info);
  if ((!info.host_data && !info.work_permit) || info.chunks.empty()) {
    return {.table = owning_table_view{make_empty_table(mem_space, stream)},
            .state = filter_state::UNFILTERED};
  }

  if (info.work_permit) {
    if (info.chunks.size() >
        info.work_permit->chunk_limit(sizeof(host_tae_representation::column_chunk_info)))
      throw failure(SIRIUS_RESOURCE_EXHAUSTED,
                    "TAE converter descriptors exceed work metadata entitlement");
    if (info.admitted_bytes.load(std::memory_order_acquire) < info.gpu_peak_bytes)
      throw failure(SIRIUS_INVALID_STATE, "TAE decode requires full GPU admission");
  }
  host_tae_representation host{const_cast<cucascade::memory::memory_space*>(&mem_space),
                               info.host_data,
                               info.chunks,
                               info.rows,
                               info.compressed_bytes,
                               info.uncompressed_bytes};
  if (info.work_permit) {
    host.set_device_loader(
      [&info](void* device, std::size_t bytes, rmm::cuda_stream_view upload_stream) {
        upload_embedded_payload(info, device, bytes, upload_stream);
      });
  }
  auto gpu = sirius::converter_registry::get().convert<cucascade::gpu_table_representation>(
    host, &mem_space, stream);
  return {.table = owning_table_view{gpu->release_table(stream)},
          .state = filter_state::UNFILTERED};
}

std::unique_ptr<cudf::table> tae_gpu_ingestible::post_filter_and_project(
  filtered_table&& input,
  const cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream,
  bool like_swar_fastpath,
  std::shared_ptr<const sirius::like_multiliteral_cache> like_cache,
  std::unique_ptr<cudf::column>* survivors,
  std::span<std::size_t const> elided)
{
  rmm::device_async_resource_ref mr_ref(mem_space.get_default_allocator());
  auto output_positions = _plan.post_filter_projection_ids;
  if (output_positions.empty() && !_info->output_types.empty()) {
    output_positions.resize(_info->output_types.size());
    std::iota(output_positions.begin(), output_positions.end(), std::size_t{0});
  }

  owning_table_view final_table;
  if (_filter_expression) {
    auto filter = sirius::ast::from_duckdb(*_filter_expression);
    if (!filter) { throw std::runtime_error("TAE scan could not lower its table filter"); }
    sirius::expression_evaluator evaluator(filter.get(),
                                           mr_ref,
                                           stream,
                                           sirius::strategy_from_config(),
                                           sirius::expression_evaluator::default_min_ast_size,
                                           like_swar_fastpath,
                                           std::move(like_cache));
    std::vector<cudf::size_type> cudf_output_positions;
    cudf_output_positions.reserve(output_positions.size());
    for (auto const position : output_positions) {
      cudf_output_positions.push_back(static_cast<cudf::size_type>(position));
    }
    if (survivors != nullptr && !output_positions.empty()) {
      final_table = owning_table_view{
        evaluator.select_with_survivors(input.table.view(), cudf_output_positions, *survivors)};
    } else if (output_positions.empty()) {
      final_table = owning_table_view{evaluator.select(input.table.view())};
    } else {
      final_table = owning_table_view{evaluator.select(input.table.view(), cudf_output_positions)};
    }
    // The select only enqueued its reads. Preserve the source table's reader
    // event before replacing the owner.
    input.table.record_reader_event(stream);
  } else {
    final_table = std::move(input.table);
  }

  if (!_filter_expression && !output_positions.empty()) {
    final_table.select_columns(output_positions);
  }
  if (!elided.empty() && elided.size() < final_table.view().num_columns()) {
    std::vector<std::size_t> kept;
    kept.reserve(final_table.view().num_columns() - elided.size());
    for (std::size_t position = 0;
         position < static_cast<std::size_t>(final_table.view().num_columns());
         ++position) {
      if (std::find(elided.begin(), elided.end(), position) == elided.end()) {
        kept.push_back(position);
      }
    }
    if (!kept.empty()) { final_table.select_columns(kept); }
  }
  return final_table.release(stream, mr_ref);
}

std::vector<std::size_t> tae_gpu_ingestible::materialized_column_order() const
{
  std::vector<std::size_t> order;
  order.reserve(_plan.projected_columns.size());
  for (auto const& column : _plan.projected_columns) {
    order.push_back(column.logical_idx);
  }
  return order;
}

std::shared_ptr<tae_gpu_ingestible> make_ingestible(std::unique_ptr<tae_ingestible_table_info> info)
{
  return std::make_shared<tae_gpu_ingestible>(std::move(info));
}

}  // namespace sirius::op::scan
