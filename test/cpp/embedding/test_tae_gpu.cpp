/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "cuda/tae/tae_decode_kernels.hpp"
#include "data/host_tae_representation_converters.hpp"
#include "embedding/control.hpp"
#include "embedding/tae_demand.hpp"
#include "helper/type_conversions.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "op/scan/tae_gpu_ingestible.hpp"
#include "scan/test_utils.hpp"

#include <rmm/device_buffer.hpp>

#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <future>
#include <mutex>

using namespace sirius;
using namespace sirius::embedding;
using namespace sirius::op::scan;
using namespace std::chrono_literals;

namespace {
struct counted_object final : io::sirius_io_object {
  std::string path;
  std::size_t bytes;
  counted_object(std::string p, std::size_t n) : path(std::move(p)), bytes(n) {}
  std::string const& raw_file_cache_id() const noexcept override { return path; }
  std::string const& object_path() const noexcept override { return path; }
  std::size_t size() const noexcept override { return bytes; }
};

// An instrumented sparse range backend: large-object tests never construct a
// hidden full host payload in their fixture. Only the requested slice is filled.
struct counted_io final : io::sirius_ioctx {
  std::size_t bytes{0};
  std::function<void(std::size_t, std::size_t, std::uint8_t*)> fill;
  std::atomic<std::size_t> reads{0}, largest_read{0};
  ~counted_io() override { pre_destroy(); }
  io::io_context_type type() const noexcept override { return io::io_context_type::uring; }
  void shutdown() noexcept override {}
  bool supports(std::string_view) const noexcept override { return true; }
  bool supports_device_read() const noexcept override { return false; }
  bool supports_host_to_device_read() const noexcept override { return false; }
  bool supports_vector_host_read() const noexcept override { return false; }
  io::cache::prefetching_stage preferred_prefetching_stage() const noexcept override
  {
    return io::cache::prefetching_stage::none;
  }
  std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<const cudf::io::text::byte_range_info> ranges,
    std::optional<std::size_t>) const noexcept override
  {
    return {ranges.begin(), ranges.end()};
  }
  std::size_t host_read_io(io::sirius_io_object const&,
                           std::size_t offset,
                           std::size_t count,
                           std::uint8_t* target) override
  {
    if (offset > bytes || count > bytes - offset) return 0;
    ++reads;
    largest_read.store(std::max(largest_read.load(), count));
    fill(offset, count, target);
    return count;
  }
  exec::semi_future<std::size_t> host_read_async_io(io::sirius_io_object const&,
                                                    std::size_t,
                                                    std::size_t,
                                                    std::uint8_t*) noexcept override
  {
    return {};
  }
  exec::semi_future<std::size_t> device_read_async_io(io::sirius_io_object const&,
                                                      std::size_t,
                                                      std::size_t,
                                                      std::uint8_t*,
                                                      rmm::cuda_stream_view) noexcept override
  {
    return {};
  }
  exec::semi_future<std::size_t> host_to_device_read_async_io(
    io::sirius_io_object const&,
    std::span<io::io_object_segment>,
    std::size_t,
    std::size_t,
    std::uint8_t*,
    rmm::cuda_stream_view) noexcept override
  {
    return {};
  }
  exec::semi_future<std::size_t> host_read_ranges_async_io(
    io::sirius_io_object const&, std::span<io::io_object_segment>) noexcept override
  {
    return {};
  }

 protected:
  std::shared_ptr<io::sirius_io_object> create_io_object(std::string path) override
  {
    return std::make_shared<counted_object>(std::move(path), bytes);
  }
};

struct source_waker final : capacity_waker {
  std::mutex mutex;
  std::condition_variable changed;
  std::size_t generation{0};
  void wake() noexcept override
  {
    {
      std::lock_guard lock(mutex);
      ++generation;
    }
    changed.notify_all();
  }
  bool await(tae_gpu_ingestible const& source)
  {
    std::unique_lock lock(mutex);
    return changed.wait_for(
      lock, 5s, [&] { return source.live_ready() || source.live_exhausted(); });
  }
};

std::unique_ptr<tae_ingestible_table_info> scan_info_for(
  duckdb::ClientContext& context,
  std::shared_ptr<tae_demand_controller> demand,
  std::shared_ptr<counted_io> io,
  duckdb::LogicalType type,
  tae::MOTypeOid oid)
{
  auto info                        = std::make_unique<tae_ingestible_table_info>();
  info->context                    = &context;
  info->embedded_manifest          = true;
  info->embedded_controller        = std::move(demand);
  info->embedded_resolve           = [io](std::string_view) { return io; };
  info->bind_data                  = duckdb::make_uniq<tae::TAEScanBindData>();
  info->bind_data->all_col_names   = {"v"};
  info->bind_data->all_col_types   = {type};
  info->bind_data->all_col_mo_oids = {static_cast<std::uint8_t>(oid)};
  info->bind_data->all_col_widths  = {0};
  info->bind_data->all_col_scales  = {0};
  info->bind_data->all_col_seqnums = {0};
  info->column_ids                 = {duckdb::ColumnIndex(0)};
  info->projection_ids             = {0};
  info->returned_types             = {from_duckdb(type)};
  info->output_types               = info->returned_types;
  info->names                      = {"v"};
  return info;
}

template <typename T>
void put(std::vector<std::uint8_t>& bytes, std::size_t offset, T const& value)
{
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}
}  // namespace

TEST_CASE("TAE demand publishes metadata only and reuses metadata across bounded blocks",
          "[tae_source][embedded_tae]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection connection(db);
  auto budget                               = std::make_shared<buffer_budget>(tae_host_capacity, 2);
  auto demand                               = make_tae_demand_controller(1, budget);
  auto backend                              = std::make_shared<counted_io>();
  constexpr std::size_t column_bytes        = 20u << 20;
  constexpr std::size_t metadata_offset     = 64 + column_bytes * 2;
  constexpr std::size_t object_header_bytes = tae::BLOCK_HEADER_SIZE + tae::COL_META_LEN;
  constexpr std::size_t block_start         = object_header_bytes + 4 + 2 * 8;
  std::vector<std::uint8_t> metadata(4 + 32 + block_start + 2 * object_header_bytes);
  put<std::uint16_t>(metadata, 4 + tae::MV3_DATA_COUNT_OFF, 1);
  put<std::uint32_t>(metadata, 4 + tae::MV3_DATA_OFFSET_OFF, 32);
  auto const data = std::size_t{4 + 32};
  put<std::uint16_t>(metadata, data + tae::BH_META_COL_CNT_OFF, 1);
  put<std::uint32_t>(metadata, data + object_header_bytes, 2);
  for (std::size_t block = 0; block < 2; ++block) {
    auto const position = block_start + block * object_header_bytes;
    put<std::uint32_t>(metadata, data + object_header_bytes + 4 + block * 8, position);
    put<std::uint32_t>(metadata, data + object_header_bytes + 8 + block * 8, object_header_bytes);
    put<std::uint32_t>(metadata, data + position + tae::BH_ROWS_OFF, 4);
    put<std::uint16_t>(metadata, data + position + tae::BH_COL_COUNT_OFF, 1);
    put<std::uint16_t>(metadata, data + position + tae::BH_META_COL_CNT_OFF, 1);
    auto const cm = data + position + tae::BLOCK_HEADER_SIZE;
    put<std::uint8_t>(metadata, cm + tae::CM_DATA_TYPE_OFF, tae::MO_T_int32);
    put(metadata,
        cm + tae::CM_LOCATION_OFF,
        tae::Extent{
          0, static_cast<std::uint32_t>(64 + block * column_bytes), column_bytes, column_bytes});
  }
  std::vector<std::uint8_t> header(64);
  put(header, 0, tae::OBJECT_MAGIC);
  put(header,
      tae::HEADER_META_EXTENT_OFF,
      tae::Extent{0,
                  metadata_offset,
                  static_cast<std::uint32_t>(metadata.size()),
                  static_cast<std::uint32_t>(metadata.size())});
  backend->bytes = metadata_offset + metadata.size();
  std::atomic<std::size_t> payload_reads{0}, metadata_reads{0};
  backend->fill = [&](auto offset, auto count, auto* target) {
    if (offset >= metadata_offset) {
      ++metadata_reads;
      std::memcpy(target, metadata.data() + offset - metadata_offset, count);
    } else if (offset + count <= header.size()) {
      std::memcpy(target, header.data() + offset, count);
    } else {
      ++payload_reads;
      std::memset(target, 0, count);
    }
  };
  auto info = scan_info_for(
    *connection.context, demand, backend, duckdb::LogicalType::INTEGER, tae::MO_T_int32);
  tae::ParseManifestBytes(
    R"({"database":"d","table":"t","columns":[{"name":"v","oid":22}],"objects":[{"path":"object.tae","rows":8,"blocks":2}]})",
    "/objects",
    *info->bind_data);
  auto source = make_ingestible(std::move(info));
  auto wake   = std::make_shared<source_waker>();
  source->live_subscribe(wake);
  CHECK(backend->reads == 0);  // subscription does not activate a later join scan
  CHECK(demand->inspect().active == 0);
  source->live_request();
  REQUIRE(wake->await(*source));
  auto first = source->live_claim();
  REQUIRE(first);
  auto const& a = dynamic_cast<tae_scan_info const&>(
    dynamic_cast<scan_operator_input const&>(*first).get_scan_info());
  CHECK(a.rows == 4);
  CHECK_FALSE(a.host_data);
  CHECK_FALSE(a.staging);
  CHECK(a.mandatory_gpu_reservation_bytes() > column_bytes);
  REQUIRE(wake->await(*source));
  auto second = source->live_claim();
  REQUIRE(second);
  CHECK(metadata_reads == 1);
  CHECK(payload_reads == 0);
  CHECK(demand->inspect().active == 2);
  auto retry = first->execution_lease();
  first.reset();
  CHECK(demand->inspect().active == 2);
  retry.reset();
  REQUIRE(wake->await(*source));
  CHECK(source->live_exhausted());
  CHECK(payload_reads == 0);
  second.reset();
  source->live_stop();
  demand->close();
  CHECK(budget->inspect().bytes == 0);
}

TEST_CASE("TAE greater-than-64-MiB string uploads only after reservation in bounded slices",
          "[tae_gpu][embedded_tae]")
{
  auto manager = initialize_memory_manager();
  auto* gpu    = scan_test_utils::get_space(*manager, cucascade::memory::Tier::GPU);
  REQUIRE(gpu);
  rmm::cuda_stream stream;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection connection(db);
  auto budget = std::make_shared<buffer_budget>(tae_host_capacity, 2);
  auto demand = make_tae_demand_controller(1, budget);
  std::promise<std::shared_ptr<tae_work_permit>> promised;
  auto future = promised.get_future();
  demand->request([&](auto permit) { promised.set_value(std::move(permit)); });
  REQUIRE(future.wait_for(5s) == std::future_status::ready);
  auto backend                      = std::make_shared<counted_io>();
  constexpr std::uint32_t chars     = 65u << 20;
  constexpr std::size_t area_offset = 29 + 24 + 4;
  backend->bytes                    = area_offset + chars + 5;
  std::vector<std::uint8_t> prefix(area_offset);
  prefix[29] = 0xff;  // out-of-line MO varlena
  put<std::uint32_t>(prefix, 29 + 8, chars);
  put<std::uint32_t>(prefix, 29 + 24, chars);
  backend->fill = [&](auto offset, auto count, auto* target) {
    std::memset(target, 'x', count);
    for (std::size_t i = 0; i < count; ++i) {
      if (offset + i < prefix.size())
        target[i] = prefix[offset + i];
      else if (offset + i >= area_offset + chars)
        target[i] = 0;
    }
  };
  auto source                    = make_ingestible(scan_info_for(
    *connection.context, demand, backend, duckdb::LogicalType::VARCHAR, tae::MO_T_varchar));
  auto descriptor                = std::make_unique<tae_scan_info>();
  descriptor->work_permit        = future.get();
  descriptor->io_context         = backend;
  descriptor->object_path        = "virtual-large.tae";
  descriptor->object_size        = backend->bytes;
  descriptor->rows               = 1;
  descriptor->compressed_bytes   = backend->bytes;
  descriptor->uncompressed_bytes = backend->bytes;
  host_tae_representation::column_chunk_info chunk{};
  chunk.type_oid = tae::MO_T_varchar;
  chunk.extent   = tae::Extent{
    0, 0, static_cast<std::uint32_t>(backend->bytes), static_cast<std::uint32_t>(backend->bytes)};
  chunk.row_count     = 1;
  chunk.pinned_length = backend->bytes;
  descriptor->chunks.push_back(chunk);
  descriptor->gpu_peak_bytes = tae_decode_reservation_floor(descriptor->chunks);

  CHECK_THROWS(source->materialize_metadata_to_table(*descriptor, *gpu, stream.view(), false, {}));
  CHECK_THROWS(descriptor->gpu_admitted(descriptor->gpu_peak_bytes - 1));
  CHECK(backend->reads == 0);
  CHECK_FALSE(descriptor->staging);
  auto reservation = gpu->make_reservation_or_null(descriptor->gpu_peak_bytes);
  REQUIRE(reservation);
  descriptor->gpu_admitted(reservation->size());
  auto* allocator = reservation->get_memory_resource_of<cucascade::memory::Tier::GPU>();
  allocator->attach_reservation_to_tracker(stream.view(), std::move(reservation));
  struct reset_reservation {
    cucascade::memory::reservation_aware_resource_adaptor* allocator;
    rmm::cuda_stream_view stream;
    ~reset_reservation()
    {
      stream.synchronize_no_throw();
      allocator->reset_stream_reservation(stream);
    }
  } reset{allocator, stream.view()};
  {
    auto result =
      source->materialize_metadata_to_table(*descriptor, *gpu, stream.view(), false, {});
    REQUIRE(result.table.view().num_rows() == 1);
    auto strings = cudf::strings_column_view(result.table.view().column(0));
    CHECK(strings.chars_size(stream.view()) == chars);
    std::array<char, 2> endpoints{};
    cudaMemcpyAsync(endpoints.data(),
                    strings.chars_begin(stream.view()),
                    1,
                    cudaMemcpyDeviceToHost,
                    stream.value());
    cudaMemcpyAsync(endpoints.data() + 1,
                    strings.chars_begin(stream.view()) + chars - 1,
                    1,
                    cudaMemcpyDeviceToHost,
                    stream.value());
    stream.synchronize();
    CHECK(endpoints[0] == 'x');
    CHECK(endpoints[1] == 'x');
    CHECK(backend->reads > 8);
    CHECK(backend->largest_read <= tae_max_slice);
    REQUIRE(descriptor->staging);
    CHECK(descriptor->staging->is_pinned());
    CHECK(descriptor->staging->size() <= tae_max_slice);
    CHECK(allocator->get_peak_allocated_bytes(stream.view()) <= descriptor->gpu_peak_bytes);
  }
  std::stop_source cancellation;
  descriptor->stop = cancellation.get_token();
  auto fill        = backend->fill;
  backend->fill    = [fill = std::move(fill), &cancellation](
                    auto offset, auto count, auto* destination) {
    fill(offset, count, destination);
    cancellation.request_stop();
  };
  auto const reads_before_cancel = backend->reads.load();
  CHECK_THROWS(source->materialize_metadata_to_table(*descriptor, *gpu, stream.view(), false, {}));
  CHECK(backend->reads == reads_before_cancel + 1);
  CHECK(budget->inspect().leases == 1);
  std::weak_ptr<pinned_host_buffer> physical = descriptor->staging;
  auto retiring                              = descriptor->execution_lease();
  descriptor.reset();
  CHECK_FALSE(physical.expired());
  CHECK(budget->inspect().leases == 1);
  retiring.reset();
  CHECK(physical.expired());
  CHECK(budget->inspect().bytes == 0);
  demand->close();
}

TEST_CASE("TAE coalesced null masks preserve unaligned neighboring blocks", "[tae_gpu][tae_nulls]")
{
  auto manager = initialize_memory_manager();
  rmm::cuda_stream stream;
  std::vector<std::uint8_t> nulls(24, 0);
  nulls[0]  = 0b101;      // block 0: rows 0, 2
  nulls[8]  = 0b10;       // block 1: row 1 => output row 4
  nulls[12] = 0b100;      // block 1: row 34 => output row 37
  nulls[16] = 0b1000000;  // block 2: row 6 => output row 44
  rmm::device_buffer input(nulls.data(), nulls.size(), stream.view());
  auto* bytes = static_cast<std::uint8_t const*>(input.data());
  std::vector<sirius::cuda::tae::BatchedNullMaskDesc> descriptors{
    {bytes, 3, 0}, {bytes + 8, 35, 3}, {bytes + 16, 7, 38}};
  rmm::device_buffer desc(
    descriptors.data(), descriptors.size() * sizeof(descriptors[0]), stream.view());
  std::array<std::uint32_t, 2> valid{~0u, ~0u};
  rmm::device_buffer output(valid.data(), sizeof(valid), stream.view());
  sirius::cuda::tae::batched_invert_null_mask(
    static_cast<sirius::cuda::tae::BatchedNullMaskDesc const*>(desc.data()),
    descriptors.size(),
    static_cast<std::uint32_t*>(output.data()),
    stream.view());
  cudaMemcpyAsync(
    valid.data(), output.data(), sizeof(valid), cudaMemcpyDeviceToHost, stream.value());
  stream.synchronize();
  for (std::size_t row = 0; row < 64; ++row) {
    auto const is_null = row == 0 || row == 2 || row == 4 || row == 37 || row == 44;
    CHECK(((valid[row / 32] >> (row % 32)) & 1) == !is_null);
  }
}

TEST_CASE("TAE metadata rejects aliased block expansion before parsed allocation",
          "[tae_source][embedded_tae]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection connection(db);
  auto budget                          = std::make_shared<buffer_budget>(tae_host_capacity, 2);
  auto demand                          = make_tae_demand_controller(1, budget);
  auto backend                         = std::make_shared<counted_io>();
  constexpr std::uint32_t blocks       = 16;
  constexpr std::uint16_t columns      = 8192;
  constexpr std::size_t dm             = 4 + 32;
  constexpr std::size_t index          = tae::BLOCK_HEADER_SIZE;
  constexpr std::size_t block_position = index + 4 + blocks * 8;
  constexpr std::size_t block_length   = tae::BLOCK_HEADER_SIZE + columns * tae::COL_META_LEN;
  std::vector<std::uint8_t> metadata(dm + block_position + block_length);
  put<std::uint16_t>(metadata, 4 + tae::MV3_DATA_COUNT_OFF, 1);
  put<std::uint32_t>(metadata, 4 + tae::MV3_DATA_OFFSET_OFF, 32);
  put(metadata, dm + index, blocks);
  for (std::size_t i = 0; i < blocks; ++i) {
    put<std::uint32_t>(metadata, dm + index + 4 + i * 8, block_position);
    put<std::uint32_t>(metadata, dm + index + 8 + i * 8, block_length);
  }
  put<std::uint32_t>(metadata, dm + block_position + tae::BH_ROWS_OFF, 1);
  put(metadata, dm + block_position + tae::BH_META_COL_CNT_OFF, columns);
  std::vector<std::uint8_t> object(64 + metadata.size());
  put(object, 0, tae::OBJECT_MAGIC);
  put(object,
      tae::HEADER_META_EXTENT_OFF,
      tae::Extent{0,
                  64,
                  static_cast<std::uint32_t>(metadata.size()),
                  static_cast<std::uint32_t>(metadata.size())});
  std::memcpy(object.data() + 64, metadata.data(), metadata.size());
  backend->bytes = object.size();
  backend->fill  = [&](auto offset, auto count, auto* target) {
    std::memcpy(target, object.data() + offset, count);
  };
  auto info = scan_info_for(
    *connection.context, demand, backend, duckdb::LogicalType::INTEGER, tae::MO_T_int32);
  tae::ParseManifestBytes(
    R"({"database":"d","table":"t","columns":[{"name":"v","oid":22}],"objects":[{"path":"object.tae","rows":16,"blocks":16}]})",
    "/objects",
    *info->bind_data);
  auto source = make_ingestible(std::move(info));
  auto wake   = std::make_shared<source_waker>();
  source->live_subscribe(wake);
  source->live_request();
  REQUIRE(wake->await(*source));
  try {
    source->live_claim();
    FAIL("aliased metadata expansion was accepted");
  } catch (sirius::embedding::failure const& error) {
    CHECK(error.error.code == SIRIUS_RESOURCE_EXHAUSTED);
  }
  CHECK(demand->inspect().cached_metadata_bytes == 0);
  source->live_stop();
  demand->close();
  CHECK(budget->inspect().bytes == 0);
}
