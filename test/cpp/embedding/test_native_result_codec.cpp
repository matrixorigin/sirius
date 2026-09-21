/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/result_codec.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <catch.hpp>

#include <cstring>

using namespace sirius::embedding;
namespace {
struct codec_storage : input_storage {
  std::vector<std::byte> data;
  explicit codec_storage(std::size_t bytes) : data(bytes) {}
  std::size_t size() const override { return data.size(); }
  void visit(std::function<void(std::size_t, std::span<std::byte>)> const& visit) override
  {
    // Deliberately split through scalar/varlena payloads.
    for (std::size_t offset = 0; offset < data.size(); offset += 13)
      visit(offset,
            std::span(data).subspan(offset, std::min<std::size_t>(13, data.size() - offset)));
  }
};
struct codec_pool : input_pool {
  std::size_t rounded(std::size_t bytes) const override { return bytes; }
  std::unique_ptr<input_storage> allocate(std::size_t bytes) override
  {
    return std::make_unique<codec_storage>(bytes);
  }
};
template <class T>
T read(result_batch const& batch, uint64_t offset)
{
  T value{};
  batch.storage->read(offset, {reinterpret_cast<std::byte*>(&value), sizeof(value)});
  return value;
}
}  // namespace
TEST_CASE("native result codec honors sliced validity varlena and MO date epoch",
          "[native_result_gpu]")
{
  rmm::cuda_stream stream;
  int32_t dates[]{-1, 0, 1, 2};
  int32_t offsets[]{0, 1, 4, 4, 7};
  char chars[]{'x', 'a', 'b', 'c', 'z', 'z', 'z'};
  uint32_t mask = 0b1011;  // Row two is NULL; slice rows one through two.
  rmm::device_buffer device_dates(dates, sizeof(dates), stream.view());
  rmm::device_buffer device_offsets(offsets, sizeof(offsets), stream.view());
  rmm::device_buffer device_chars(chars, sizeof(chars), stream.view());
  rmm::device_buffer device_mask(&mask, sizeof(mask), stream.view());
  stream.synchronize();
  cudf::column_view date(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS},
                         2,
                         device_dates.data(),
                         static_cast<cudf::bitmask_type const*>(device_mask.data()),
                         1,
                         1);
  cudf::column_view offsets_view(
    cudf::data_type{cudf::type_id::INT32}, 5, device_offsets.data(), nullptr, 0);
  cudf::column_view string(cudf::data_type{cudf::type_id::STRING},
                           2,
                           device_chars.data(),
                           static_cast<cudf::bitmask_type const*>(device_mask.data()),
                           1,
                           1,
                           {offsets_view});
  cudf::table_view table({date, string});
  std::vector<owned_column> schema{{50, 0, 0, true, "date"}, {61, 0, 0, true, "string"}};
  auto result = std::make_shared<native_result>();
  result->activate(std::make_shared<codec_pool>());
  std::shared_ptr<result_batch> scratch, batch;
  REQUIRE(result->try_allocate(64u << 10, 0, scratch) == SIRIUS_OK);
  auto layout = size_native_result(table, 0, schema, *scratch, stream.view(), 4096);
  REQUIRE(layout.rows == 2);
  REQUIRE(result->try_allocate(layout.bytes, 2, batch) == SIRIUS_OK);
  encode_native_result(table, 0, schema, layout, *batch, *scratch, stream.view());
  CHECK(read<int32_t>(*batch, batch->columns[0].data_offset) == 719162);
  CHECK(read<int32_t>(*batch, batch->columns[0].data_offset + 4) == 0);
  CHECK(read<uint64_t>(*batch, batch->columns[0].null_offset) == 2);
  auto const& strings = batch->columns[1];
  CHECK(read<uint32_t>(*batch, strings.data_offset) == UINT32_MAX);
  CHECK(read<uint32_t>(*batch, strings.data_offset + 4) == 0);
  CHECK(read<uint32_t>(*batch, strings.data_offset + 8) == 3);
  char observed[3]{};
  batch->storage->read(strings.area_offset,
                       {reinterpret_cast<std::byte*>(observed), sizeof(observed)});
  CHECK(std::string(observed, 3) == "abc");
  CHECK(read<uint64_t>(*batch, strings.null_offset) == 2);
  result->cancel();
}
TEST_CASE("native result codec slices at row boundaries and rejects physical mismatch",
          "[native_result_gpu]")
{
  rmm::cuda_stream stream;
  int64_t values[]{7, 8, 9};
  rmm::device_buffer data(values, sizeof(values), stream.view());
  stream.synchronize();
  cudf::table_view table(
    {cudf::column_view(cudf::data_type{cudf::type_id::INT64}, 3, data.data(), nullptr, 0)});
  std::vector<owned_column> schema{{23, 0, 0, false, "n"}};
  auto result = std::make_shared<native_result>();
  result->activate(std::make_shared<codec_pool>());
  std::shared_ptr<result_batch> scratch, batch;
  REQUIRE(result->try_allocate(64u << 10, 0, scratch) == SIRIUS_OK);
  auto layout = size_native_result(table, 0, schema, *scratch, stream.view(), 16);
  REQUIRE(layout.rows == 1);
  REQUIRE(result->try_allocate(layout.bytes, 1, batch) == SIRIUS_OK);
  encode_native_result(table, 1, schema, layout, *batch, *scratch, stream.view());
  CHECK(read<int64_t>(*batch, batch->columns[0].data_offset) == 8);
  schema[0].oid = 31;
  REQUIRE_THROWS_AS(size_native_result(table, 0, schema, *scratch, stream.view(), 16), failure);
  result->cancel();
}

TEST_CASE("native result codec keeps decimal integer bits and unsigned high bits",
          "[native_result_gpu]")
{
  rmm::cuda_stream stream;
  int64_t decimals[]{-123456789012345678LL, 123456789012345678LL};
  uint64_t unsigned_values[]{UINT64_MAX, uint64_t(1) << 63};
  rmm::device_buffer decimal_data(decimals, sizeof(decimals), stream.view());
  rmm::device_buffer unsigned_data(unsigned_values, sizeof(unsigned_values), stream.view());
  stream.synchronize();
  cudf::table_view table(
    {cudf::column_view(
       cudf::data_type{cudf::type_id::DECIMAL64, -4}, 2, decimal_data.data(), nullptr, 0),
     cudf::column_view(
       cudf::data_type{cudf::type_id::UINT64}, 2, unsigned_data.data(), nullptr, 0)});
  std::vector<owned_column> schema{{32, 18, 4, false, "decimal"}, {28, 0, 0, false, "unsigned"}};
  auto result = std::make_shared<native_result>();
  result->activate(std::make_shared<codec_pool>());
  std::shared_ptr<result_batch> scratch, batch;
  REQUIRE(result->try_allocate(64u << 10, 0, scratch) == SIRIUS_OK);
  auto layout = size_native_result(table, 0, schema, *scratch, stream.view(), 4096);
  REQUIRE(result->try_allocate(layout.bytes, 2, batch) == SIRIUS_OK);
  encode_native_result(table, 0, schema, layout, *batch, *scratch, stream.view());
  CHECK(read<int64_t>(*batch, batch->columns[0].data_offset) == decimals[0]);
  CHECK(read<int64_t>(*batch, batch->columns[0].data_offset + 8) == decimals[1]);
  CHECK(read<uint64_t>(*batch, batch->columns[1].data_offset) == UINT64_MAX);
  CHECK(read<uint64_t>(*batch, batch->columns[1].data_offset + 8) == unsigned_values[1]);
  result->cancel();
}
