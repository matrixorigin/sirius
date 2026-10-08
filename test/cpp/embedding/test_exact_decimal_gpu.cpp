/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/result_codec.hpp"
#include "numeric/exact_decimal_gpu.hpp"

#include <cudf/copying.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>

#include <array>
#include <cstring>
#include <vector>

using namespace sirius::mo_decimal;
#include "exact_decimal_wide_cases.hpp"
namespace {
coefficient decimal_fixture(char const* text)
{
  coefficient result;
  bool negative = *text == '-';
  if (negative) ++text;
  for (; *text; ++text) {
    uint64_t carry = *text - '0';
    for (auto& word : result.words) {
      auto value = uint64_t(word) * 10 + carry;
      word       = value;
      carry      = value >> 32;
    }
    REQUIRE(carry == 0);
  }
  if (negative) {
    uint64_t carry = 1;
    for (auto& word : result.words) {
      auto value = uint64_t(~word) + carry;
      word       = value;
      carry      = value >> 32;
    }
  }
  return result;
}
coefficient small(int64_t value)
{
  coefficient result;
  auto bits       = static_cast<uint64_t>(value);
  result.words[0] = bits;
  result.words[1] = bits >> 32;
  for (int i = 2; i < 8; ++i)
    result.words[i] = value < 0 ? UINT32_MAX : 0;
  return result;
}
std::unique_ptr<cudf::column> input_column(decimal_type type,
                                           std::vector<coefficient> const& values,
                                           uint32_t validity,
                                           rmm::cuda_stream_view stream)
{
  auto column = make_decimal_column(type,
                                    values.size(),
                                    cudf::mask_state::ALL_VALID,
                                    stream,
                                    cudf::get_current_device_resource_ref());
  auto view   = column->mutable_view();
  if (type.bits != 256) {
    std::vector<uint8_t> raw(values.size() * type.bytes());
    for (std::size_t i = 0; i < values.size(); ++i)
      store_coefficient(values[i], raw.data() + i * type.bytes(), type.bytes());
    REQUIRE(
      cudaMemcpyAsync(
        view.data<uint8_t>(), raw.data(), raw.size(), cudaMemcpyHostToDevice, stream.value()) ==
      cudaSuccess);
    stream.synchronize();
  } else {
    std::vector<uint64_t> lane(values.size());
    for (int i = 0; i < 4; ++i) {
      for (std::size_t row = 0; row < values.size(); ++row)
        lane[row] = uint64_t(values[row].words[(3 - i) * 2]) |
                    uint64_t(values[row].words[(3 - i) * 2 + 1]) << 32;
      REQUIRE(cudaMemcpyAsync(view.child(i).data<uint64_t>(),
                              lane.data(),
                              lane.size() * 8,
                              cudaMemcpyHostToDevice,
                              stream.value()) == cudaSuccess);
      stream.synchronize();
    }
  }
  REQUIRE(cudaMemcpyAsync(view.null_mask(), &validity, 4, cudaMemcpyHostToDevice, stream.value()) ==
          cudaSuccess);
  stream.synchronize();
  column->set_null_count(cudf::null_count(view.null_mask(), 0, values.size(), stream));
  return column;
}
std::vector<coefficient> download(cudf::column_view column,
                                  uint32_t bytes,
                                  rmm::cuda_stream_view stream)
{
  std::vector<coefficient> result(column.size());
  if (column.type().id() == cudf::type_id::STRUCT) {
    std::vector<uint64_t> lane(column.size());
    for (int i = 0; i < 4; ++i) {
      REQUIRE(cudaMemcpyAsync(lane.data(),
                              column.child(i).head<uint64_t>() + column.offset(),
                              lane.size() * 8,
                              cudaMemcpyDeviceToHost,
                              stream.value()) == cudaSuccess);
      stream.synchronize();
      for (std::size_t row = 0; row < result.size(); ++row) {
        result[row].words[(3 - i) * 2]     = lane[row];
        result[row].words[(3 - i) * 2 + 1] = lane[row] >> 32;
      }
    }
  } else {
    std::vector<uint8_t> raw(result.size() * bytes);
    REQUIRE(
      cudaMemcpyAsync(
        raw.data(), column.head<uint8_t>(), raw.size(), cudaMemcpyDeviceToHost, stream.value()) ==
      cudaSuccess);
    stream.synchronize();
    for (std::size_t row = 0; row < result.size(); ++row) {
      if (bytes == 1)
        result[row].words[0] = raw[row];
      else
        result[row] = load_coefficient(raw.data() + row * bytes, bytes);
    }
  }
  return result;
}
struct storage : sirius::embedding::input_storage {
  std::vector<std::byte> data;
  explicit storage(std::size_t size) : data(size) {}
  std::size_t size() const override { return data.size(); }
  void visit(std::function<void(std::size_t, std::span<std::byte>)> const& visitor) override
  {
    for (std::size_t at = 0; at < data.size(); at += 13)
      visitor(at, std::span(data).subspan(at, std::min<std::size_t>(13, data.size() - at)));
  }
};
struct pool : sirius::embedding::input_pool {
  std::size_t rounded(std::size_t size) const override { return size; }
  std::unique_ptr<sirius::embedding::input_storage> allocate(std::size_t size) override
  {
    return std::make_unique<storage>(size);
  }
};
}  // namespace

TEST_CASE("MO checked decimal GPU kernels preserve masks and all coefficient widths",
          "[exact_decimal_gpu]")
{
  rmm::cuda_stream stream;
  std::vector<coefficient> left{small(125), small(-125), small(9007199254740993LL), small(0)};
  std::vector<coefficient> right{small(100), small(100), small(1), small(0)};
  bool active[]{true, true, false, true};
  rmm::device_buffer mask(active, sizeof(active), stream.view());
  stream.synchronize();
  for (auto width : {64u, 128u, 256u}) {
    decimal_type type{static_cast<uint16_t>(width),
                      static_cast<uint8_t>(width == 64    ? 18
                                           : width == 128 ? 38
                                                          : 65),
                      2};
    auto a = input_column(type, left, 0b0111, stream.view());
    auto b = input_column(type, right, 0b1111, stream.view());
    for (auto op : {decimal_op::add,
                    decimal_op::subtract,
                    decimal_op::multiply,
                    decimal_op::divide,
                    decimal_op::integer_divide,
                    decimal_op::modulo,
                    decimal_op::negate,
                    decimal_op::cast,
                    decimal_op::equal,
                    decimal_op::not_equal,
                    decimal_op::less,
                    decimal_op::less_equal,
                    decimal_op::greater,
                    decimal_op::greater_equal}) {
      INFO("width=" << width << " op=" << static_cast<int>(op));
      auto result = evaluate_decimal_columns(op,
                                             a->view(),
                                             type,
                                             b->view(),
                                             type,
                                             type,
                                             static_cast<bool const*>(mask.data()),
                                             stream.view(),
                                             cudf::get_current_device_resource_ref());
      auto bytes  = op >= decimal_op::equal            ? 1
                    : op == decimal_op::integer_divide ? 8
                                                       : type.bytes();
      auto values = download(result.values->view(), bytes, stream.view());
      std::array<uint8_t, 4> errors{};
      uint32_t validity{};
      REQUIRE(cudaMemcpyAsync(
                errors.data(), result.errors.data(), 4, cudaMemcpyDeviceToHost, stream.value()) ==
              cudaSuccess);
      REQUIRE(cudaMemcpyAsync(&validity,
                              result.values->view().null_mask(),
                              4,
                              cudaMemcpyDeviceToHost,
                              stream.value()) == cudaSuccess);
      stream.synchronize();
      for (uint32_t row = 0; row < 4; ++row) {
        auto expected = evaluate_decimal(
          op, left[row], type, row != 3, right[row], type, true, type, active[row]);
        CHECK(errors[row] == static_cast<uint8_t>(expected.error));
        CHECK(bool((validity >> row) & 1) == expected.valid);
        if (expected.valid)
          for (int limb = 0; limb < 8; ++limb)
            CHECK(values[row].words[limb] == expected.value.words[limb]);
      }
      CHECK(result.values->null_count() == 2);
    }
    // The operation owns the failure class even when the two operations
    // overflow the same declared coefficient domain. Inactive and NULL rows
    // must retain neither a value nor an arithmetic failure.
    for (auto op : {decimal_op::add, decimal_op::cast}) {
      auto result = evaluate_decimal_columns(op,
                                             a->view(),
                                             type,
                                             a->view(),
                                             type,
                                             {type.bits, 2, 2},
                                             static_cast<bool const*>(mask.data()),
                                             stream.view(),
                                             cudf::get_current_device_resource_ref());
      std::array<uint8_t, 4> errors{};
      REQUIRE(cudaMemcpyAsync(errors.data(),
                              result.errors.data(),
                              errors.size(),
                              cudaMemcpyDeviceToHost,
                              stream.value()) == cudaSuccess);
      stream.synchronize();
      auto expected =
        op == decimal_op::cast ? decimal_error::invalid_input : decimal_error::out_of_range;
      CHECK(errors[0] == static_cast<uint8_t>(expected));
      CHECK(errors[1] == static_cast<uint8_t>(expected));
      CHECK(errors[2] == 0);
      CHECK(errors[3] == 0);
      CHECK(result.values->null_count() == 4);
    }
    // A sliced parent carries both a data offset and a validity offset;
    // Decimal256 children remain in their original physical allocation.
    auto sliced_a = cudf::slice(a->view(), {2, 4});
    auto sliced_b = cudf::slice(b->view(), {2, 4});
    auto sliced   = evaluate_decimal_columns(decimal_op::cast,
                                           sliced_a.front(),
                                           type,
                                           sliced_b.front(),
                                           type,
                                           type,
                                           nullptr,
                                           stream.view(),
                                           cudf::get_current_device_resource_ref());
    auto observed = download(sliced.values->view(), type.bytes(), stream.view());
    CHECK(sliced.values->null_count() == 1);
    for (int word = 0; word < 8; ++word)
      CHECK(observed[0].words[word] == left[2].words[word]);
  }
}

TEST_CASE("MO Decimal64 validity spans complete warps and a partial tail", "[exact_decimal_gpu]")
{
  rmm::cuda_stream stream;
  decimal_type type{64, 18, 0};
  std::vector<coefficient> ones(65, small(1));
  auto a = input_column(type, ones, UINT32_MAX, stream.view());
  auto b = input_column(type, ones, UINT32_MAX, stream.view());
  uint32_t validity[]{UINT32_MAX ^ (uint32_t(1) << 31), UINT32_MAX ^ 1u, 1};
  REQUIRE(cudaMemcpyAsync(a->mutable_view().null_mask(),
                          validity,
                          sizeof(validity),
                          cudaMemcpyHostToDevice,
                          stream.value()) == cudaSuccess);
  std::array<bool, 65> active;
  active.fill(true);
  active[63] = false;
  rmm::device_buffer mask(active.data(), active.size() * sizeof(bool), stream.view());
  stream.synchronize();
  a->set_null_count(2);
  auto result = evaluate_decimal_columns(decimal_op::add,
                                         a->view(),
                                         type,
                                         b->view(),
                                         type,
                                         type,
                                         static_cast<bool const*>(mask.data()),
                                         stream.view(),
                                         cudf::get_current_device_resource_ref());
  auto values = download(result.values->view(), 8, stream.view());
  std::array<uint8_t, 65> errors{};
  REQUIRE(
    cudaMemcpyAsync(
      errors.data(), result.errors.data(), errors.size(), cudaMemcpyDeviceToHost, stream.value()) ==
    cudaSuccess);
  REQUIRE(cudaMemcpyAsync(validity,
                          result.values->view().null_mask(),
                          sizeof(validity),
                          cudaMemcpyDeviceToHost,
                          stream.value()) == cudaSuccess);
  stream.synchronize();
  CHECK(result.values->null_count() == 3);
  for (uint32_t row = 0; row < ones.size(); ++row) {
    auto valid = row != 31 && row != 32 && row != 63;
    CHECK(bool((validity[row / 32] >> (row % 32)) & 1) == valid);
    CHECK(errors[row] == 0);
    CHECK(values[row].words[0] == (valid ? 2 : 0));
  }
}

TEST_CASE("MO Decimal256 result codec honors slices and bounded scratch", "[exact_decimal_gpu]")
{
  using namespace sirius::embedding;
  rmm::cuda_stream stream;
  std::vector<coefficient> values{small(17), small(-17), small(9007199254740993LL), small(99)};
  auto column = input_column({256, 15, 2}, values, 0b1011, stream.view());
  auto view   = column->view();
  cudf::column_view sliced(view.type(),
                           3,
                           nullptr,
                           view.null_mask(),
                           1,
                           1,
                           {view.child(0), view.child(1), view.child(2), view.child(3)});
  CHECK(decimal_column_matches(sliced, {256, 15, 2}));
  std::vector<owned_column> schema{{34, 15, 2, true, "wide"}};
  auto result = std::make_shared<native_result>();
  result->activate(std::make_shared<pool>());
  std::shared_ptr<result_batch> scratch;
  // Two rows fit the scratch slab: the three-row slice crosses its chunk
  // boundary. The mock storage also splits rows and limbs into 13-byte blocks.
  REQUIRE(result->try_allocate(64, 0, scratch) == SIRIUS_OK);
  for (uint32_t begin : {0, 1}) {
    auto layout =
      size_native_result(cudf::table_view({sliced}), begin, schema, *scratch, stream.view(), 4096);
    REQUIRE(layout.rows == 3 - begin);
    std::shared_ptr<result_batch> batch;
    REQUIRE(result->try_allocate(layout.bytes, 1, batch) == SIRIUS_OK);
    encode_native_result(
      cudf::table_view({sliced}), begin, schema, layout, *batch, *scratch, stream.view());
    std::array<uint8_t, 96> encoded{};
    batch->storage->read(batch->columns[0].data_offset,
                         {reinterpret_cast<std::byte*>(encoded.data()), layout.rows * 32});
    for (uint32_t row = 0; row < layout.rows; ++row) {
      auto observed = load_coefficient(encoded.data() + row * 32, 32);
      auto expected = row + begin == 1 ? coefficient{} : values[row + begin + 1];
      for (int word = 0; word < 8; ++word)
        CHECK(observed.words[word] == expected.words[word]);
    }
    uint64_t nulls{};
    batch->storage->read(batch->columns[0].null_offset,
                         {reinterpret_cast<std::byte*>(&nulls), sizeof(nulls)});
    CHECK(nulls == (begin == 0 ? 2 : 1));
  }
  result->cancel();
}

TEST_CASE("MO full-width GPU arithmetic matches independent wide rational vectors",
          "[exact_decimal_gpu]")
{
  rmm::cuda_stream stream;
  bool enabled[]{true, false};
  rmm::device_buffer active(enabled, sizeof(enabled), stream.view());
  stream.synchronize();
  for (auto const& test : wide_cases) {
    INFO("op=" << static_cast<int>(test.op) << " a=" << test.a << " b=" << test.b);
    auto av = decimal_fixture(test.a), bv = decimal_fixture(test.b);
    auto a        = input_column({256, 76, test.sa}, {av, av}, 3, stream.view());
    auto b        = input_column({256, 76, test.sb}, {bv, bv}, 3, stream.view());
    auto result   = evaluate_decimal_columns(test.op,
                                           a->view(),
                                             {256, 76, test.sa},
                                           b->view(),
                                             {256, 76, test.sb},
                                             {256, 65, test.so},
                                           static_cast<bool const*>(active.data()),
                                           stream.view(),
                                           cudf::get_current_device_resource_ref());
    auto observed = download(result.values->view(), 32, stream.view());
    uint8_t errors[2]{};
    uint32_t validity{};
    REQUIRE(
      cudaMemcpyAsync(errors, result.errors.data(), 2, cudaMemcpyDeviceToHost, stream.value()) ==
      cudaSuccess);
    REQUIRE(
      cudaMemcpyAsync(
        &validity, result.values->view().null_mask(), 4, cudaMemcpyDeviceToHost, stream.value()) ==
      cudaSuccess);
    stream.synchronize();
    CHECK(errors[0] ==
          static_cast<uint8_t>(test.overflow ? decimal_error::out_of_range : decimal_error::none));
    CHECK(bool(validity & 1) == !test.overflow);
    CHECK(errors[1] == 0);
    CHECK((validity & 2) == 0);
    if (!test.overflow) {
      auto expected = decimal_fixture(test.expected);
      for (int i = 0; i < 8; ++i)
        CHECK(observed[0].words[i] == expected.words[i]);
    }
  }
}

TEST_CASE("MO proven 64-to-128 cuDF paths cover the full signed physical input domain",
          "[exact_decimal_gpu]")
{
  rmm::cuda_stream stream;
  std::vector<coefficient> av{small(INT64_MAX), small(INT64_MIN), small(INT64_MAX)};
  std::vector<coefficient> bv{small(INT64_MAX), small(-1), small(INT64_MIN)};
  auto a = input_column({64, 18, 0}, av, 7, stream.view());
  auto b = input_column({64, 18, 0}, bv, 7, stream.view());
  for (auto op : {decimal_op::add, decimal_op::subtract, decimal_op::multiply}) {
    auto result   = evaluate_decimal_columns(op,
                                           a->view(),
                                             {64, 18, 0},
                                           b->view(),
                                             {64, 18, 0},
                                             {128, 38, 0},
                                           nullptr,
                                           stream.view(),
                                           cudf::get_current_device_resource_ref());
    auto observed = download(result.values->view(), 16, stream.view());
    uint8_t errors[3]{};
    REQUIRE(
      cudaMemcpyAsync(errors, result.errors.data(), 3, cudaMemcpyDeviceToHost, stream.value()) ==
      cudaSuccess);
    stream.synchronize();
    for (int row = 0; row < 3; ++row) {
      int64_t left[]  = {INT64_MAX, INT64_MIN, INT64_MAX};
      int64_t right[] = {INT64_MAX, -1, INT64_MIN};
      __int128 lhs = left[row], rhs = right[row];
      __int128 value = op == decimal_op::add        ? lhs + rhs
                       : op == decimal_op::subtract ? lhs - rhs
                                                    : lhs * rhs;
      auto expected  = load_coefficient(reinterpret_cast<uint8_t const*>(&value), 16);
      CHECK(errors[row] == 0);
      for (int word = 0; word < 8; ++word) {
        CHECK(observed[row].words[word] == expected.words[word]);
      }
    }
  }
}
