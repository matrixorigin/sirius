/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "data/convertible_data_batch.hpp"
#include "numeric/decimal_aggregate_gpu.hpp"
#include "numeric/decimal_aggregate_layout.hpp"
#include "numeric/decimal_error.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/hash_join.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/callback_memory_resource.hpp>

#include <catch.hpp>

#include <array>

using namespace sirius;
using namespace sirius::mo_decimal;
namespace {
coefficient small(int64_t value)
{
  return load_coefficient(reinterpret_cast<uint8_t const*>(&value), 8);
}
coefficient read(cudf::column_view column, uint32_t bytes, rmm::cuda_stream_view stream)
{
  coefficient result;
  if (bytes == 32) {
    for (int i = 0; i < 4; ++i) {
      uint64_t word = 0;
      REQUIRE(cudaMemcpyAsync(&word,
                              column.child(3 - i).head<uint64_t>() + column.child(3 - i).offset() +
                                column.offset(),
                              8,
                              cudaMemcpyDeviceToHost,
                              stream.value()) == cudaSuccess);
      stream.synchronize();
      result.words[2 * i]     = word;
      result.words[2 * i + 1] = word >> 32;
    }
  } else {
    uint8_t data[16]{};
    REQUIRE(cudaMemcpyAsync(data,
                            column.head<uint8_t>() + uint64_t(column.offset()) * bytes,
                            bytes,
                            cudaMemcpyDeviceToHost,
                            stream.value()) == cudaSuccess);
    stream.synchronize();
    result = load_coefficient(data, bytes);
  }
  return result;
}
void expect(cudf::column_view column,
            decimal_type type,
            int64_t value,
            rmm::cuda_stream_view stream)
{
  REQUIRE(column.size() == 1);
  REQUIRE(column.null_count() == 0);
  CHECK(decimal_column_matches(column, type));
  auto observed = read(column, type.bytes(), stream), expected = small(value);
  for (int i = 0; i < 8; ++i)
    CHECK(observed.words[i] == expected.words[i]);
}
std::shared_ptr<aggregate_layout const> layout(aggregate_op op,
                                               logical_type input,
                                               logical_type output,
                                               bool grouped = false,
                                               bool count   = false)
{
  duckdb::vector<std::unique_ptr<ast::node>> groups, aggregates;
  if (grouped)
    groups.push_back(
      std::make_unique<ast::node>(ast::reference{1, logical_type::make(type_id::INTEGER)}));
  std::vector<std::unique_ptr<ast::node>> args;
  args.push_back(std::make_unique<ast::node>(ast::reference{0, input}));
  aggregates.push_back(std::make_unique<ast::node>(
    ast::aggregate{aggregate_function(op), std::move(args), output, false}));
  if (count)
    aggregates.push_back(std::make_unique<ast::node>(
      ast::aggregate{aggregate_id::count_star, {}, logical_type::make(type_id::BIGINT), false}));
  return make_aggregate_layout(groups, aggregates);
}
}  // namespace
TEST_CASE("MO GPU aggregate local and merge retain all widths and ordinary slots",
          "[decimal_aggregate_gpu]")
{
  rmm::cuda_stream stream;
  auto mr            = cudf::get_current_device_resource_ref();
  constexpr int rows = 10003;
  for (uint16_t bits : {64, 128, 256})
    for (auto op : {aggregate_op::sum, aggregate_op::avg, aggregate_op::min, aggregate_op::max}) {
      INFO("bits=" << bits << " op=" << static_cast<int>(op));
      decimal_type input{bits, 15, 2}, output = input;
      if (op == aggregate_op::sum) output = {static_cast<uint16_t>(bits == 64 ? 128 : 256), 37, 2};
      if (op == aggregate_op::avg) output = {static_cast<uint16_t>(bits == 64 ? 128 : 256), 19, 6};
      auto definition = layout(op,
                               logical_type::make_mo_decimal(input, false),
                               logical_type::make_mo_decimal(output, true),
                               false,
                               true);
      auto values     = make_decimal_literal(input, small(125), true, rows, stream, mr);
      auto partial =
        local_aggregate_table(cudf::table_view{{values->view()}}, *definition, stream, mr);
      REQUIRE(partial->num_rows() == 1);
      REQUIRE(partial->num_columns() == 3);
      auto result =
        merge_aggregate_tables({partial->view(), partial->view()}, *definition, stream, mr);
      expect(result->view().column(0),
             output,
             op == aggregate_op::sum   ? 125 * rows * 2
             : op == aggregate_op::avg ? 1250000
                                       : 125,
             stream);
      auto count = read(result->view().column(1), 8, stream);
      CHECK(count.words[0] == rows * 2);
      auto single = merge_aggregate_tables({partial->view()}, *definition, stream, mr);
      expect(single->view().column(0),
             output,
             op == aggregate_op::sum   ? 125 * rows
             : op == aggregate_op::avg ? 1250000
                                       : 125,
             stream);
    }
}
TEST_CASE("MO GPU grouped states preserve common ordinals and NULL identities",
          "[decimal_aggregate_gpu]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  decimal_type type{256, 15, 2}, output{256, 37, 2};
  for (int rows : {0, 3})
    for (bool valid : {false, true}) {
      auto values = make_decimal_literal(type, small(125), valid, rows, stream, mr);
      auto groups = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32}, rows, cudf::mask_state::UNALLOCATED, stream, mr);
      int32_t labels[]{1, 0, 1};
      if (rows)
        REQUIRE(cudaMemcpyAsync(groups->mutable_view().data<int32_t>(),
                                labels,
                                sizeof(labels),
                                cudaMemcpyHostToDevice,
                                stream.value()) == cudaSuccess);
      stream.synchronize();
      auto definition = layout(aggregate_op::sum,
                               logical_type::make_mo_decimal(type, true),
                               logical_type::make_mo_decimal(output, true),
                               true,
                               true);
      auto partial    = local_aggregate_table(
        cudf::table_view{{values->view(), groups->view()}}, *definition, stream, mr);
      auto result =
        merge_aggregate_tables({partial->view(), partial->view()}, *definition, stream, mr);
      CHECK(result->num_rows() == (rows ? 2 : 0));
      if (rows) {
        CHECK(result->view().column(1).null_count() == (valid ? 0 : 2));
        auto count = read(result->view().column(2), 8, stream);
        CHECK(count.words[0] == 2);
        if (valid) {
          auto first = read(result->view().column(1), 32, stream);
          CHECK(first.words[0] == 250);
        }
      }
      auto ungrouped = layout(aggregate_op::avg,
                              logical_type::make_mo_decimal(type, true),
                              logical_type::make_mo_decimal({256, 19, 6}, true));
      auto empty_partial =
        local_aggregate_table(cudf::table_view{{values->view()}}, *ungrouped, stream, mr);
      auto empty_result = merge_aggregate_tables({empty_partial->view()}, *ungrouped, stream, mr);
      CHECK(empty_result->num_rows() == 1);
      CHECK(empty_result->view().column(0).null_count() == (!rows || !valid ? 1 : 0));
    }
}
TEST_CASE("MO GPU SUM checks final precision after partial cancellation", "[decimal_aggregate_gpu]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  decimal_type type{256, 65, 0};
  auto definition = layout(aggregate_op::sum,
                           logical_type::make_mo_decimal(type, false),
                           logical_type::make_mo_decimal(type, true));
  auto magnitude  = decimal_detail::power10(65);
  decimal_detail::magnitude one;
  one.words[0] = 1;
  decimal_detail::subtract(magnitude, one);
  std::vector<std::unique_ptr<cudf::table>> partials;
  for (bool negative : {false, false, true, true}) {
    auto coefficient =
      decimal_detail::finish(magnitude, negative, type, decimal_error::invalid_input).value;
    auto values = make_decimal_literal(type, coefficient, true, 1, stream, mr);
    partials.push_back(
      local_aggregate_table(cudf::table_view{{values->view()}}, *definition, stream, mr));
  }
  auto result = merge_aggregate_tables(
    {partials[0]->view(), partials[1]->view(), partials[2]->view(), partials[3]->view()},
    *definition,
    stream,
    mr);
  expect(result->view().column(0), type, 0, stream);
  auto one_value = make_decimal_literal(type, small(1), true, 1, stream, mr);
  auto one_partial =
    local_aggregate_table(cudf::table_view{{one_value->view()}}, *definition, stream, mr);
  try {
    merge_aggregate_tables({partials[0]->view(), one_partial->view()}, *definition, stream, mr);
    FAIL("final SUM precision overflow was ignored");
  } catch (numeric_error const& error) {
    CHECK(error.code() == decimal_error::invalid_input);
  }
  auto recovered =
    merge_aggregate_tables({partials[0]->view(), partials[2]->view()}, *definition, stream, mr);
  expect(recovered->view().column(0), type, 0, stream);
}
TEST_CASE("MO segmented reductions retain boundaries and nonzero subranges",
          "[decimal_aggregate_gpu]")
{
  rmm::cuda_stream stream;
  auto mr     = cudf::get_current_device_resource_ref();
  auto values = make_decimal_literal({64, 15, 0}, small(1), true, 8202, stream, mr);
  for (auto host : {std::array<cudf::size_type, 4>{0, 4095, 4096, 8197},
                    std::array<cudf::size_type, 4>{4093, 4095, 4096, 8197}}) {
    rmm::device_uvector<cudf::size_type> offsets(host.size(), stream, mr);
    REQUIRE(cudaMemcpyAsync(
              offsets.data(), host.data(), sizeof(host), cudaMemcpyHostToDevice, stream.value()) ==
            cudaSuccess);
    stream.synchronize();
    auto state = reduce_aggregate_columns(
      aggregate_op::sum, values->view(), {64, 15, 0}, 128, {}, {}, offsets.data(), 3, stream, mr);
    auto result = finalize_aggregate_columns(aggregate_op::sum,
                                             state.values->view(),
                                             state.counts->view(),
                                             {64, 15, 0},
                                             {128, 37, 0},
                                             stream,
                                             mr);
    for (int group = 0; group < 3; ++group) {
      auto row = cudf::slice(result->view(), {group, group + 1}, stream);
      expect(row[0], {128, 37, 0}, host[group + 1] - host[group], stream);
      auto count = cudf::slice(state.counts->view(), {group, group + 1}, stream);
      CHECK(read(count[0], 8, stream).words[0] == host[group + 1] - host[group]);
    }
  }
}
TEST_CASE("MO GPU equality keys match across widths and scales", "[decimal_aggregate_gpu]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  for (int sign : {-1, 0, 1}) {
    auto narrow = make_decimal_literal({64, 18, 2}, small(sign * 120), true, 2, stream, mr);
    auto wide   = make_decimal_literal({256, 65, 3}, small(sign * 1200), true, 1, stream, mr);
    auto a      = make_equality_key(narrow->view(), {64, 18, 2}, stream, mr);
    auto b      = make_equality_key(wide->view(), {256, 65, 3}, stream, mr);
    cudf::hash_join join(cudf::table_view{{b->view()}}, cudf::null_equality::UNEQUAL, stream);
    auto indices = join.inner_join(cudf::table_view{{a->view()}}, std::nullopt, stream, mr);
    stream.synchronize();
    CHECK(indices.first->size() == 2);
    CHECK(indices.second->size() == 2);
  }
}
TEST_CASE("MO Decimal256 validity restoration retains outer gather NULLs",
          "[decimal_aggregate_gpu]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  decimal_type type{256, 15, 2};
  auto value   = make_decimal_literal(type, small(125), true, 1, stream, mr);
  auto indices = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 2, cudf::mask_state::UNALLOCATED, stream, mr);
  int32_t host[]{0, 1};
  REQUIRE(cudaMemcpyAsync(indices->mutable_view().data<int32_t>(),
                          host,
                          sizeof(host),
                          cudaMemcpyHostToDevice,
                          stream.value()) == cudaSuccess);
  stream.synchronize();
  auto gathered = cudf::gather(cudf::table_view{{value->view()}},
                               indices->view(),
                               cudf::out_of_bounds_policy::NULLIFY,
                               stream,
                               mr)
                    ->release();
  auto restored = restore_decimal_validity(std::move(gathered[0]), type, stream, mr);
  CHECK(decimal_column_matches(restored->view(), type));
  CHECK(restored->null_count() == 1);
  auto zero   = make_decimal_literal(type, {}, true, 2, stream, mr);
  auto result = evaluate_decimal_columns(
    decimal_op::add, restored->view(), type, zero->view(), type, type, nullptr, stream, mr);
  CHECK(column_error(result, stream, mr) == decimal_error::none);
  CHECK(result.values->null_count() == 1);
}
TEST_CASE("MO aggregate allocation cutpoints retire submitted buffers", "[decimal_aggregate_gpu]")
{
  rmm::cuda_stream stream;
  auto upstream = cudf::get_current_device_resource_ref();
  for (bool wide : {false, true})
    for (int groups : {1, 2}) {
      decimal_type type = wide ? decimal_type{256, 65, 2} : decimal_type{64, 15, 2};
      auto values       = make_decimal_literal(type, small(125), true, 3, stream, upstream);
      rmm::device_uvector<cudf::size_type> offsets(groups + 1, stream, upstream);
      cudf::size_type host[]{0, groups == 1 ? 3 : 1, 3};
      REQUIRE(cudaMemcpyAsync(offsets.data(),
                              host,
                              (groups + 1) * sizeof(*host),
                              cudaMemcpyHostToDevice,
                              stream.value()) == cudaSuccess);
      stream.synchronize();
      int failures = 0, successes = 0;
      for (int cutpoint = 0; cutpoint < 24; ++cutpoint) {
        size_t live   = 0;
        int remaining = cutpoint;
        bool bad_free = false;
        rmm::mr::callback_memory_resource resource(
          [&](size_t bytes, rmm::cuda_stream_view stream, void*) {
            if (!remaining--) throw std::bad_alloc();
            auto p = upstream.allocate(stream, bytes, 256);
            live += bytes;
            return p;
          },
          [&](void* p, size_t bytes, rmm::cuda_stream_view stream, void*) {
            if (bytes > live)
              bad_free = true;
            else
              live -= bytes;
            upstream.deallocate(stream, p, bytes, 256);
          });
        auto mr = rmm::to_device_async_resource_ref_checked(&resource);
        try {
          auto result = reduce_aggregate_columns(aggregate_op::sum,
                                                 values->view(),
                                                 type,
                                                 wide ? 256 : 128,
                                                 {},
                                                 {},
                                                 offsets.data(),
                                                 groups,
                                                 stream,
                                                 mr);
          REQUIRE(result.counts->size() == groups);
          ++successes;
        } catch (std::bad_alloc const&) {
          ++failures;
        }
        stream.synchronize();
        CHECK(live == 0);
        CHECK_FALSE(bad_free);
      }
      CHECK(failures > 0);
      CHECK(successes > 0);
    }
}
TEST_CASE("MO proven local reduction contains the full signed 64-bit domain",
          "[decimal_aggregate_gpu]")
{
  rmm::cuda_stream stream;
  auto mr     = cudf::get_current_device_resource_ref();
  auto values = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, 3, cudf::mask_state::UNALLOCATED, stream, mr);
  int64_t endpoints[]{INT64_MAX, INT64_MAX, INT64_MIN};
  REQUIRE(cudaMemcpyAsync(values->mutable_view().data<int64_t>(),
                          endpoints,
                          sizeof(endpoints),
                          cudaMemcpyHostToDevice,
                          stream.value()) == cudaSuccess);
  stream.synchronize();
  auto decimal = cudf::column_view(
    cudf::data_type{cudf::type_id::DECIMAL64, 0}, 3, values->view().head<uint8_t>(), nullptr, 0);
  rmm::device_uvector<cudf::size_type> offsets(2, stream, mr);
  cudf::size_type host[]{0, 3};
  REQUIRE(
    cudaMemcpyAsync(offsets.data(), host, sizeof(host), cudaMemcpyHostToDevice, stream.value()) ==
    cudaSuccess);
  stream.synchronize();
  for (uint16_t bits : {128, 256}) {
    auto state = reduce_aggregate_columns(
      aggregate_op::sum, decimal, {64, 18, 0}, bits, {}, {}, offsets.data(), 1, stream, mr);
    decimal_type output{bits, static_cast<uint8_t>(bits == 128 ? 38 : 65), 0};
    auto result = finalize_aggregate_columns(aggregate_op::sum,
                                             state.values->view(),
                                             state.counts->view(),
                                             {64, 18, 0},
                                             output,
                                             stream,
                                             mr);
    expect(result->view(), output, INT64_MAX - 1, stream);
    auto nulls = make_decimal_literal({64, 15, 2}, {}, false, 3, stream, mr);
    auto empty = reduce_aggregate_columns(
      aggregate_op::sum, nulls->view(), {64, 15, 2}, bits, {}, {}, offsets.data(), 1, stream, mr);
    output.scale = 2;
    auto absent  = finalize_aggregate_columns(aggregate_op::sum,
                                             empty.values->view(),
                                             empty.counts->view(),
                                              {64, 15, 2},
                                             output,
                                             stream,
                                             mr);
    CHECK(absent->null_count() == 1);
  }
}
TEST_CASE("MO local widening contains the full signed 128-bit domain", "[decimal_aggregate_gpu]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  coefficient maximum, minimum;
  for (int i = 0; i < 4; ++i)
    maximum.words[i] = UINT32_MAX;
  maximum.words[3] >>= 1;
  minimum.words[3] = uint32_t(1) << 31;
  for (int i = 4; i < 8; ++i)
    minimum.words[i] = UINT32_MAX;
  std::array<uint8_t, 48> bytes{};
  store_coefficient(maximum, bytes.data(), 16);
  store_coefficient(maximum, bytes.data() + 16, 16);
  store_coefficient(minimum, bytes.data() + 32, 16);
  auto values = make_decimal_column({128, 38, 0}, 3, cudf::mask_state::UNALLOCATED, stream, mr);
  REQUIRE(cudaMemcpyAsync(values->mutable_view().data<uint8_t>(),
                          bytes.data(),
                          bytes.size(),
                          cudaMemcpyHostToDevice,
                          stream.value()) == cudaSuccess);
  rmm::device_uvector<cudf::size_type> offsets(2, stream, mr);
  cudf::size_type host[]{0, 3};
  REQUIRE(
    cudaMemcpyAsync(offsets.data(), host, sizeof(host), cudaMemcpyHostToDevice, stream.value()) ==
    cudaSuccess);
  stream.synchronize();
  auto state = reduce_aggregate_columns(
    aggregate_op::sum, values->view(), {128, 38, 0}, 256, {}, {}, offsets.data(), 1, stream, mr);
  auto result   = finalize_aggregate_columns(aggregate_op::sum,
                                           state.values->view(),
                                           state.counts->view(),
                                             {128, 38, 0},
                                             {256, 65, 0},
                                           stream,
                                           mr);
  auto observed = read(result->view(), 32, stream);
  maximum.words[0] -= 1;
  for (int i = 0; i < 8; ++i)
    CHECK(observed.words[i] == maximum.words[i]);
}
TEST_CASE("MO full-width aggregate states survive existing host spill and restore",
          "[decimal_aggregate_gpu]")
{
  auto manager = test::operator_utils::initialize_memory_manager();
  auto* gpu    = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto* host   = manager->get_memory_space(cucascade::memory::Tier::HOST, 0);
  REQUIRE(gpu);
  REQUIRE(host);
  rmm::cuda_stream stream;
  auto mr = gpu->get_default_allocator();
  decimal_type type{256, 65, 0};
  auto definition = layout(aggregate_op::sum,
                           logical_type::make_mo_decimal(type, false),
                           logical_type::make_mo_decimal(type, true));
  auto value      = make_decimal_literal(type, small(125), true, 3, stream, mr);
  auto partial = local_aggregate_table(cudf::table_view{{value->view()}}, *definition, stream, mr);
  auto batch   = make_data_batch(std::move(partial), *gpu, stream, {});
  convertible_data_batch wrapper(batch);
  REQUIRE(wrapper.convert({host}, stream, *manager, true).has_value());
  stream.synchronize();
  {
    auto ro = batch->to_read_only();
    CHECK(ro.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }
  REQUIRE(wrapper.convert({gpu}, stream, *manager, true).has_value());
  stream.synchronize();
  auto ro       = batch->to_read_only();
  auto restored = get_cudf_table_view(ro);
  auto result   = merge_aggregate_tables({restored}, *definition, stream, mr);
  expect(result->view().column(0), type, 375, stream);
}
