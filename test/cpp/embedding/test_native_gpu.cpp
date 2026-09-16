/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
// Private test backend: public C entry points + real native source, scheduler,
// pooled staging and CUDA conversion. No production fake-plan syntax/capability.
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "embedding/native_gpu.hpp"
#include "helper/type_conversions.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"
#include "sirius_engine.hpp"
#include "sirius_interface.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>

#include <atomic>
#include <cstring>
#include <future>
#include <thread>

using namespace sirius::embedding;
using namespace std::chrono_literals;
namespace {
std::atomic<uint64_t> observed_rows{0}, observed_sum{0};
std::atomic<bool> invalid_values{false};
auto test_types()
{
  return sirius::from_duckdb_vec(
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT, duckdb::LogicalType::VARCHAR});
}
class checking_sink final : public sirius::op::sirius_physical_operator {
 public:
  checking_sink()
    : sirius_physical_operator(
        sirius::op::SiriusPhysicalOperatorType::STREAMING_SINK, test_types(), 0)
  {
  }
  bool is_sink() const override { return true; }
  std::unique_ptr<sirius::op::operator_data> execute(sirius::op::operator_data const& data,
                                                     rmm::cuda_stream_view) override
  {
    return std::make_unique<sirius::op::pipelineable_operator_data>(
      dynamic_cast<sirius::op::pipelineable_operator_data const&>(data).get_read_only_batches());
  }
  void sink(sirius::op::operator_data const& data, rmm::cuda_stream_view stream) override
  {
    for (auto const& batch : dynamic_cast<sirius::op::pipelineable_operator_data const&>(data)
                               .get_read_only_batches()) {
      auto table = sirius::get_cudf_table_view(batch);
      std::vector<int64_t> numbers(table.num_rows());
      std::vector<int32_t> offsets(table.num_rows() + 1);
      auto strings = cudf::strings_column_view(table.column(1));
      std::vector<char> chars(strings.chars_size(stream));
      std::vector<uint32_t> mask((table.num_rows() + 31) / 32);
      if (!numbers.empty()) {
        cudaMemcpyAsync(numbers.data(),
                        table.column(0).data<int64_t>(),
                        numbers.size() * 8,
                        cudaMemcpyDeviceToHost,
                        stream.value());
        cudaMemcpyAsync(offsets.data(),
                        strings.offsets().data<int32_t>(),
                        offsets.size() * 4,
                        cudaMemcpyDeviceToHost,
                        stream.value());
        if (!chars.empty())
          cudaMemcpyAsync(chars.data(),
                          strings.chars_begin(stream),
                          chars.size(),
                          cudaMemcpyDeviceToHost,
                          stream.value());
        cudaMemcpyAsync(mask.data(),
                        strings.null_mask(),
                        mask.size() * 4,
                        cudaMemcpyDeviceToHost,
                        stream.value());
      }
      stream.synchronize();
      for (std::size_t r = 0; r < numbers.size(); ++r) {
        auto n    = numbers[r];
        auto null = n % 7 == 0;
        if (((mask[r / 32] >> (r % 32)) & 1u) == null) invalid_values = true;
        if (!null) {
          auto expected = std::to_string(n);
          if (offsets[r + 1] - offsets[r] != static_cast<int32_t>(expected.size()) ||
              std::string_view(chars.data() + offsets[r], offsets[r + 1] - offsets[r]) != expected)
            invalid_values = true;
        }
        observed_sum.fetch_add(n);
      }
      observed_rows.fetch_add(numbers.size());
    }
  }
  void build_pipelines(sirius::pipeline::sirius_pipeline& current,
                       sirius::pipeline::sirius_meta_pipeline& meta) override
  {
    meta.get_state().add_pipeline_operator(current, *this);
    for (auto& child : children)
      meta.create_child_meta_pipeline(current, *this).build(*child);
  }
};
class gpu_driver final : public query_driver {
 public:
  gpu_driver(duckdb::SiriusContext& context, duckdb::Connection& connection, input_registry& inputs)
    : inputs_(inputs),
      scope_(context, *connection.context, "native_input_test"),
      iface_(*connection.context, std::optional<std::string>("native_input_test")),
      engine_(*connection.context, iface_, scope_.query_id())
  {
    auto sink = duckdb::make_uniq<checking_sink>();
    auto left = duckdb::make_uniq<sirius::op::scan::sirius_gpu_scan_operator>(
      test_types(), 10000, make_native_ingestible(inputs.reads.at(0)));
    auto right = duckdb::make_uniq<sirius::op::scan::sirius_gpu_scan_operator>(
      test_types(), 10000, make_native_ingestible(inputs.reads.at(1)));
    duckdb::LogicalComparisonJoin logical(duckdb::JoinType::INNER);
    logical.types = {duckdb::LogicalType::BIGINT,
                     duckdb::LogicalType::VARCHAR,
                     duckdb::LogicalType::BIGINT,
                     duckdb::LogicalType::VARCHAR};
    duckdb::vector<duckdb::JoinCondition> conditions;
    duckdb::JoinCondition condition;
    condition.left =
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0);
    condition.right =
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0);
    condition.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
    conditions.push_back(std::move(condition));
    sink->types = sirius::from_duckdb_vec(logical.types);
    sink->children.push_back(duckdb::make_uniq<sirius::op::sirius_physical_hash_join>(
      logical,
      std::move(left),
      std::move(right),
      sirius::wrap_join_conditions(std::move(conditions)),
      duckdb::JoinType::INNER,
      duckdb::vector<std::size_t>{},
      duckdb::vector<std::size_t>{},
      duckdb::vector<sirius::logical_type>{},
      10000));
    duckdb::unique_ptr<sirius::op::sirius_physical_operator> root = std::move(sink);
    sirius::planner::sirius_physical_plan_generator generator(*connection.context);
    generator.insert_gpu_pipeline_operators(root);
    engine_.initialize(std::move(root));
  }
  void run(std::stop_token, clock::time_point deadline) override
  {
    std::jthread timer([&](std::stop_token stop) {
      std::mutex mutex;
      std::condition_variable_any cv;
      std::unique_lock lock(mutex);
      cv.wait_until(lock, stop, deadline, [] { return false; });
      if (!stop.stop_requested())
        for (auto const& input : inputs_.reads)
          input->stop(SIRIUS_TIMEOUT, "test query deadline");
    });
    engine_.execute();
  }
  void finish() override { scope_.finish(); }

 private:
  input_registry& inputs_;
  duckdb::SiriusContext::StandaloneQueryScope scope_;
  sirius::sirius_interface iface_;
  sirius::sirius_engine engine_;
};
class gpu_backend final : public engine_backend {
 public:
  gpu_backend(std::string const& path, uint32_t streams)
  {
    sirius::sirius_config config;
    config.load_from_file(path);
    config.set_gpu_pipeline_executor_threads(streams);
    context_ = duckdb::make_shared_ptr<duckdb::SiriusContext>();
    context_->initialize(config);
    sirius::converter_registry::initialize();
    duckdb::DBConfig db_config;
    db_config.options.load_extensions = false;
    db_                               = duckdb::make_uniq<duckdb::DuckDB>(nullptr, &db_config);
    connection_                       = duckdb::make_uniq<duckdb::Connection>(*db_);
    connection_->context->registered_state->Insert("sirius_state", context_);
    connection_->context->registered_state->Insert(
      "sirius_connection_state", duckdb::make_shared_ptr<duckdb::SiriusConnectionState>());
  }
  std::unique_ptr<query_driver> prepare(std::string_view,
                                        std::stop_token,
                                        clock::time_point) override
  {
    throw failure(SIRIUS_UNSUPPORTED, "test requires native inputs");
  }
  std::unique_ptr<query_driver> prepare_inputs(std::string_view,
                                               input_registry& inputs,
                                               std::stop_token,
                                               clock::time_point) override
  {
    auto hosts =
      context_->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    if (hosts.empty()) throw std::runtime_error("missing host pool");
    activate_native_inputs(inputs, *const_cast<cucascade::memory::memory_space*>(hosts.front()));
    return std::make_unique<gpu_driver>(*context_, *connection_, inputs);
  }

 private:
  duckdb::shared_ptr<duckdb::SiriusContext> context_;
  duckdb::unique_ptr<duckdb::DuckDB> db_;
  duckdb::unique_ptr<duckdb::Connection> connection_;
};
struct handles {
  sirius_engine_handle* engine{};
  sirius_query_handle* query{};
  sirius_input_handle* inputs[2]{};
  ~handles()
  {
    sirius_query_cancel(query, nullptr);
    for (auto& input : inputs)
      sirius_input_close(&input, nullptr);
    sirius_query_close(&query, 30000, nullptr);
    sirius_engine_close(&engine, 30000, nullptr);
  }
};
void ok(sirius_status code, sirius_error const& error)
{
  if (code != SIRIUS_OK) throw std::runtime_error(error.message);
}
void produce(sirius_input_handle* input, int64_t base, int batches)
{
  sirius_error error{};
  for (int b = 0; b < batches; ++b) {
    uint32_t rows = b % 2 ? 65 : 33;
    std::vector<std::byte> data(rows * 32 + ((rows + 63) / 64) * 8);
    for (uint32_t r = 0; r < rows; ++r) {
      int64_t n = base + b * 100 + r;
      std::memcpy(data.data() + r * 8, &n, 8);
      auto* v = data.data() + rows * 8 + r * 24;
      if (n % 7 == 0) {
        std::memset(v, 0xff, 24);
        data[rows * 32 + r / 8] |= std::byte(1u << (r % 8));
      } else {
        auto s = std::to_string(n);
        v[0]   = std::byte(s.size());
        std::memcpy(v + 1, s.data(), s.size());
      }
    }
    sirius_batch_handle* batch = nullptr;
    ok(sirius_input_acquire(input, data.size(), 10000, &batch, &error), error);
    try {
      ok(sirius_input_write(batch, 0, data.data(), data.size(), &error), error);
      sirius_input_vector columns[2]{};
      columns[0].data_bytes  = rows * 8;
      columns[1].data_offset = rows * 8;
      columns[1].data_bytes  = rows * 24;
      columns[1].null_offset = rows * 32;
      columns[1].null_bytes  = ((rows + 63) / 64) * 8;
      ok(sirius_input_publish(input, &batch, rows, columns, 2, &error), error);
    } catch (...) {
      sirius_batch_release(&batch, nullptr);
      throw;
    }
  }
  ok(sirius_input_finish(input, &error), error);
}
}  // namespace
namespace sirius::embedding {
engine_control::factory native_backend_factory(std::string path, uint32_t streams)
{
  return [path = std::move(path), streams] { return std::make_unique<gpu_backend>(path, streams); };
}
}  // namespace sirius::embedding
TEST_CASE("native C producers execute on the real GPU scheduler", "[native_gpu]")
{
  auto streams = GENERATE(1u, 2u, 4u);
  handles h;
  sirius_error error{};
  auto path = std::string(SIRIUS_PROJECT_ROOT) + "/test/cpp/scan/memory.yaml";
  sirius_engine_options options{sizeof(options),
                                SIRIUS_ABI_VERSION,
                                path.data(),
                                static_cast<uint32_t>(path.size()),
                                0,
                                streams,
                                0};
  ok(sirius_engine_create(&options, &h.engine, &error), error);
  sirius_query_options query_options{sizeof(query_options), SIRIUS_ABI_VERSION, 30000, 0};
  ok(sirius_query_create(h.engine, &query_options, "private test", 12, &h.query, &error), error);
  sirius_input_column schema[2]{{23, 0, 0, 0}, {61, 0, 0, 1}};
  for (int i = 0; i < 2; ++i)
    ok(sirius_input_register(h.query, i, schema, 2, &h.inputs[i], &error), error);
  ok(sirius_query_prepare(h.query, 10000, &error), error);
  observed_rows  = 0;
  observed_sum   = 0;
  invalid_values = false;
  ok(sirius_query_start(h.query, &error), error);
  auto a = std::async(std::launch::async, [&] { produce(h.inputs[0], 10000, 64); });
  auto b = std::async(std::launch::async, [&] { produce(h.inputs[1], 10000, 64); });
  a.get();
  b.get();
  auto status = sirius_query_wait(h.query, 30000, &error);
  INFO(error.message);
  REQUIRE(status == SIRIUS_OK);
  uint64_t expected_rows = 0, expected_sum = 0;
  for (int64_t base : {10000})
    for (int batch = 0; batch < 64; ++batch)
      for (int row = 0; row < (batch % 2 ? 65 : 33); ++row) {
        ++expected_rows;
        expected_sum += base + batch * 100 + row;
      }
  CHECK(observed_rows == expected_rows);
  CHECK(observed_sum == expected_sum);
  CHECK_FALSE(invalid_values);
  for (auto input : h.inputs) {
    sirius_input_stats stats{sizeof(stats), SIRIUS_ABI_VERSION};
    ok(sirius_input_get_stats(input, &stats, &error), error);
    CHECK(stats.peak_bytes <= input_window);
    CHECK(stats.retained_bytes == 0);
    CHECK(stats.source_units == 0);
  }
}

TEST_CASE("native C capacity cancellation keeps filling leases valid and permits reuse",
          "[native_gpu]")
{
  handles h;
  sirius_error error{};
  auto path = std::string(SIRIUS_PROJECT_ROOT) + "/test/cpp/scan/memory.yaml";
  sirius_engine_options options{
    sizeof(options), SIRIUS_ABI_VERSION, path.data(), static_cast<uint32_t>(path.size()), 0, 2, 0};
  ok(sirius_engine_create(&options, &h.engine, &error), error);
  auto create = [&] {
    sirius_query_options opts{sizeof(opts), SIRIUS_ABI_VERSION, 30000, 0};
    ok(sirius_query_create(h.engine, &opts, "test", 4, &h.query, &error), error);
    sirius_input_column schema[2]{{23, 0, 0, 0}, {61, 0, 0, 1}};
    for (int i = 0; i < 2; ++i)
      ok(sirius_input_register(h.query, i, schema, 2, &h.inputs[i], &error), error);
  };
  create();
  sirius_batch_handle* held = nullptr;
  REQUIRE(sirius_input_acquire(h.inputs[0], 8, 0, &held, &error) == SIRIUS_INVALID_STATE);
  ok(sirius_query_prepare(h.query, 10000, &error), error);
  ok(sirius_input_acquire(h.inputs[0], input_window - (1u << 20), 0, &held, &error), error);
  auto waiting = std::async(std::launch::async, [&] {
    sirius_batch_handle* extra = nullptr;
    sirius_error local{};
    auto code = sirius_input_acquire(h.inputs[0], 2u << 20, 10000, &extra, &local);
    sirius_batch_release(&extra, nullptr);
    return code;
  });
  // Observe the wait itself, rather than a timing-dependent sleep.
  auto limit = clock::now() + 5s;
  sirius_input_stats stats{sizeof(stats), SIRIUS_ABI_VERSION};
  do {
    ok(sirius_input_get_stats(h.inputs[0], &stats, &error), error);
    std::this_thread::yield();
  } while (!stats.blocked_acquires && clock::now() < limit);
  CHECK(stats.blocked_acquires == 1);
  ok(sirius_query_cancel(h.query, &error), error);
  CHECK(waiting.get() == SIRIUS_CANCELLED);
  CHECK(sirius_query_close(&h.query, 10000, &error) == SIRIUS_BUSY);
  uint64_t value = 42;
  CHECK(sirius_input_write(held, 0, &value, 8, &error) == SIRIUS_OK);
  ok(sirius_batch_release(&held, &error), error);
  for (auto& input : h.inputs)
    ok(sirius_input_close(&input, &error), error);
  ok(sirius_query_close(&h.query, 10000, &error), error);
  create();
  ok(sirius_query_prepare(h.query, 10000, &error), error);
  observed_rows = 0;
  for (auto input : h.inputs)
    ok(sirius_input_finish(input, &error), error);
  ok(sirius_query_start(h.query, &error), error);
  auto status = sirius_query_wait(h.query, 10000, &error);
  INFO(error.message);
  REQUIRE(status == SIRIUS_OK);
  CHECK(observed_rows == 0);
  for (auto& input : h.inputs)
    ok(sirius_input_close(&input, &error), error);
  ok(sirius_query_close(&h.query, 10000, &error), error);
  create();
  ok(sirius_query_prepare(h.query, 10000, &error), error);
  ok(sirius_query_start(h.query, &error), error);
  ok(sirius_input_fail(h.inputs[0], "reader failed", 13, &error), error);
  REQUIRE(sirius_query_wait(h.query, 10000, &error) == SIRIUS_EXECUTION_FAILED);
  CHECK(std::string(error.message) == "reader failed");
}

namespace {
struct fail_after_upload final : input_storage {
  std::unique_ptr<input_storage> inner;
  bool fail{true};
  explicit fail_after_upload(std::unique_ptr<input_storage> p) : inner(std::move(p)) {}
  std::size_t size() const override { return inner->size(); }
  void read(std::size_t offset, std::span<std::byte> bytes) const override
  {
    inner->read(offset, bytes);
  }
  void visit(std::function<void(std::size_t, std::span<std::byte>)> const& fn) override
  {
    inner->visit(fn);
    if (fail) throw std::runtime_error("injected failure after asynchronous upload");
  }
};
}  // namespace

TEST_CASE("native GPU scalar decoding preserves bits nulls constants and epochs", "[native_gpu]")
{
  sirius::sirius_config config;
  config.load_from_file(std::string(SIRIUS_PROJECT_ROOT) + "/test/cpp/scan/memory.yaml");
  auto context = duckdb::make_shared_ptr<duckdb::SiriusContext>();
  context->initialize(config);
  auto& manager = context->get_memory_manager();
  auto* host    = const_cast<cucascade::memory::memory_space*>(
    manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).front());
  auto* gpu = manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU).front();
  auto pool = make_native_input_pool(*host);
  rmm::cuda_stream stream;
  for (uint32_t oid : {10, 20, 21, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33, 50, 52}) {
    for (uint32_t cls : {SIRIUS_VECTOR_FLAT, SIRIUS_VECTOR_CONSTANT, SIRIUS_VECTOR_NULL}) {
      INFO("oid=" << oid << " vector_class=" << cls);
      auto width = input_element_size(oid);
      sirius_input_column schema{oid, oid == 32 ? 18 : 38, 2, 1};
      auto input = std::make_shared<native_input>(
        1, std::vector{schema}, std::stop_token{}, clock::now() + 30s);
      input->activate(pool);
      auto physical = cls == SIRIUS_VECTOR_FLAT ? 65u : 1u;
      auto batch    = input->acquire(physical * width + 8, clock::now());
      std::vector<std::byte> raw(physical * width + 8);
      for (uint32_t row = 0; row < physical; ++row) {
        uint64_t lo = oid == 10 ? row % 2 : row + 17, hi = 0x123456789abcdefULL;
        if (oid == 50) lo += 719162;
        if (oid == 52) lo += 62135596800000000ULL;
        std::memcpy(raw.data() + row * width, &lo, std::min<std::size_t>(width, 8));
        if (width == 16) std::memcpy(raw.data() + row * width + 8, &hi, 8);
      }
      if (cls == SIRIUS_VECTOR_FLAT)
        raw[physical * width + 7] = std::byte{128};  // row 63 NULL, omitted row-64 word is zero
      batch->write(0, raw);
      sirius_input_vector vector{};
      vector.vector_class = cls;
      vector.data_bytes   = physical * width;
      vector.null_offset  = physical * width;
      vector.null_bytes   = 8;
      input->publish(batch, 65, {&vector, 1});
      batch.reset();
      auto unit = input->claim();
      REQUIRE(unit);
      // Re-materialize the same immutable source, as an OOM retry would.
      for (int attempt = 0; attempt < 2; ++attempt) {
        auto table = convert_native_input(*unit, input->schema, *gpu, stream.view());
        std::vector<std::byte> values(65 * width);
        uint32_t mask[3]{};
        cudaMemcpyAsync(values.data(),
                        table->view().column(0).head<uint8_t>(),
                        values.size(),
                        cudaMemcpyDeviceToHost,
                        stream.value());
        cudaMemcpyAsync(mask,
                        table->view().column(0).null_mask(),
                        sizeof(mask),
                        cudaMemcpyDeviceToHost,
                        stream.value());
        stream.synchronize();
        for (uint32_t row = 0; row < 65; ++row) {
          bool null = cls == SIRIUS_VECTOR_NULL || (cls == SIRIUS_VECTOR_FLAT && row == 63);
          REQUIRE(bool((mask[row / 32] >> (row % 32)) & 1u) == !null);
          if (null) continue;
          auto source   = cls == SIRIUS_VECTOR_CONSTANT ? 0 : row;
          auto expected = std::vector<std::byte>(raw.begin() + source * width,
                                                 raw.begin() + (source + 1) * width);
          if (oid == 50) {
            uint32_t n = source + 17;
            std::memcpy(expected.data(), &n, 4);
          }
          if (oid == 52) {
            uint64_t n = source + 17;
            std::memcpy(expected.data(), &n, 8);
          }
          REQUIRE(std::memcmp(values.data() + row * width, expected.data(), width) == 0);
        }
      }
      unit.reset();
      input->stop();
      input->discard();
      CHECK(input->inspect().retained_bytes == 0);
    }
  }
  for (uint32_t cls : {SIRIUS_VECTOR_FLAT, SIRIUS_VECTOR_CONSTANT, SIRIUS_VECTOR_NULL}) {
    auto input = std::make_shared<native_input>(
      2, std::vector<sirius_input_column>{{61, 0, 0, 1}}, std::stop_token{}, clock::now() + 30s);
    input->activate(pool);
    std::string text = "external native string longer than the inline varlena payload";
    auto physical    = cls == SIRIUS_VECTOR_FLAT ? 65u : 1u;
    std::vector<std::byte> raw(physical * 24 + 8 + text.size());
    for (uint32_t row = 0; row < physical; ++row) {
      uint32_t descriptor[6]{UINT32_MAX, 0, static_cast<uint32_t>(text.size())};
      std::memcpy(raw.data() + row * 24, descriptor, 24);
    }
    if (cls == SIRIUS_VECTOR_FLAT) raw[physical * 24 + 7] = std::byte{128};
    std::memcpy(raw.data() + physical * 24 + 8, text.data(), text.size());
    auto batch = input->acquire(raw.size(), clock::now());
    batch->write(0, raw);
    sirius_input_vector v{};
    v.vector_class = cls;
    v.data_bytes   = physical * 24;
    v.null_offset  = physical * 24;
    v.null_bytes   = 8;
    v.area_offset  = physical * 24 + 8;
    v.area_bytes   = text.size();
    input->publish(batch, 65, {&v, 1});
    auto fault     = std::make_unique<fail_after_upload>(std::move(batch->storage));
    auto* inject   = fault.get();
    batch->storage = std::move(fault);
    batch.reset();
    auto unit = input->claim();
    REQUIRE(unit);
    REQUIRE_THROWS_WITH(convert_native_input(*unit, input->schema, *gpu, stream.view()),
                        "injected failure after asynchronous upload");
    CHECK(input->inspect().retained_bytes > 0);
    inject->fail = false;
    auto table   = convert_native_input(*unit, input->schema, *gpu, stream.view());
    auto strings = cudf::strings_column_view(table->view().column(0));
    std::vector<char> chars(strings.chars_size(stream.view()));
    int32_t offsets[66]{};
    uint32_t mask[3]{};
    cudaMemcpyAsync(offsets,
                    strings.offsets().data<int32_t>(),
                    sizeof(offsets),
                    cudaMemcpyDeviceToHost,
                    stream.value());
    cudaMemcpyAsync(
      mask, strings.null_mask(), sizeof(mask), cudaMemcpyDeviceToHost, stream.value());
    if (!chars.empty())
      cudaMemcpyAsync(chars.data(),
                      strings.chars_begin(stream.view()),
                      chars.size(),
                      cudaMemcpyDeviceToHost,
                      stream.value());
    stream.synchronize();
    for (int row = 0; row < 65; ++row) {
      bool null = cls == SIRIUS_VECTOR_NULL || (cls == SIRIUS_VECTOR_FLAT && row == 63);
      REQUIRE(bool((mask[row / 32] >> (row % 32)) & 1u) == !null);
      CHECK(offsets[row + 1] - offsets[row] == (null ? 0 : static_cast<int>(text.size())));
      if (!null) CHECK(std::string_view(chars.data() + offsets[row], text.size()) == text);
    }
    unit.reset();
    input->stop();
    input->discard();
    CHECK(input->inspect().retained_bytes == 0);
  }
}
