/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
// Production ABI integration. Unlike native_gpu_unittest this executable does
// not replace native_backend_factory with a private backend.
#include "sirius_c.h"
#include "substrait/plan.pb.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
namespace {
constexpr uint64_t rows = 12u * 1024u * 1024u;  // 96 MiB of scalar values alone.
void require(bool condition, char const* reason)
{
  if (!condition) throw std::runtime_error(reason);
}
void ok(sirius_status code, sirius_error const& error)
{
  if (code != SIRIUS_OK) throw std::runtime_error(std::to_string(code) + ": " + error.message);
}
std::string plan(bool ordered)
{
  substrait::Plan value;
  value.mutable_version()->set_minor_number(78);
  auto root = value.add_relations()->mutable_root();
  root->add_names("c");
  auto relation = root->mutable_input();
  if (ordered) {
    auto sort = relation->mutable_sort();
    auto key  = sort->add_sorts();
    key->set_direction(substrait::SortField::SORT_DIRECTION_ASC_NULLS_FIRST);
    auto selection = key->mutable_expr()->mutable_selection();
    selection->mutable_direct_reference()->mutable_struct_field()->set_field(0);
    selection->mutable_root_reference();
    relation = sort->mutable_input();
  }
  auto read = relation->mutable_read();
  read->mutable_named_table()->add_names("__sirius_embedded_v1");
  read->mutable_named_table()->add_names("1");
  read->mutable_base_schema()->add_names("c");
  read->mutable_base_schema()->mutable_struct_()->add_types()->mutable_i64()->set_nullability(
    substrait::Type::NULLABILITY_REQUIRED);
  return value.SerializeAsString();
}
struct query {
  sirius_query_handle* handle{};
  sirius_input_handle* input{};
  sirius_batch_handle* borrowed{};
  std::thread producer;
  std::atomic<sirius_status> producer_outcome{SIRIUS_OK};
  query(sirius_engine_handle* engine, bool ordered, uint32_t timeout)
  {
    sirius_error error{};
    auto bytes = plan(ordered);
    sirius_query_options options{sizeof(options), SIRIUS_ABI_VERSION, timeout, 0};
    ok(sirius_query_create(engine, &options, bytes.data(), bytes.size(), &handle, &error), error);
    sirius_column column{23, 0, 0, 0, "c", 1, 0};
    sirius_query_contract contract{
      sizeof(contract), SIRIUS_ABI_VERSION, 0, 0, "result-integration", 18, {}, &column, 1};
    sirius_read_column physical{column, 7, 3, 0};
    sirius_read_binding binding{sizeof(binding),
                                SIRIUS_ABI_VERSION,
                                1,
                                SIRIUS_READ_MO,
                                0,
                                "db",
                                2,
                                "t",
                                1,
                                "s",
                                1,
                                &physical,
                                1,
                                nullptr,
                                0,
                                nullptr,
                                0};
    sirius_input_column scalar{23, 0, 0, 0};
    try {
      ok(sirius_query_bind(handle, &contract, &error), error);
      ok(sirius_read_register(handle, &binding, &error), error);
      ok(sirius_input_register(handle, 1, &scalar, 1, &input, &error), error);
      ok(sirius_query_prepare(handle, 30000, &error), error);
      ok(sirius_query_start(handle, &error), error);
      producer = std::thread([this, ordered] {
        sirius_error error{};
        sirius_batch_handle* batch{};
        try {
          std::vector<int64_t> values(128u << 10);
          for (uint64_t first = 0; first < rows; first += values.size()) {
            for (uint64_t i = 0; i < values.size(); ++i)
              values[i] = ordered ? rows - first - i - 1 : first + i;
            auto status = sirius_input_acquire(input, values.size() * 8, 30000, &batch, &error);
            if (status != SIRIUS_OK) {
              producer_outcome = status;
              return;
            }
            status = sirius_input_write(batch, 0, values.data(), values.size() * 8, &error);
            if (status == SIRIUS_OK) {
              sirius_input_vector vector{};
              vector.data_bytes = values.size() * 8;
              status = sirius_input_publish(input, &batch, values.size(), &vector, 1, &error);
            }
            if (status != SIRIUS_OK) {
              sirius_batch_release(&batch, &error);
              producer_outcome = status;
              return;
            }
          }
          producer_outcome = sirius_input_finish(input, &error);
        } catch (...) {
          sirius_batch_release(&batch, &error);
          producer_outcome = SIRIUS_EXECUTION_FAILED;
          sirius_query_cancel(handle, &error);
        }
      });
    } catch (...) {
      cleanup();
      throw;
    }
  }
  ~query() { cleanup(); }
  void cleanup()
  {
    sirius_error error{};
    if (handle) sirius_query_cancel(handle, &error);
    if (producer.joinable()) producer.join();
    if (borrowed) sirius_batch_release(&borrowed, &error);
    if (input) sirius_input_close(&input, &error);
    if (handle) sirius_query_close(&handle, 30000, &error);
  }
  sirius_result_stats stats()
  {
    sirius_error error{};
    sirius_result_stats value{sizeof(value), SIRIUS_ABI_VERSION};
    ok(sirius_query_get_result_stats(handle, &value, &error), error);
    require(value.retained_bytes <= (64u << 20) && value.peak_bytes <= (64u << 20),
            "native result window exceeded 64 MiB");
    return value;
  }
  void stall()
  {
    sirius_error error{};
    ok(sirius_query_next_result(handle, 30000, &borrowed, &error), error);
    auto until = std::chrono::steady_clock::now() + 8s;
    while (std::chrono::steady_clock::now() < until) {
      auto state = stats();
      if (state.parked_publications) {
        require(state.borrowed_batches == 1, "dequeue dropped the borrowed result owner");
        // A failed nonblocking acquisition is the full-window oracle. Pool
        // rounding can leave unused bytes that cannot admit the next slice.
        require(state.blocked_publications != 0 && state.retained_bytes != 0 && state.leases < 128,
                "publication did not park on byte capacity with a borrowed owner");
        return;
      }
      std::this_thread::sleep_for(10ms);
    }
    throw std::runtime_error("stalled consumer never parked native publication");
  }
};
void equivalence(sirius_engine_handle* engine, bool ordered)
{
  query run(engine, ordered, 90000);
  run.stall();
  uint64_t count{}, sum{}, batches{};
  std::vector<uint8_t> seen(ordered ? 0 : (rows + 7) / 8);
  sirius_error error{};
  do {
    sirius_result_batch_info info{sizeof(info), SIRIUS_ABI_VERSION};
    ok(sirius_result_describe(run.borrowed, &info, &error), error);
    require(info.column_count == 1, "unexpected result schema");
    std::vector<int64_t> values(info.rows);
    ok(sirius_result_read(
         run.borrowed, info.columns[0].data_offset, values.data(), values.size() * 8, &error),
       error);
    for (auto value : values) {
      require(value >= 0 && static_cast<uint64_t>(value) < rows, "result value outside domain");
      if (ordered)
        require(static_cast<uint64_t>(value) == count, "ORDER BY result lost global order");
      else {
        auto bit = uint8_t(1u << (value % 8));
        require(!(seen[value / 8] & bit), "native result duplicated a row");
        seen[value / 8] |= bit;
      }
      ++count;
      sum += value;
    }
    ++batches;
    ok(sirius_batch_release(&run.borrowed, &error), error);
    run.stats();
    auto next = sirius_query_next_result(run.handle, 30000, &run.borrowed, &error);
    if (next == SIRIUS_EOF) break;
    ok(next, error);
  } while (true);
  run.producer.join();
  require(run.producer_outcome == SIRIUS_OK, "producer failed in equivalence query");
  ok(sirius_query_wait(run.handle, 30000, &error), error);
  require(count == rows && sum == rows * (rows - 1) / 2 && batches >= 3,
          "multi-batch native output differs from input");
  require(run.stats().blocked_publications != 0, "full result did not exercise backpressure");
}
void interruption(sirius_engine_handle* engine, bool deadline)
{
  query run(engine, false, deadline ? 12000 : 90000);
  run.stall();
  sirius_error error{};
  if (!deadline) ok(sirius_query_cancel(run.handle, &error), error);
  auto outcome = sirius_query_wait(run.handle, 30000, &error);
  require(outcome == (deadline ? SIRIUS_TIMEOUT : SIRIUS_CANCELLED), "wrong interruption outcome");
  run.producer.join();
  ok(sirius_input_close(&run.input, &error), error);
  require(sirius_query_close(&run.handle, 30000, &error) == SIRIUS_BUSY,
          "query closed while result was borrowed");
  sirius_result_batch_info info{sizeof(info), SIRIUS_ABI_VERSION};
  ok(sirius_result_describe(run.borrowed, &info, &error), error);
  int64_t value{};
  ok(sirius_result_read(run.borrowed, info.columns[0].data_offset, &value, sizeof(value), &error),
     error);
  require(value >= 0 && static_cast<uint64_t>(value) < rows, "borrow invalid after interruption");
  require(run.stats().retained_bytes != 0, "borrow lost physical charge during query cleanup");
  ok(sirius_batch_release(&run.borrowed, &error), error);
  require(run.stats().retained_bytes == 0, "result storage leaked after final release");
  ok(sirius_query_close(&run.handle, 30000, &error), error);
}

void tae_results(sirius_engine_handle* engine)
{
#ifdef SIRIUS_PROJECT_ROOT
  auto fixture = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "tae-scanner" / "test" / "data";
#else
  auto fixture =
    std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() /
    "tae-scanner" / "test" / "data";
#endif
  require(std::filesystem::exists(fixture / "manifest_multi.json") &&
            std::filesystem::exists(fixture / "multi_block.tae"),
          "checked-in TAE multi-block fixture is missing");
  auto data_root = std::filesystem::absolute(fixture).string();
  // Project manifest_multi.json's first physical column. Its omitted seqnum
  // defaults to position zero; make that mapping explicit in this binding.
  std::string manifest =
    R"({"database":"test_db","table":"test_multi","columns":[{"name":"col_int","oid":22,"seqnum":0}],"objects":[{"path":"multi_block.tae","rows":8,"blocks":2}]})";
  substrait::Plan value;
  require(value.ParseFromString(plan(false)), "could not construct TAE Substrait plan");
  auto root = value.mutable_relations(0)->mutable_root();
  root->set_names(0, "col_int");
  auto read = root->mutable_input()->mutable_read();
  read->mutable_base_schema()->set_names(0, "col_int");
  read->mutable_base_schema()->mutable_struct_()->mutable_types(0)->mutable_i32()->set_nullability(
    substrait::Type::NULLABILITY_REQUIRED);
  auto bytes = value.SerializeAsString();
  struct handles {
    sirius_query_handle* query{};
    sirius_batch_handle* batch{};
    ~handles()
    {
      sirius_error error{};
      if (query) sirius_query_cancel(query, &error);
      if (batch) sirius_batch_release(&batch, &error);
      if (query) sirius_query_close(&query, 30000, &error);
    }
  } run;
  sirius_error error{};
  sirius_query_options options{sizeof(options), SIRIUS_ABI_VERSION, 30000, 0};
  sirius_column column{22, 0, 0, 0, "col_int", 7, 0};
  sirius_read_column physical{column, 0, 0, 0};
  sirius_query_contract contract{
    sizeof(contract), SIRIUS_ABI_VERSION, 0, 0, "tae-native", 10, {}, &column, 1};
  sirius_read_binding binding{sizeof(binding),
                              SIRIUS_ABI_VERSION,
                              1,
                              SIRIUS_READ_TAE,
                              0,
                              "test_db",
                              7,
                              "test_multi",
                              10,
                              "s",
                              1,
                              &physical,
                              1,
                              manifest.data(),
                              manifest.size(),
                              data_root.data(),
                              static_cast<uint32_t>(data_root.size())};
  ok(sirius_query_create(engine, &options, bytes.data(), bytes.size(), &run.query, &error), error);
  ok(sirius_query_bind(run.query, &contract, &error), error);
  ok(sirius_read_register(run.query, &binding, &error), error);
  // No MO producer is registered: the real TAE adapter must supply all rows.
  ok(sirius_query_prepare(run.query, 30000, &error), error);
  sirius_result_schema schema{sizeof(schema), SIRIUS_ABI_VERSION};
  ok(sirius_query_get_schema(run.query, &schema, &error), error);
  require(schema.column_count == 1 && schema.columns[0].oid == 22,
          "TAE native output lost its i32 schema");
  ok(sirius_query_start(run.query, &error), error);
  std::vector<int32_t> observed;
  while (true) {
    auto status = sirius_query_next_result(run.query, 30000, &run.batch, &error);
    if (status == SIRIUS_EOF) break;
    ok(status, error);
    sirius_result_batch_info info{sizeof(info), SIRIUS_ABI_VERSION};
    ok(sirius_result_describe(run.batch, &info, &error), error);
    require(info.column_count == 1 && info.rows <= 8 && observed.size() + info.rows <= 8,
            "TAE native result shape differs from fixture");
    require(info.columns[0].vector_class == SIRIUS_VECTOR_FLAT &&
              info.columns[0].data_bytes == uint64_t(info.rows) * sizeof(int32_t),
            "TAE native i32 vector layout is incorrect");
    auto offset = observed.size();
    observed.resize(offset + info.rows);
    if (info.rows)
      ok(sirius_result_read(run.batch,
                            info.columns[0].data_offset,
                            observed.data() + offset,
                            uint64_t(info.rows) * sizeof(int32_t),
                            &error),
         error);
    ok(sirius_batch_release(&run.batch, &error), error);
  }
  ok(sirius_query_wait(run.query, 30000, &error), error);
  std::sort(observed.begin(), observed.end());
  require(observed == std::vector<int32_t>{1, 2, 3, 4, 100, 200, 300, 400},
          "TAE native result values differ from the two checked-in blocks");
  sirius_result_stats stats{sizeof(stats), SIRIUS_ABI_VERSION};
  ok(sirius_query_get_result_stats(run.query, &stats, &error), error);
  require(stats.peak_bytes <= (64u << 20) && stats.retained_bytes == 0,
          "TAE native result exceeded its window or leaked released output");
  ok(sirius_query_close(&run.query, 30000, &error), error);
}
}  // namespace
int main(int argc, char** argv)
{
  if (argc < 2 || argc > 3) return 2;
  sirius_engine_handle* engine{};
  sirius_error error{};
  try {
    std::string config = argv[1];
    uint32_t streams   = argc == 3 ? std::stoul(argv[2]) : 2;
    sirius_engine_options options{sizeof(options),
                                  SIRIUS_ABI_VERSION,
                                  config.data(),
                                  static_cast<uint32_t>(config.size()),
                                  0,
                                  streams,
                                  0};
    auto rejected_path = (std::filesystem::path(__FILE__).parent_path().parent_path() /
                          "config/data/embedding-multigpu.yaml")
                           .string();
    auto rejected              = options;
    rejected.config_path       = rejected_path.data();
    rejected.config_path_bytes = static_cast<uint32_t>(rejected_path.size());
    require(sirius_engine_create(&rejected, &engine, &error) == SIRIUS_UNSUPPORTED && !engine,
            "multi-GPU native configuration was not rejected before runtime initialization");
    ok(sirius_engine_create(&options, &engine, &error), error);
    tae_results(engine);
    std::cout << "production TAE multi-block native result: passed\n";
    equivalence(engine, false);
    std::cout << "unordered 96 MiB bounded result: passed\n";
    interruption(engine, false);
    std::cout << "full-window cancellation and borrowed lifetime: passed\n";
    interruption(engine, true);
    std::cout << "full-window deadline and borrowed lifetime: passed\n";
    equivalence(engine, true);
    std::cout << "ordered 96 MiB result: passed\n";
    ok(sirius_engine_close(&engine, 30000, &error), error);
    return 0;
  } catch (std::exception const& e) {
    std::cerr << "native result integration: " << e.what() << '\n';
    if (engine) sirius_engine_close(&engine, 30000, &error);
    return 1;
  }
}
