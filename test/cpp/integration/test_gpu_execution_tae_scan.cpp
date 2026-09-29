/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// End-to-end GPU coverage for the TAE scanner. These use the scanner's checked-in
// fixtures and disable DuckDB fallback so a missing planner admission or a runtime
// incompatibility cannot silently pass on CPU.

#include <catch.hpp>
#include <duckdb/common/local_file_system.hpp>
#include <embedding/plan_bindings.hpp>
#include <op/scan/tae_gpu_ingestible.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <filesystem>
#include <string>

namespace {

std::string tae_manifest(const std::string& name)
{
#ifdef SIRIUS_PROJECT_ROOT
  auto const path =
    std::filesystem::path(SIRIUS_PROJECT_ROOT) / "tae-scanner" / "test" / "data" / name;
#else
  auto const path =
    std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() /
    "tae-scanner" / "test" / "data" / name;
#endif
  REQUIRE(std::filesystem::exists(path));
  return path.string();
}

std::string tae_scan(const std::string& manifest) { return "tae_scan('" + manifest + "')"; }

class TaeScanGpuFixture : public sirius::test::GpuExecutionFixture {
 public:
  TaeScanGpuFixture() { run_ok("SET enable_duckdb_fallback = false;"); }

  ~TaeScanGpuFixture() { con->Query("SET enable_duckdb_fallback = true;"); }
};

}  // namespace

TEST_CASE("TAE scan metadata provides a path-free diagnostic name", "[tae_scan][metadata]")
{
  sirius::op::scan::tae_ingestible_table_info info;
  auto const& base = static_cast<sirius::op::scan::ingestible_table_info const&>(info);
  REQUIRE(base.display_name() == "tae_scan");
  REQUIRE(base.file_paths().empty());
  REQUIRE(base.column_names().empty());
}

TEST_CASE_METHOD(TaeScanGpuFixture,
                 "embedded TAE mode follows its binding and respects bounded staging entitlements",
                 "[integration][gpu_execution][tae_scan][embedded_tae]")
{
  using namespace sirius::embedding;
  auto const root = std::filesystem::path(tae_manifest("manifest_multi.json")).parent_path();
  duckdb::LocalFileSystem fs;
  tae::TAEObjectReader reader(fs, (root / "multi_block.tae").string());
  reader.ReadMeta();
  std::size_t block_bytes  = 0;
  std::size_t object_bytes = 0;
  for (auto const& block : reader.Meta().blocks) {
    REQUIRE_FALSE(block.columns.empty());
    block_bytes = std::max(block_bytes, static_cast<std::size_t>(block.columns[0].location.length));
    object_bytes += block.columns[0].location.length;
  }
  REQUIRE(block_bytes > 0);
  REQUIRE(object_bytes > block_bytes);

  auto catalog = duckdb::make_shared_ptr<embedded_bind_catalog>();
  con->context->registered_state->Insert(embedded_bind_catalog::key, catalog);
  register_embedded_read_functions(duckdb::DatabaseInstance::GetDatabase(*con->context));
  for (bool sorted : {false, true}) {
    CAPTURE(sorted);
    auto manifest = std::make_shared<tae::TAEScanBindData>();
    auto json =
      std::string(
        R"({"database":"test_db","table":"test_multi","columns":[{"name":"col_int","oid":22}],"objects":[{"path":"multi_block.tae","rows":8,"blocks":2}])") +
      (sorted ? R"(,"sort_column":"col_int"})" : "}");
    tae::ParseManifestBytes(json, root.string(), *manifest);
    constexpr std::size_t host_capacity = 4096;
    auto budget                         = std::make_shared<buffer_budget>(host_capacity, 2);
    auto demand                         = make_tae_demand_controller(1, budget, host_capacity);
    embedded_binding binding;
    binding.names           = {"col_int"};
    binding.types           = {duckdb::LogicalType::INTEGER};
    binding.tae             = std::move(manifest);
    binding.tae_host_budget = budget;
    binding.tae_demand      = demand;
    auto const id           = sorted ? 2 : 1;
    catalog->declare(id, std::move(binding));

    run_ok("SET gpu_execution = true;");
    auto before = sirius::test::get_transparent_execution_stats(*con);
    auto gpu    = con->Query("SELECT col_int FROM sirius_embedded_tae_read(" + std::to_string(id) +
                          ") ORDER BY col_int");
    REQUIRE(gpu);
    if (gpu->HasError()) UNSCOPED_INFO(gpu->GetError());
    REQUIRE_FALSE(gpu->HasError());
    auto after = sirius::test::get_transparent_execution_stats(*con);
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    demand->close();
    auto usage = budget->inspect();
    CHECK(usage.peak <= host_capacity);
    CHECK(usage.peak >= 2048);
    CHECK(usage.bytes == 0);
    CHECK(usage.leases == 0);

    run_ok("SET gpu_execution = false;");
    auto cpu = con->Query("SELECT col_int FROM " + tae_scan(tae_manifest("manifest_multi.json")) +
                          " ORDER BY col_int");
    REQUIRE(cpu);
    REQUIRE_FALSE(cpu->HasError());
    CHECK(collect_rows(*gpu, false) == collect_rows(*cpu, false));
  }
  catalog->clear();
}

TEST_CASE_METHOD(TaeScanGpuFixture,
                 "gpu_execution executes tae_scan fixtures on GPU without DuckDB fallback",
                 "[integration][gpu_execution][tae_scan]")
{
  SECTION("basic projection and aggregation")
  {
    auto const scan = tae_scan(tae_manifest("manifest.json"));
    compare_gpu_vs_cpu("SELECT col_int, col_str FROM " + scan + " ORDER BY col_int");
    compare_gpu_vs_cpu("SELECT count(*), min(col_int), max(col_int) FROM " + scan);
  }

  SECTION("LZ4 data with a filter on a non-projected column")
  {
    auto const scan = tae_scan(tae_manifest("manifest_lz4.json"));
    compare_gpu_vs_cpu("SELECT col_str FROM " + scan + " WHERE col_int > 50 ORDER BY col_str");
  }

  SECTION("nullable values and multi-file manifests")
  {
    auto const null_scan = tae_scan(tae_manifest("manifest_nulls.json"));
    compare_gpu_vs_cpu("SELECT col_int, col_str FROM " + null_scan +
                       " WHERE col_int IS NULL OR col_str IS NULL ORDER BY col_int NULLS FIRST");

    auto const multi_scan = tae_scan(tae_manifest("manifest_multifile.json"));
    compare_gpu_vs_cpu("SELECT col_int, col_str FROM " + multi_scan +
                       " WHERE col_int > 50 ORDER BY col_int");
  }
}
