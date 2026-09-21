/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/control.hpp"
#include "substrait/plan.pb.h"
#include "tae_scanner.hpp"

#include <catch.hpp>

namespace {
substrait::Plan one_read(std::string table = "1")
{
  substrait::Plan plan;
  plan.mutable_version()->set_major_number(0);
  plan.mutable_version()->set_minor_number(78);
  auto* root = plan.add_relations()->mutable_root();
  root->add_names("c");
  auto* read = root->mutable_input()->mutable_read();
  read->mutable_named_table()->add_names("__sirius_embedded_v1");
  read->mutable_named_table()->add_names(std::move(table));
  read->mutable_base_schema()->add_names("c");
  read->mutable_base_schema()->mutable_struct_()->add_types()->mutable_i64()->set_nullability(
    substrait::Type::NULLABILITY_REQUIRED);
  return plan;
}
sirius::embedding::query_state bound_query()
{
  sirius::embedding::query_state query;
  query.contract = std::make_unique<sirius::embedding::owned_query_contract>();
  query.contract->outputs.push_back({23, 0, 0, false, "c"});
  sirius::embedding::owned_read_binding read;
  read.binding_id  = 1;
  read.source_kind = SIRIUS_READ_MO;
  read.columns.push_back({{23, 0, 0, false, "c"}, 9, 3});
  query.bindings.push_back(std::move(read));
  return query;
}
}  // namespace

TEST_CASE("embedded plan admission accepts only exact registered reads", "[native_binding]")
{
  auto query = bound_query();
  auto plan  = one_read();
  REQUIRE_NOTHROW(sirius::embedding::validate_embedded_plan(plan.SerializeAsString(), query));

  plan = one_read("01");
  REQUIRE_THROWS_AS(sirius::embedding::validate_embedded_plan(plan.SerializeAsString(), query),
                    sirius::embedding::failure);
  plan = one_read("2");
  REQUIRE_THROWS_AS(sirius::embedding::validate_embedded_plan(plan.SerializeAsString(), query),
                    sirius::embedding::failure);
  plan = one_read();
  plan.mutable_version()->set_minor_number(79);
  REQUIRE_THROWS_AS(sirius::embedding::validate_embedded_plan(plan.SerializeAsString(), query),
                    sirius::embedding::failure);
}

TEST_CASE("embedded plan admission rejects schema and arbitrary reads", "[native_binding]")
{
  auto query = bound_query();
  auto plan  = one_read();
  plan.mutable_relations(0)
    ->mutable_root()
    ->mutable_input()
    ->mutable_read()
    ->mutable_base_schema()
    ->set_names(0, "wrong");
  REQUIRE_THROWS_AS(sirius::embedding::validate_embedded_plan(plan.SerializeAsString(), query),
                    sirius::embedding::failure);
  plan = one_read();
  plan.mutable_relations(0)
    ->mutable_root()
    ->mutable_input()
    ->mutable_read()
    ->mutable_base_schema()
    ->mutable_struct_()
    ->mutable_types(0)
    ->Clear();
  plan.mutable_relations(0)
    ->mutable_root()
    ->mutable_input()
    ->mutable_read()
    ->mutable_base_schema()
    ->mutable_struct_()
    ->mutable_types(0)
    ->mutable_i32()
    ->set_nullability(substrait::Type::NULLABILITY_REQUIRED);
  REQUIRE_THROWS_AS(sirius::embedding::validate_embedded_plan(plan.SerializeAsString(), query),
                    sirius::embedding::failure);
  plan        = one_read();
  auto* names = plan.mutable_relations(0)
                  ->mutable_root()
                  ->mutable_input()
                  ->mutable_read()
                  ->mutable_named_table();
  names->set_names(0, "main");
  REQUIRE_THROWS_AS(sirius::embedding::validate_embedded_plan(plan.SerializeAsString(), query),
                    sirius::embedding::failure);
  plan = one_read();
  plan.mutable_relations(0)->mutable_root()->mutable_input()->mutable_write();
  REQUIRE_THROWS_AS(sirius::embedding::validate_embedded_plan(plan.SerializeAsString(), query),
                    sirius::embedding::failure);
}

TEST_CASE("embedded TAE manifest bytes are bounded and path confined", "[native_binding][tae]")
{
  constexpr std::string_view manifest =
    R"({"database":"tpch","table":"lineitem","data_dir":"s3://bucket/table","columns":[{"name":"l_orderkey","oid":23,"seqnum":7}],"objects":[{"path":"obj/0001","rows":8192,"blocks":1,"size":4096}]})";
  tae::TAEScanBindData bind;
  REQUIRE_NOTHROW(tae::ParseManifestBytes(manifest, "s3://bucket/table", bind));
  CHECK(bind.data_dir == "s3://bucket/table");
  REQUIRE(bind.all_col_seqnums.size() == 1);
  CHECK(bind.all_col_seqnums[0] == 7);
  REQUIRE(bind.objects.size() == 1);
  CHECK(bind.objects[0].file_path == "obj/0001");

  tae::TAEScanBindData mismatch;
  REQUIRE_THROWS(tae::ParseManifestBytes(manifest, "s3://bucket/other", mismatch));
  auto traversal = std::string(manifest);
  auto path      = traversal.find("obj/0001");
  traversal.replace(path, 8, "../evil");
  tae::TAEScanBindData unsafe;
  REQUIRE_THROWS(tae::ParseManifestBytes(traversal, "s3://bucket/table", unsafe));
}
