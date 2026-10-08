/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "decimal_import_fixture.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/to_duckdb.hpp"
#include "expression/ast/utils.hpp"
#include "expression/value.hpp"
#include "helper/type_conversions.hpp"
#include "numeric/decimal_plan.hpp"
#include "numeric/decimal_types.hpp"
#include "numeric/exact_decimal.hpp"

#include <catch.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>

using namespace sirius;

TEST_CASE("MO import carriers retain width precision scale and nullability", "[decimal_import]")
{
  for (uint16_t bits : {64, 128, 256}) {
    for (bool nullable : {false, true}) {
      auto type = logical_type::make_mo_decimal({bits, 9, 2}, nullable);
      CHECK(type.fixed_width_byte_size() == bits / 8);
      auto carrier = to_duckdb(type);
      CHECK(from_duckdb(carrier) == type);
      CHECK(carrier.id() == (bits == 64    ? duckdb::LogicalTypeId::BIGINT
                             : bits == 128 ? duckdb::LogicalTypeId::HUGEINT
                                           : duckdb::LogicalTypeId::STRUCT));
      for (int64_t number : {0, 125, -125}) {
        auto coefficient =
          mo_decimal::load_coefficient(reinterpret_cast<uint8_t const*>(&number), 8);
        auto value    = mo_decimal::duckdb_value(coefficient, type);
        auto observed = mo_decimal::from_duckdb_value(value, type);
        for (int i = 0; i < 8; ++i)
          CHECK(observed.words[i] == coefficient.words[i]);
        auto literal = ast::node(ast::constant{from_duckdb(value, type), type});
        auto copied  = ast::clone(literal);
        CHECK(copied->return_type() == type);
        CHECK(
          to_duckdb(std::get<mo_decimal::coefficient>(copied->get<ast::constant>().payload), type)
            .type() == carrier);
      }
      if (nullable) {
        auto null = from_duckdb(duckdb::Value(carrier), type);
        CHECK(std::holds_alternative<null_value>(null));
        CHECK(to_duckdb(null, type).type() == carrier);
        CHECK(to_duckdb(null, type).IsNull());
      } else
        REQUIRE_THROWS(from_duckdb(duckdb::Value(carrier), type));
    }
  }
  auto ordinary = from_duckdb(duckdb::LogicalType::DECIMAL(9, 2));
  CHECK(ordinary.id() == type_id::DECIMAL);
  CHECK(ordinary.fixed_width_byte_size() == 4);
  CHECK(to_duckdb(ordinary) == duckdb::LogicalType::DECIMAL(9, 2));
}

TEST_CASE("MO import rejects malformed and forged physical carriers", "[decimal_import]")
{
  REQUIRE_THROWS(logical_type::make_mo_decimal({64, 19, 0}, false));
  REQUIRE_THROWS(logical_type::make_mo_decimal({256, 77, 0}, true));
  auto real                  = to_duckdb(logical_type::make_mo_decimal({256, 15, 2}, true));
  duckdb::LogicalType forged = duckdb::LogicalType::BIGINT;
  forged.SetAlias(real.GetAlias());
  REQUIRE_THROWS(from_duckdb(forged));
  for (auto alias : {"__sirius_mo_exact_v1_d_256_15_2_0",
                     "__sirius_mo_exact_v1_d_256_015_2_2",
                     "__sirius_mo_exact_v1_d_256_15_16_2",
                     "__sirius_mo_exact_v1_d_256_15_2_2_extra"}) {
    auto invalid = real;
    invalid.SetAlias(alias);
    REQUIRE_THROWS(from_duckdb(invalid));
  }
  int64_t value    = 1000;
  auto coefficient = mo_decimal::load_coefficient(reinterpret_cast<uint8_t const*>(&value), 8);
  REQUIRE_THROWS(
    mo_decimal::duckdb_value(coefficient, logical_type::make_mo_decimal({64, 3, 0}, false)));
}

TEST_CASE("MO scalar import binds exact types without numeric coercion", "[decimal_import]")
{
  decimal_fixture::importer importer;
  for (uint16_t bits : {64, 128, 256}) {
    mo_decimal::decimal_type input{bits, 9, 2}, output{bits, 15, 2};
    auto plan = decimal_fixture::plan(
      "mo_decimal_add",
      decimal_fixture::type(output),
      {decimal_fixture::literal(input, 125), decimal_fixture::literal(input, -25)});
    auto expression = importer.bind(plan);
    REQUIRE(expression->is_function_call());
    CHECK(expression->as_function_call().function() == function_id::mo_decimal_add);
    CHECK(expression->return_type() == logical_type::make_mo_decimal(output, false));
    REQUIRE(expression->as_function_call().arguments().size() == 2);
    for (auto const& argument : expression->as_function_call().arguments())
      CHECK(argument->return_type() == logical_type::make_mo_decimal(input, false));
    auto copied        = ast::clone(*expression);
    auto reconstructed = ast::to_duckdb(*copied);
    auto roundtrip     = ast::from_duckdb(*reconstructed);
    CHECK(roundtrip->return_type() == expression->return_type());
    CHECK(roundtrip->as_function_call().function() == function_id::mo_decimal_add);
  }
}

TEST_CASE("MO scalar import rejects unrelated identities and malformed literals",
          "[decimal_import]")
{
  decimal_fixture::importer importer;
  mo_decimal::decimal_type input{256, 15, 2};
  auto original = decimal_fixture::plan(
    "mo_decimal_cast", decimal_fixture::type(input), {decimal_fixture::literal(input, 125)});
  REQUIRE_NOTHROW(importer.bind(original));
  auto bad    = original;
  auto scalar = bad.mutable_relations(0)
                  ->mutable_root()
                  ->mutable_input()
                  ->mutable_project()
                  ->mutable_expressions(0)
                  ->mutable_scalar_function();
  scalar->mutable_output_type()->mutable_user_defined()->set_type_reference(999);
  REQUIRE_THROWS(importer.bind(bad));
  bad    = original;
  scalar = bad.mutable_relations(0)
             ->mutable_root()
             ->mutable_input()
             ->mutable_project()
             ->mutable_expressions(0)
             ->mutable_scalar_function();
  scalar->mutable_arguments(0)
    ->mutable_value()
    ->mutable_literal()
    ->mutable_user_defined()
    ->mutable_value()
    ->set_value(std::string(32, '\0'));
  REQUIRE_THROWS(importer.bind(bad));
  bad = original;
  bad.mutable_extensions(1)->mutable_extension_function()->set_name("mo_decimal_cast:unproven");
  REQUIRE_THROWS(importer.bind(bad));
  REQUIRE_NOTHROW(importer.bind(original));
  bad    = original;
  scalar = bad.mutable_relations(0)
             ->mutable_root()
             ->mutable_input()
             ->mutable_project()
             ->mutable_expressions(0)
             ->mutable_scalar_function();
  auto payload = scalar->mutable_arguments(0)
                   ->mutable_value()
                   ->mutable_literal()
                   ->mutable_user_defined()
                   ->mutable_value();
  auto extended = payload->value();
  extended.append("\x10\x11", 2);  // Unknown protobuf field 2, varint 17.
  payload->set_value(extended);
  REQUIRE_NOTHROW(importer.bind(bad));
  extended    = payload->value();
  extended[1] = static_cast<char>(static_cast<uint8_t>(extended[1]) | 0x80);
  extended.insert(extended.begin() + 2, '\0');  // Valid nonminimal length varint.
  payload->set_value(extended);
  REQUIRE_NOTHROW(importer.bind(bad));
  payload->set_type_url("type.googleapis.com/unrelated.Numeric");
  REQUIRE_THROWS(importer.bind(bad));
  auto type = logical_type::make_mo_decimal({128, 19, 0}, false);
  REQUIRE_NOTHROW(mo_decimal::validate_signature(
    mo_decimal::decimal_op::cast, {logical_type::make(type_id::BIGINT)}, type));
  REQUIRE_THROWS(mo_decimal::validate_signature(mo_decimal::decimal_op::cast,
                                                {logical_type::make(type_id::BIGINT)},
                                                logical_type::make_mo_decimal({64, 18, 0}, false)));
}

TEST_CASE("MO aggregate import preserves checked signatures and rejects modifiers",
          "[decimal_import]")
{
  decimal_fixture::importer importer;
  for (auto op : {mo_decimal::aggregate_op::sum,
                  mo_decimal::aggregate_op::avg,
                  mo_decimal::aggregate_op::min,
                  mo_decimal::aggregate_op::max}) {
    for (uint16_t bits : {64, 128, 256}) {
      mo_decimal::decimal_type input{bits, 15, 2};
      auto output = input;
      if (op == mo_decimal::aggregate_op::sum)
        output = {static_cast<uint16_t>(bits == 64 ? 128 : 256), 37, 2};
      if (op == mo_decimal::aggregate_op::avg)
        output = {static_cast<uint16_t>(bits == 64 ? 128 : 256), 19, 6};
      auto original = decimal_fixture::plan(
        "mo_decimal_cast", decimal_fixture::type(input), {decimal_fixture::literal(input, 125)});
      auto root                     = original.mutable_relations(0)->mutable_root();
      auto seed                     = root->input().project().input();
      auto aggregation              = root->mutable_input()->mutable_aggregate();
      *aggregation->mutable_input() = std::move(seed);
      auto measure                  = aggregation->add_measures()->mutable_measure();
      measure->set_function_reference(21);
      measure->set_phase(substrait::AGGREGATION_PHASE_INITIAL_TO_RESULT);
      measure->set_invocation(substrait::AggregateFunction::AGGREGATION_INVOCATION_ALL);
      *measure->mutable_output_type()            = decimal_fixture::type(output, true);
      *measure->add_arguments()->mutable_value() = decimal_fixture::literal(input, 125);
      auto name =
        std::string(to_duckdb_aggregate_name(mo_decimal::aggregate_function(op))).substr(9);
      original.mutable_extensions(1)->mutable_extension_function()->set_name(name);
      auto expression = importer.bind(original);
      REQUIRE(expression->is_aggregate());
      CHECK(expression->as_aggregate().function() == mo_decimal::aggregate_function(op));
      CHECK(expression->return_type() == logical_type::make_mo_decimal(output, true));
      auto cloned        = ast::clone(*expression);
      auto reverse       = ast::to_duckdb(*cloned);
      auto reconstructed = ast::from_duckdb(*reverse);
      REQUIRE(reconstructed->is_aggregate());
      CHECK(reconstructed->return_type() == expression->return_type());
      auto bad = original;
      bad.mutable_relations(0)
        ->mutable_root()
        ->mutable_input()
        ->mutable_aggregate()
        ->mutable_measures(0)
        ->mutable_measure()
        ->set_invocation(substrait::AggregateFunction::AGGREGATION_INVOCATION_DISTINCT);
      REQUIRE_THROWS(importer.bind(bad));
      bad = original;
      bad.mutable_relations(0)
        ->mutable_root()
        ->mutable_input()
        ->mutable_aggregate()
        ->mutable_measures(0)
        ->mutable_measure()
        ->set_phase(substrait::AGGREGATION_PHASE_INTERMEDIATE_TO_RESULT);
      REQUIRE_THROWS(importer.bind(bad));
    }
  }
}
TEST_CASE("MO alias-only inspection preserves shared immutable comparison bindings",
          "[decimal_import]")
{
  auto input  = logical_type::make_mo_decimal({64, 9, 2}, false);
  auto output = logical_type::make(type_id::BOOLEAN).with_nullability(false);
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> args;
  for (int64_t number : {125, 200})
    args.push_back(duckdb::make_uniq<duckdb::BoundConstantExpression>(
      mo_decimal::duckdb_value(decimal_fixture::small(number), input)));
  auto expression = mo_decimal::bound_scalar(mo_decimal::decimal_op::less, std::move(args), output);
  auto original   = expression->return_type;
  auto alias      = original.GetAlias();
  expression      = duckdb::make_uniq<duckdb::BoundCastExpression>(
    std::move(expression), duckdb::LogicalType::BOOLEAN, duckdb::BoundCastInfo(nullptr));
  mo_decimal::restore_exact_bound_types(expression);
  REQUIRE(expression->GetExpressionClass() == duckdb::ExpressionClass::BOUND_FUNCTION);
  CHECK(original.GetAlias() == alias);
  CHECK(from_duckdb(original) == output);
  REQUIRE_NOTHROW(
    mo_decimal::validate_bound_scalar(expression->Cast<duckdb::BoundFunctionExpression>()));
}
TEST_CASE("exact wire normalization is bounded and rejects ordinary numeric coercion",
          "[decimal_import]")
{
  mo_decimal::decimal_type input{64, 9, 2}, output{128, 15, 2};
  auto plan = decimal_fixture::plan(
    "mo_decimal_add",
    decimal_fixture::type(output),
    {decimal_fixture::literal(input, 125), decimal_fixture::literal(input, 25)});
  auto bytes = plan.SerializeAsString();
  REQUIRE_NOTHROW(mo_decimal::normalize_exact_substrait(bytes, 4096));
  REQUIRE_THROWS_AS(mo_decimal::normalize_exact_substrait(bytes, 0), std::bad_alloc);
  auto uri = plan.add_extension_urns();
  uri->set_extension_urn_anchor(2);
  uri->set_urn("extension:io.substrait:functions_arithmetic");
  auto function = plan.mutable_extensions(1)->mutable_extension_function();
  function->set_extension_urn_reference(2);
  function->set_name("add");
  REQUIRE_THROWS(mo_decimal::normalize_exact_substrait(plan.SerializeAsString(), 4096));
}
TEST_CASE("exact preparation rejects grouping loss and ignored aggregate filters",
          "[decimal_import]")
{
  mo_decimal::decimal_type input{64, 9, 2};
  auto plan      = decimal_fixture::plan("mo_decimal_sum", decimal_fixture::type(input), {});
  auto root      = plan.mutable_relations(0)->mutable_root();
  auto seed      = root->input().project().input();
  auto aggregate = root->mutable_input()->mutable_aggregate();
  *aggregate->mutable_input() = std::move(seed);
  for (int64_t value : {125, 250})
    *aggregate->add_grouping_expressions() = decimal_fixture::literal(input, value);
  auto grouping = aggregate->add_groupings();
  grouping->add_expression_references(0);
  grouping->add_expression_references(1);
  auto measure = aggregate->add_measures()->mutable_measure();
  measure->set_function_reference(21);
  measure->set_phase(substrait::AGGREGATION_PHASE_INITIAL_TO_RESULT);
  measure->set_invocation(substrait::AggregateFunction::AGGREGATION_INVOCATION_ALL);
  *measure->mutable_output_type()            = decimal_fixture::type({128, 15, 2}, true);
  *measure->add_arguments()->mutable_value() = decimal_fixture::literal(input, 125);
  REQUIRE_NOTHROW(mo_decimal::normalize_exact_substrait(plan.SerializeAsString(), 4096));
  auto bad = plan;
  auto get = [](substrait::Plan& p) {
    return p.mutable_relations(0)->mutable_root()->mutable_input()->mutable_aggregate();
  };
  get(bad)->mutable_groupings(0)->mutable_expression_references()->RemoveLast();
  REQUIRE_THROWS(mo_decimal::normalize_exact_substrait(bad.SerializeAsString(), 4096));
  bad = plan;
  get(bad)->mutable_groupings(0)->set_expression_references(1, 0);
  REQUIRE_THROWS(mo_decimal::normalize_exact_substrait(bad.SerializeAsString(), 4096));
  bad = plan;
  get(bad)->mutable_groupings(0)->set_expression_references(1, 2);
  REQUIRE_THROWS(mo_decimal::normalize_exact_substrait(bad.SerializeAsString(), 4096));
  bad = plan;
  get(bad)->mutable_measures(0)->mutable_filter()->mutable_literal()->set_boolean(false);
  REQUIRE_THROWS(mo_decimal::normalize_exact_substrait(bad.SerializeAsString(), 4096));
  // Ordinary COUNT shares the typed layout, so its filter must also be
  // rejected rather than silently counting rows the caller excluded.
  auto urn = bad.add_extension_urns();
  urn->set_extension_urn_anchor(2);
  urn->set_urn("extension:io.substrait:functions_aggregate_generic");
  auto function = bad.mutable_extensions(1)->mutable_extension_function();
  function->set_extension_urn_reference(2);
  function->set_name("count");
  get(bad)
    ->mutable_measures(0)
    ->mutable_measure()
    ->mutable_output_type()
    ->mutable_i64()
    ->set_nullability(substrait::Type::NULLABILITY_REQUIRED);
  REQUIRE_THROWS(mo_decimal::normalize_exact_substrait(bad.SerializeAsString(), 4096));
  get(bad)->mutable_measures(0)->clear_filter();
  REQUIRE_NOTHROW(mo_decimal::normalize_exact_substrait(bad.SerializeAsString(), 4096));
}
TEST_CASE("exact read projection carries the selected descriptor into checked binding",
          "[decimal_import]")
{
  mo_decimal::decimal_type first{64, 9, 2}, second{256, 15, 3};
  substrait::Expression reference;
  reference.mutable_selection()->mutable_root_reference();
  reference.mutable_selection()->mutable_direct_reference()->mutable_struct_field()->set_field(0);
  auto plan = decimal_fixture::plan("mo_decimal_add",
                                    decimal_fixture::type(second),
                                    {reference, decimal_fixture::literal(second, 25)});
  auto read = plan.mutable_relations(0)
                ->mutable_root()
                ->mutable_input()
                ->mutable_project()
                ->mutable_input()
                ->mutable_read();
  read->mutable_base_schema()->clear_names();
  read->mutable_base_schema()->add_names("first");
  read->mutable_base_schema()->add_names("second");
  auto schema = read->mutable_base_schema()->mutable_struct_();
  schema->clear_types();
  *schema->add_types() = decimal_fixture::type(first);
  *schema->add_types() = decimal_fixture::type(second);
  auto row             = read->mutable_virtual_table()->mutable_expressions(0);
  row->clear_fields();
  *row->add_fields() = decimal_fixture::literal(first, 125);
  *row->add_fields() = decimal_fixture::literal(second, 2500);
  read->mutable_projection()->mutable_select()->add_struct_items()->set_field(1);
  auto normalized = mo_decimal::normalize_exact_substrait(plan.SerializeAsString(), 4096);
  substrait::Plan parsed;
  REQUIRE(parsed.ParseFromString(normalized));
  decimal_fixture::importer importer;
  auto expression = importer.bind(parsed);
  REQUIRE(expression->is_function_call());
  CHECK(expression->return_type() == logical_type::make_mo_decimal(second, false));
  CHECK(expression->as_function_call().arguments()[0]->return_type() ==
        logical_type::make_mo_decimal(second, false));
  read->mutable_projection()->mutable_select()->mutable_struct_items(0)->set_field(2);
  REQUIRE_THROWS(mo_decimal::normalize_exact_substrait(plan.SerializeAsString(), 4096));
  // A zero-column cuDF table cannot retain input row cardinality for a later
  // COUNT(*). Decline this shape before it could silently turn N rows into 0.
  read->mutable_projection()->mutable_select()->clear_struct_items();
  auto seed      = plan.relations(0).root().input().project().input();
  auto aggregate = plan.mutable_relations(0)->mutable_root()->mutable_input()->mutable_aggregate();
  *aggregate->mutable_input() = std::move(seed);
  auto measure                = aggregate->add_measures()->mutable_measure();
  measure->set_function_reference(21);
  measure->set_phase(substrait::AGGREGATION_PHASE_INITIAL_TO_RESULT);
  measure->set_invocation(substrait::AggregateFunction::AGGREGATION_INVOCATION_ALL);
  measure->mutable_output_type()->mutable_i64()->set_nullability(
    substrait::Type::NULLABILITY_REQUIRED);
  auto urn = plan.add_extension_urns();
  urn->set_extension_urn_anchor(2);
  urn->set_urn("extension:io.substrait:functions_aggregate_generic");
  plan.mutable_extensions(1)->mutable_extension_function()->set_extension_urn_reference(2);
  plan.mutable_extensions(1)->mutable_extension_function()->set_name("count");
  REQUIRE_THROWS(mo_decimal::normalize_exact_substrait(plan.SerializeAsString(), 4096));
}
