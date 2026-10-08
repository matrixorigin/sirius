/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "decimal_import_fixture.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/to_duckdb.hpp"
#include "expression/ast/utils.hpp"
#include "expression/value.hpp"
#include "helper/type_conversions.hpp"
#include "numeric/decimal_types.hpp"
#include "numeric/exact_decimal.hpp"

#include <catch.hpp>

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
