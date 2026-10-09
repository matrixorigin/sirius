/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
// This target links the production C ABI, not native_gpu_unittest's test backend.
#include "numeric/decimal_aggregate.hpp"
#include "numeric/exact_decimal.hpp"
#include "sirius_c.h"
#include "substrait/plan.pb.h"

#include <catch.hpp>

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace sirius::mo_decimal;
namespace {
void ok(sirius_status status, sirius_error const& error)
{
  INFO(error.message);
  REQUIRE(status == SIRIUS_OK);
}
coefficient small(int64_t value)
{
  return load_coefficient(reinterpret_cast<uint8_t const*>(&value), 8);
}
substrait::Type type(decimal_type value, bool nullable)
{
  substrait::Type result;
  auto t = result.mutable_user_defined();
  t->set_type_reference(11);
  t->set_nullability(nullable ? substrait::Type::NULLABILITY_NULLABLE
                              : substrait::Type::NULLABILITY_REQUIRED);
  for (int parameter : {int(value.bits), int(value.precision), int(value.scale)})
    t->add_type_parameters()->set_integer(parameter);
  return result;
}
substrait::Expression field(int index)
{
  substrait::Expression result;
  auto s = result.mutable_selection();
  s->mutable_direct_reference()->mutable_struct_field()->set_field(index);
  s->mutable_root_reference();
  return result;
}
substrait::Expression literal(decimal_type type, int64_t value)
{
  substrait::Expression result;
  auto u = result.mutable_literal()->mutable_user_defined();
  u->set_type_reference(11);
  for (int parameter : {int(type.bits), int(type.precision), int(type.scale)})
    u->add_type_parameters()->set_integer(parameter);
  auto any = u->mutable_value();
  any->set_type_url("type.googleapis.com/matrixone.sirius.numeric.v1.ExactDecimalLiteral");
  std::string payload(type.bytes() + 2, '\0');
  payload[0] = 10;
  payload[1] = type.bytes();
  store_coefficient(small(value), reinterpret_cast<uint8_t*>(payload.data() + 2), type.bytes());
  any->set_value(payload);
  return result;
}
substrait::Expression scalar(uint32_t anchor,
                             substrait::Type output,
                             std::vector<substrait::Expression> args)
{
  substrait::Expression result;
  auto f = result.mutable_scalar_function();
  f->set_function_reference(anchor);
  *f->mutable_output_type() = std::move(output);
  for (auto& arg : args)
    *f->add_arguments()->mutable_value() = std::move(arg);
  return result;
}
struct plan_case {
  substrait::Plan plan;
  decimal_type input, output;
  bool input_nullable{}, output_nullable{true};
  uint32_t output_oid{};
  plan_case(decimal_type input,
            decimal_type output,
            std::string name,
            bool aggregate,
            bool nullable = false)
    : input(input),
      output(output),
      input_nullable(nullable),
      output_oid(output.bits == 64    ? 32
                 : output.bits == 128 ? 33
                                      : 34)
  {
    plan.mutable_version()->set_minor_number(78);
    auto urn = plan.add_extension_urns();
    urn->set_extension_urn_anchor(1);
    urn->set_urn("urn:matrixone:sirius:exact-decimal:v1");
    auto t = plan.add_extensions()->mutable_extension_type();
    t->set_type_anchor(11);
    t->set_extension_urn_reference(1);
    t->set_name("mo_exact_decimal");
    function(21, name);
    auto root = plan.add_relations()->mutable_root();
    root->add_names("answer");
    substrait::Rel* source;
    if (aggregate) {
      auto a = root->mutable_input()->mutable_aggregate();
      source = a->mutable_input();
      auto m = a->add_measures()->mutable_measure();
      m->set_function_reference(21);
      m->set_phase(substrait::AGGREGATION_PHASE_INITIAL_TO_RESULT);
      m->set_invocation(substrait::AggregateFunction::AGGREGATION_INVOCATION_ALL);
      *m->mutable_output_type()            = type(output, true);
      *m->add_arguments()->mutable_value() = field(0);
    } else {
      auto p = root->mutable_input()->mutable_project();
      source = p->mutable_input();
      p->mutable_common()->mutable_emit()->add_output_mapping(1);
      *p->add_expressions() = scalar(21, type(output, true), {field(0), literal(input, 1)});
    }
    auto read = source->mutable_read();
    read->mutable_named_table()->add_names("__sirius_embedded_v1");
    read->mutable_named_table()->add_names("1");
    read->mutable_base_schema()->add_names("v");
    *read->mutable_base_schema()->mutable_struct_()->add_types() = type(input, nullable);
  }
  void function(uint32_t anchor, std::string const& name)
  {
    auto f = plan.add_extensions()->mutable_extension_function();
    f->set_function_anchor(anchor);
    f->set_extension_urn_reference(1);
    f->set_name(name);
  }
};
struct engine {
  sirius_engine_handle* handle{};
  engine()
  {
    auto config = std::getenv("MO_SIRIUS_TEST_CONFIG");
    REQUIRE(config);
    sirius_engine_options options{sizeof(options),
                                  SIRIUS_ABI_VERSION,
                                  config,
                                  static_cast<uint32_t>(std::strlen(config)),
                                  16,
                                  2,
                                  0};
    sirius_error error{};
    ok(sirius_engine_create(&options, &handle, &error), error);
  }
  ~engine()
  {
    sirius_error error{};
    if (handle) {
      auto status = sirius_engine_close(&handle, 30000, &error);
      if (status != SIRIUS_OK)
        std::cerr << "numeric engine cleanup: " << status << " " << error.message << '\n';
    }
  }
};
struct query {
  sirius_query_handle* handle{};
  sirius_input_handle* input{};
  sirius_batch_handle* batch{};
  std::vector<sirius_input_handle*> extra_inputs;
  std::string prepare_error;
  explicit query(engine& engine, plan_case const& plan)
  {
    extra_inputs.reserve(1);
    auto wire = plan.plan.SerializeAsString();
    sirius_error error{};
    sirius_query_options options{sizeof(options), SIRIUS_ABI_VERSION, 30000, 0};
    ok(sirius_query_create(engine.handle, &options, wire.data(), wire.size(), &handle, &error),
       error);
    try {
      sirius_column result{plan.output_oid,
                           plan.output_oid == 10 ? 0 : plan.output.precision,
                           plan.output_oid == 10 ? 0 : plan.output.scale,
                           plan.output_nullable,
                           "answer",
                           6,
                           0};
      sirius_query_contract contract{
        sizeof(contract), SIRIUS_ABI_VERSION, 0, 0, "numeric", 7, {}, &result, 1};
      ok(sirius_query_bind(handle, &contract, &error), error);
      sirius_column column{plan.input.bits == 64    ? 32u
                           : plan.input.bits == 128 ? 33u
                                                    : 34u,
                           plan.input.precision,
                           plan.input.scale,
                           plan.input_nullable,
                           "v",
                           1,
                           0};
      sirius_read_column physical{column, 7, 0, 0};
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
      ok(sirius_read_register(handle, &binding, &error), error);
      sirius_input_column schema{column.oid, column.width, column.scale, column.nullable};
      ok(sirius_input_register(handle, 1, &schema, 1, &input, &error), error);
    } catch (...) {
      if (input) sirius_input_close(&input, &error);
      sirius_query_close(&handle, 30000, &error);
      throw;
    }
  }
  ~query()
  {
    sirius_error error{};
    if (handle) sirius_query_cancel(handle, &error);
    if (batch) sirius_batch_release(&batch, &error);
    if (input) sirius_input_close(&input, &error);
    for (auto& extra : extra_inputs)
      if (extra) sirius_input_close(&extra, &error);
    if (handle) {
      auto status = sirius_query_close(&handle, 30000, &error);
      if (status != SIRIUS_OK)
        std::cerr << "numeric query cleanup: " << status << " " << error.message << '\n';
    }
  }
  sirius_status prepare()
  {
    sirius_error error{};
    auto status   = sirius_query_prepare(handle, 30000, &error);
    prepare_error = error.message;
    return status;
  }
  void publish(plan_case const& plan,
               std::vector<coefficient> const& values,
               bool last_null = false)
  {
    sirius_error error{};
    ok(sirius_query_start(handle, &error), error);
    write(input, plan.input, values, last_null);
  }
  void register_extra(decimal_type type, bool nullable)
  {
    sirius_error error{};
    sirius_column column{type.bits == 64    ? 32u
                         : type.bits == 128 ? 33u
                                            : 34u,
                         type.precision,
                         type.scale,
                         nullable,
                         "v",
                         1,
                         0};
    sirius_read_column physical{column, 8, 0, 0};
    sirius_read_binding binding{sizeof(binding),
                                SIRIUS_ABI_VERSION,
                                2,
                                SIRIUS_READ_MO,
                                0,
                                "db",
                                2,
                                "u",
                                1,
                                "s",
                                1,
                                &physical,
                                1,
                                nullptr,
                                0,
                                nullptr,
                                0};
    ok(sirius_read_register(handle, &binding, &error), error);
    sirius_input_column schema{column.oid, column.width, column.scale, column.nullable};
    extra_inputs.push_back(nullptr);
    ok(sirius_input_register(handle, 2, &schema, 1, &extra_inputs.back(), &error), error);
  }
  void write(sirius_input_handle* target,
             decimal_type type,
             std::vector<coefficient> const& values,
             bool last_null = false)
  {
    sirius_error error{};
    if (!values.empty()) {
      std::vector<uint8_t> data(values.size() * type.bytes());
      for (size_t i = 0; i < values.size(); ++i)
        store_coefficient(values[i], data.data() + i * type.bytes(), type.bytes());
      auto null_bytes = last_null ? 8u : 0u;
      uint64_t mask   = last_null ? uint64_t(1) << (values.size() - 1) : 0;
      ok(sirius_input_acquire(target, data.size() + null_bytes, 30000, &batch, &error), error);
      ok(sirius_input_write(batch, 0, data.data(), data.size(), &error), error);
      if (last_null) ok(sirius_input_write(batch, data.size(), &mask, 8, &error), error);
      sirius_input_vector vector{};
      vector.data_bytes  = data.size();
      vector.null_offset = data.size();
      vector.null_bytes  = null_bytes;
      ok(sirius_input_publish(target, &batch, values.size(), &vector, 1, &error), error);
    }
    ok(sirius_input_finish(target, &error), error);
  }
  std::vector<coefficient> results(uint32_t bytes, std::vector<bool>* nulls = nullptr)
  {
    std::vector<coefficient> values;
    sirius_error error{};
    for (;;) {
      auto code = sirius_query_next_result(handle, 30000, &batch, &error);
      if (code == SIRIUS_EOF) break;
      ok(code, error);
      sirius_result_batch_info info{sizeof(info), SIRIUS_ABI_VERSION};
      ok(sirius_result_describe(batch, &info, &error), error);
      REQUIRE(info.column_count == 1);
      std::vector<uint8_t> data(info.columns[0].data_bytes), validity(info.columns[0].null_bytes);
      ok(sirius_result_read(batch, info.columns[0].data_offset, data.data(), data.size(), &error),
         error);
      if (!validity.empty())
        ok(sirius_result_read(
             batch, info.columns[0].null_offset, validity.data(), validity.size(), &error),
           error);
      for (uint32_t row = 0; row < info.rows; ++row) {
        if (bytes == 1) {
          coefficient value;
          value.words[0] = data[row];
          values.push_back(value);
        } else
          values.push_back(load_coefficient(data.data() + row * bytes, bytes));
        if (nulls) nulls->push_back(!validity.empty() && (validity[row / 8] & (1u << (row % 8))));
      }
      ok(sirius_batch_release(&batch, &error), error);
    }
    ok(sirius_query_wait(handle, 30000, &error), error);
    return values;
  }
};
void equal(coefficient const& value, int64_t expected)
{
  auto reference = small(expected);
  for (int i = 0; i < 8; ++i)
    CHECK(value.words[i] == reference.words[i]);
}
}  // namespace
TEST_CASE("production ABI executes exact aggregate widths and empty NULL metadata",
          "[native_numeric]")
{
  CHECK(sirius_abi_version() == 1);
  CHECK((sirius_capabilities() & SIRIUS_CAP_MO_EXACT_DECIMAL_V1) != 0);
  engine engine;
  for (uint16_t bits : {64, 128, 256})
    for (auto name : {"mo_decimal_sum", "mo_decimal_avg", "mo_decimal_min", "mo_decimal_max"}) {
      INFO("bits=" << bits << " name=" << name);
      decimal_type input{bits, 15, 2}, output = input;
      if (std::string_view(name) == "mo_decimal_sum")
        output = {static_cast<uint16_t>(bits == 64 ? 128 : 256), 37, 2};
      if (std::string_view(name) == "mo_decimal_avg")
        output = {static_cast<uint16_t>(bits == 64 ? 128 : 256), 19, 6};
      for (bool empty : {false, true}) {
        plan_case plan(input, output, name, true);
        query run(engine, plan);
        auto status = run.prepare();
        INFO(run.prepare_error);
        REQUIRE(status == SIRIUS_OK);
        run.publish(
          plan,
          empty ? std::vector<coefficient>{} : std::vector<coefficient>{small(125), small(-25)});
        std::vector<bool> nulls;
        auto values = run.results(output.bytes(), &nulls);
        REQUIRE(values.size() == 1);
        REQUIRE(nulls.size() == 1);
        CHECK(nulls[0] == empty);
        if (!empty)
          equal(values[0],
                std::string_view(name) == "mo_decimal_sum"   ? 100
                : std::string_view(name) == "mo_decimal_avg" ? 500000
                : std::string_view(name) == "mo_decimal_min" ? -25
                                                             : 125);
        sirius_query_execution_stats stats{sizeof(stats), SIRIUS_ABI_VERSION};
        sirius_error error{};
        ok(sirius_query_get_execution_stats(run.handle, &stats, &error), error);
        CHECK(stats.gpu_tasks_started > 0);
        CHECK(stats.terminal_status == SIRIUS_OK);
        CHECK(stats.result_retained_charged_bytes == 0);
      }
    }
}
TEST_CASE("production exact ABI rejects raw numeric casts before GPU work", "[native_numeric]")
{
  engine engine;
  plan_case plan({64, 9, 2}, {64, 9, 2}, "mo_decimal_add", false);
  auto project = plan.plan.mutable_relations(0)->mutable_root()->mutable_input()->mutable_project();
  project->mutable_expressions(0)->Clear();
  auto cast              = project->mutable_expressions(0)->mutable_cast();
  *cast->mutable_input() = field(0);
  cast->mutable_type()->mutable_i64()->set_nullability(substrait::Type::NULLABILITY_REQUIRED);
  query run(engine, plan);
  CHECK(run.prepare() == SIRIUS_UNSUPPORTED);
  sirius_query_execution_stats stats{sizeof(stats), SIRIUS_ABI_VERSION};
  sirius_error error{};
  ok(sirius_query_get_execution_stats(run.handle, &stats, &error), error);
  CHECK(stats.gpu_tasks_started == 0);
  CHECK(stats.mo_input_units == 0);
}
TEST_CASE("production numeric statuses retain scalar aggregate and cast owners", "[native_numeric]")
{
  engine engine;
  auto magnitude = decimal_detail::power10(65);
  decimal_detail::magnitude one;
  one.words[0] = 1;
  decimal_detail::subtract(magnitude, one);
  auto maximum =
    decimal_detail::finish(magnitude, false, {256, 65, 0}, decimal_error::invalid_input).value;
  for (int mode : {0, 1, 2, 3}) {
    plan_case plan(mode == 3 ? decimal_type{64, 3, 0} : decimal_type{256, 65, 0},
                   mode == 3 ? decimal_type{64, 2, 0} : decimal_type{256, 65, 0},
                   mode == 3   ? "mo_decimal_cast"
                   : mode == 1 ? "mo_decimal_add"
                               : "mo_decimal_sum",
                   mode == 0 || mode == 2);
    if (mode == 2) {
      plan.function(22, "mo_decimal_add");
      auto measure = plan.plan.mutable_relations(0)
                       ->mutable_root()
                       ->mutable_input()
                       ->mutable_aggregate()
                       ->mutable_measures(0)
                       ->mutable_measure();
      *measure->mutable_arguments(0)->mutable_value() =
        scalar(22, type({256, 65, 0}, false), {field(0), literal({256, 65, 0}, 1)});
    }
    if (mode == 3)
      plan.plan.mutable_relations(0)
        ->mutable_root()
        ->mutable_input()
        ->mutable_project()
        ->mutable_expressions(0)
        ->mutable_scalar_function()
        ->mutable_arguments()
        ->RemoveLast();
    {
      query run(engine, plan);
      auto status = run.prepare();
      INFO(run.prepare_error);
      REQUIRE(status == SIRIUS_OK);
      run.publish(plan,
                  mode == 3 ? std::vector<coefficient>{small(999)}
                            : std::vector<coefficient>{maximum, small(1)});
      sirius_error error{};
      auto result = sirius_query_next_result(run.handle, 30000, &run.batch, &error);
      INFO(error.message);
      CHECK(result ==
            (mode == 0 || mode == 3 ? SIRIUS_NUMERIC_INVALID_INPUT : SIRIUS_NUMERIC_OUT_OF_RANGE));
      CHECK(run.batch == nullptr);
    }
    plan_case control({64, 15, 2}, {128, 37, 2}, "mo_decimal_sum", true);
    query recovered(engine, control);
    auto status = recovered.prepare();
    INFO(recovered.prepare_error);
    REQUIRE(status == SIRIUS_OK);
    recovered.publish(control, {small(125)});
    auto result = recovered.results(16);
    REQUIRE(result.size() == 1);
    equal(result[0], 125);
  }
}
TEST_CASE("production exact read filters use checked types before projection", "[native_numeric]")
{
  engine engine;
  for (uint16_t bits : {64, 128, 256}) {
    decimal_type domain{bits, 15, 2};
    plan_case plan(domain, domain, "mo_decimal_add", false);
    plan.function(22, "mo_decimal_less");
    auto read = plan.plan.mutable_relations(0)
                  ->mutable_root()
                  ->mutable_input()
                  ->mutable_project()
                  ->mutable_input()
                  ->mutable_read();
    substrait::Type boolean;
    boolean.mutable_bool_()->set_nullability(substrait::Type::NULLABILITY_REQUIRED);
    *read->mutable_filter() = scalar(22, boolean, {field(0), literal(domain, 200)});
    read->mutable_projection()->mutable_select()->add_struct_items()->set_field(0);
    {
      query run(engine, plan);
      auto status = run.prepare();
      INFO(run.prepare_error);
      REQUIRE(status == SIRIUS_OK);
      run.publish(plan, {small(125), small(999)});
      auto values = run.results(domain.bytes());
      REQUIRE(values.size() == 1);
      equal(values[0], 126);
    }
    auto urn = plan.plan.add_extension_urns();
    urn->set_extension_urn_anchor(2);
    urn->set_urn("extension:io.substrait:functions_comparison");
    auto ordinary = plan.plan.mutable_extensions(2)->mutable_extension_function();
    ordinary->set_extension_urn_reference(2);
    ordinary->set_name("lt");
    query rejected(engine, plan);
    CHECK(rejected.prepare() == SIRIUS_UNSUPPORTED);
    sirius_query_execution_stats stats{sizeof(stats), SIRIUS_ABI_VERSION};
    sirius_error error{};
    ok(sirius_query_get_execution_stats(rejected.handle, &stats, &error), error);
    CHECK(stats.gpu_tasks_started == 0);
    CHECK(stats.mo_input_units == 0);
  }
}
TEST_CASE("production exact CASE suppresses inactive failures and COALESCE retains types",
          "[native_numeric]")
{
  engine engine;
  for (uint16_t bits : {64, 128, 256}) {
    decimal_type domain{bits, 3, 0};
    plan_case plan(domain, domain, "mo_decimal_add", false);
    plan.output_nullable = false;
    plan.function(22, "mo_decimal_less");
    auto expression = plan.plan.mutable_relations(0)
                        ->mutable_root()
                        ->mutable_input()
                        ->mutable_project()
                        ->mutable_expressions(0);
    expression->Clear();
    auto c      = expression->mutable_if_then();
    auto branch = c->add_ifs();
    substrait::Type boolean;
    boolean.mutable_bool_()->set_nullability(substrait::Type::NULLABILITY_REQUIRED);
    *branch->mutable_if_()  = scalar(22, boolean, {field(0), literal(domain, 200)});
    *branch->mutable_then() = scalar(21, type(domain, false), {field(0), literal(domain, 1)});
    *c->mutable_else_()     = literal(domain, 9);
    {
      query run(engine, plan);
      auto status = run.prepare();
      INFO(run.prepare_error);
      REQUIRE(status == SIRIUS_OK);
      run.publish(plan, {small(999), small(125)});
      auto values = run.results(domain.bytes());
      REQUIRE(values.size() == 2);
      equal(values[0], 9);
      equal(values[1], 126);
    }
    plan_case coalesce(domain, domain, "coalesce", false, true);
    coalesce.output_nullable = false;
    auto urn                 = coalesce.plan.add_extension_urns();
    urn->set_extension_urn_anchor(2);
    urn->set_urn("extension:io.substrait:functions_boolean");
    coalesce.plan.mutable_extensions(1)->mutable_extension_function()->set_extension_urn_reference(
      2);
    auto f = coalesce.plan.mutable_relations(0)
               ->mutable_root()
               ->mutable_input()
               ->mutable_project()
               ->mutable_expressions(0)
               ->mutable_scalar_function();
    *f->mutable_output_type()                 = type(domain, false);
    *f->mutable_arguments(1)->mutable_value() = literal(domain, 9);
    {
      query run(engine, coalesce);
      auto status = run.prepare();
      INFO(run.prepare_error);
      REQUIRE(status == SIRIUS_OK);
      run.publish(coalesce, {small(999), small(125)}, true);
      auto values = run.results(domain.bytes());
      REQUIRE(values.size() == 2);
      equal(values[0], 999);
      equal(values[1], 9);
    }
    auto lower = coalesce.plan.relations(0).root().input();
    auto upper =
      coalesce.plan.mutable_relations(0)->mutable_root()->mutable_input()->mutable_project();
    upper->Clear();
    *upper->mutable_input() = std::move(lower);
    upper->mutable_common()->mutable_emit()->add_output_mapping(1);
    coalesce.output = {bits, 4, 0};
    coalesce.function(23, "mo_decimal_add");
    *upper->add_expressions() =
      scalar(23, type(coalesce.output, false), {field(0), literal(domain, 1)});
    query projected(engine, coalesce);
    auto status = projected.prepare();
    INFO(projected.prepare_error);
    REQUIRE(status == SIRIUS_OK);
    projected.publish(coalesce, {small(999), small(125)}, true);
    auto values = projected.results(domain.bytes());
    REQUIRE(values.size() == 2);
    equal(values[0], 1000);
    equal(values[1], 10);
  }
}
TEST_CASE("production exact join keys and outer NULLs preserve values across scales",
          "[native_numeric]")
{
  engine engine;
  for (bool outer : {false, true}) {
    decimal_type left{64, 15, 2}, right{256, 15, 3};
    plan_case plan(left, outer ? right : left, "mo_decimal_equal", false);
    plan.output_nullable = outer;
    auto project =
      plan.plan.mutable_relations(0)->mutable_root()->mutable_input()->mutable_project();
    project->clear_expressions();
    *project->add_expressions() = field(outer ? 1 : 0);
    project->mutable_common()->mutable_emit()->set_output_mapping(0, 2);
    auto read              = project->input();
    auto join              = project->mutable_input()->mutable_join();
    *join->mutable_left()  = read;
    *join->mutable_right() = read;
    join->mutable_right()->mutable_read()->mutable_named_table()->set_names(1, "2");
    *join->mutable_right()->mutable_read()->mutable_base_schema()->mutable_struct_()->mutable_types(
      0) = type(right, false);
    join->set_type(outer ? substrait::JoinRel::JOIN_TYPE_LEFT
                         : substrait::JoinRel::JOIN_TYPE_INNER);
    substrait::Type boolean;
    boolean.mutable_bool_()->set_nullability(substrait::Type::NULLABILITY_REQUIRED);
    *join->mutable_expression() = scalar(21, boolean, {field(0), field(1)});
    query run(engine, plan);
    run.register_extra(right, false);
    auto status = run.prepare();
    INFO(run.prepare_error);
    REQUIRE(status == SIRIUS_OK);
    run.publish(plan, {small(125), small(500)});
    run.write(run.extra_inputs[0], right, {small(1250)});
    std::vector<bool> nulls;
    auto values = run.results((outer ? right : left).bytes(), &nulls);
    REQUIRE(values.size() == (outer ? 2 : 1));
    int valid = 0, missing = 0;
    for (size_t i = 0; i < values.size(); ++i)
      if (nulls[i])
        ++missing;
      else {
        equal(values[i], outer ? 1250 : 125);
        ++valid;
      }
    CHECK(valid == 1);
    CHECK(missing == (outer ? 1 : 0));
  }
}
TEST_CASE("production exact true joins preserve multiplicity payloads and empty outer sides",
          "[native_numeric]")
{
  engine engine;
  decimal_type domain{256, 65, 0};
  coefficient high;
  high.words[4] = 1;  // 2^128: passthrough must retain the high limbs.
  for (auto kind : {substrait::JoinRel::JOIN_TYPE_INNER,
                    substrait::JoinRel::JOIN_TYPE_LEFT,
                    substrait::JoinRel::JOIN_TYPE_RIGHT,
                    substrait::JoinRel::JOIN_TYPE_OUTER}) {
    for (bool left_empty : {false, true}) {
      for (bool right_empty : {false, true}) {
        INFO("kind=" << kind << " left_empty=" << left_empty << " right_empty=" << right_empty);
        plan_case plan(domain, domain, "mo_decimal_equal", false, true);
        auto project =
          plan.plan.mutable_relations(0)->mutable_root()->mutable_input()->mutable_project();
        project->clear_expressions();
        *project->add_expressions() = field(1);
        project->mutable_common()->mutable_emit()->set_output_mapping(0, 2);
        auto read              = project->input();
        auto join              = project->mutable_input()->mutable_join();
        *join->mutable_left()  = read;
        *join->mutable_right() = read;
        join->mutable_right()->mutable_read()->mutable_named_table()->set_names(1, "2");
        join->set_type(kind);
        join->mutable_expression()->mutable_literal()->set_boolean(true);
        query run(engine, plan);
        run.register_extra(domain, true);
        auto status = run.prepare();
        INFO(run.prepare_error);
        REQUIRE(status == SIRIUS_OK);
        run.publish(
          plan,
          left_empty ? std::vector<coefficient>{} : std::vector<coefficient>{small(1), small(2)});
        run.write(
          run.extra_inputs[0],
          domain,
          right_empty ? std::vector<coefficient>{} : std::vector<coefficient>{high, high, small(0)},
          !right_empty);
        std::vector<bool> nulls;
        auto values = run.results(domain.bytes(), &nulls);
        bool preserve_left =
          kind == substrait::JoinRel::JOIN_TYPE_LEFT || kind == substrait::JoinRel::JOIN_TYPE_OUTER;
        bool preserve_right = kind == substrait::JoinRel::JOIN_TYPE_RIGHT ||
                              kind == substrait::JoinRel::JOIN_TYPE_OUTER;
        size_t rows  = left_empty || right_empty ? 0 : 6;
        size_t valid = left_empty || right_empty ? 0 : 4;
        if (right_empty && !left_empty && preserve_left) rows = 2;
        if (left_empty && !right_empty && preserve_right) {
          rows  = 3;
          valid = 2;
        }
        REQUIRE(values.size() == rows);
        size_t observed_valid = 0;
        for (size_t i = 0; i < values.size(); ++i) {
          if (nulls[i]) continue;
          ++observed_valid;
          for (int word = 0; word < 8; ++word)
            CHECK(values[i].words[word] == high.words[word]);
        }
        CHECK(observed_valid == valid);
      }
    }
  }
}
TEST_CASE("production exact true-join lowering does not admit FALSE or NULL predicates",
          "[native_numeric]")
{
  engine engine;
  decimal_type domain{256, 65, 0};
  for (bool null_predicate : {false, true}) {
    plan_case plan(domain, domain, "mo_decimal_equal", false, true);
    auto project =
      plan.plan.mutable_relations(0)->mutable_root()->mutable_input()->mutable_project();
    project->clear_expressions();
    *project->add_expressions() = field(1);
    project->mutable_common()->mutable_emit()->set_output_mapping(0, 2);
    auto read              = project->input();
    auto join              = project->mutable_input()->mutable_join();
    *join->mutable_left()  = read;
    *join->mutable_right() = read;
    join->mutable_right()->mutable_read()->mutable_named_table()->set_names(1, "2");
    join->set_type(substrait::JoinRel::JOIN_TYPE_LEFT);
    auto predicate = join->mutable_expression()->mutable_literal();
    if (null_predicate)
      predicate->mutable_null()->mutable_bool_()->set_nullability(
        substrait::Type::NULLABILITY_NULLABLE);
    else
      predicate->set_boolean(false);
    query run(engine, plan);
    run.register_extra(domain, true);
    REQUIRE(run.prepare() != SIRIUS_OK);
    sirius_query_execution_stats stats{sizeof(stats), SIRIUS_ABI_VERSION};
    sirius_error error{};
    ok(sirius_query_get_execution_stats(run.handle, &stats, &error), error);
    CHECK(stats.gpu_tasks_started == 0);
    CHECK(stats.mo_input_units == 0);
  }
}
TEST_CASE("production Decimal256 ordering retains signed coefficients and NULL position",
          "[native_numeric]")
{
  engine engine;
  decimal_type domain{256, 15, 2};
  plan_case plan(domain, domain, "mo_decimal_add", false, true);
  auto root                        = plan.plan.mutable_relations(0)->mutable_root();
  auto project                     = root->mutable_input()->mutable_project();
  *project->mutable_expressions(0) = field(0);
  auto input                       = root->input();
  auto sort                        = root->mutable_input()->mutable_sort();
  *sort->mutable_input()           = std::move(input);
  auto key                         = sort->add_sorts();
  *key->mutable_expr()             = field(0);
  key->set_direction(substrait::SortField::SORT_DIRECTION_DESC_NULLS_LAST);
  query run(engine, plan);
  auto status = run.prepare();
  INFO(run.prepare_error);
  REQUIRE(status == SIRIUS_OK);
  run.publish(plan, {small(-125), small(500), small(99)}, true);
  std::vector<bool> nulls;
  auto values = run.results(32, &nulls);
  REQUIRE(values.size() == 3);
  equal(values[0], 500);
  equal(values[1], -125);
  CHECK(nulls[2]);
}
TEST_CASE("production grouped exact and ordinary slots finalize through the pipeline",
          "[native_numeric]")
{
  engine engine;
  for (bool count : {false, true})
    for (bool ordered : {false, true}) {
      decimal_type input{256, 15, 2}, output{256, 37, 2};
      plan_case plan(input, output, "mo_decimal_sum", true);
      auto root                              = plan.plan.mutable_relations(0)->mutable_root();
      auto aggregate                         = root->mutable_input()->mutable_aggregate();
      *aggregate->add_grouping_expressions() = field(0);
      aggregate->add_groupings()->add_expression_references(0);
      plan.function(22, "count");
      plan.plan.mutable_extensions(2)->mutable_extension_function()->set_extension_urn_reference(2);
      auto urn = plan.plan.add_extension_urns();
      urn->set_extension_urn_anchor(2);
      urn->set_urn("extension:io.substrait:functions_aggregate_generic");
      auto ordinary = aggregate->add_measures()->mutable_measure();
      ordinary->set_function_reference(22);
      ordinary->set_phase(substrait::AGGREGATION_PHASE_INITIAL_TO_RESULT);
      ordinary->set_invocation(substrait::AggregateFunction::AGGREGATION_INVOCATION_ALL);
      ordinary->mutable_output_type()->mutable_i64()->set_nullability(
        substrait::Type::NULLABILITY_REQUIRED);
      auto grouped    = root->input();
      auto projection = root->mutable_input()->mutable_project();
      projection->mutable_common()->mutable_emit()->add_output_mapping(3);
      *projection->add_expressions() = field(count ? 2 : 1);
      if (ordered) {
        auto sort              = projection->mutable_input()->mutable_sort();
        *sort->mutable_input() = std::move(grouped);
        auto key               = sort->add_sorts();
        *key->mutable_expr()   = field(0);
        key->set_direction(substrait::SortField::SORT_DIRECTION_DESC_NULLS_LAST);
      } else
        *projection->mutable_input() = std::move(grouped);
      if (count) {
        plan.output_oid      = 23;
        plan.output          = {};
        plan.output_nullable = false;
      }
      query run(engine, plan);
      auto status = run.prepare();
      INFO(run.prepare_error);
      REQUIRE(status == SIRIUS_OK);
      run.publish(plan, {small(125), small(-25), small(125)});
      auto values = run.results(count ? 8 : 32);
      REQUIRE(values.size() == 2);
      if (ordered) {
        equal(values[0], count ? 2 : 250);
        equal(values[1], count ? 1 : -25);
      } else {
        std::sort(values.begin(), values.end(), [](auto const& a, auto const& b) {
          return compare_coefficient(a, b) < 0;
        });
        equal(values[0], count ? 1 : -25);
        equal(values[1], count ? 2 : 250);
      }
    }
}
