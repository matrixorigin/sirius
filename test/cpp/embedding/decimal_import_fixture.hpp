/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "expression/ast/from_duckdb.hpp"
#include "from_substrait.hpp"
#include "numeric/decimal_aggregate_bind.hpp"
#include "numeric/decimal_functions.hpp"
#include "numeric/decimal_import.hpp"
#include "numeric/exact_decimal.hpp"

#include <core_functions_extension.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/parser/statement/relation_statement.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/planner.hpp>

namespace decimal_fixture {
inline sirius::mo_decimal::coefficient small(int64_t value)
{
  return sirius::mo_decimal::load_coefficient(reinterpret_cast<uint8_t const*>(&value), 8);
}
template <class User>
void parameters(User* value, sirius::mo_decimal::decimal_type type)
{
  for (auto item : {int(type.bits), int(type.precision), int(type.scale)})
    value->add_type_parameters()->set_integer(item);
}
inline substrait::Type type(sirius::mo_decimal::decimal_type descriptor, bool nullable = false)
{
  substrait::Type result;
  auto user = result.mutable_user_defined();
  user->set_type_reference(11);
  user->set_nullability(nullable ? substrait::Type::NULLABILITY_NULLABLE
                                 : substrait::Type::NULLABILITY_REQUIRED);
  parameters(user, descriptor);
  return result;
}
inline substrait::Expression literal(sirius::mo_decimal::decimal_type type,
                                     sirius::mo_decimal::coefficient const& value)
{
  substrait::Expression result;
  auto user = result.mutable_literal()->mutable_user_defined();
  user->set_type_reference(11);
  parameters(user, type);
  user->mutable_value()->set_type_url(sirius::mo_decimal::literal_type_url);
  std::string payload(type.bytes() + 2, '\0');
  payload[0] = 10;
  payload[1] = type.bytes();
  sirius::mo_decimal::store_coefficient(
    value, reinterpret_cast<uint8_t*>(payload.data() + 2), type.bytes());
  user->mutable_value()->set_value(payload);
  return result;
}
inline substrait::Expression literal(sirius::mo_decimal::decimal_type type, int64_t value)
{
  return literal(type, small(value));
}
inline substrait::Plan plan(std::string name,
                            substrait::Type output,
                            std::vector<substrait::Expression> arguments)
{
  substrait::Plan result;
  result.mutable_version()->set_minor_number(78);
  auto urn = result.add_extension_urns();
  urn->set_extension_urn_anchor(1);
  urn->set_urn(sirius::mo_decimal::extension_uri);
  auto type = result.add_extensions()->mutable_extension_type();
  type->set_type_anchor(11);
  type->set_extension_urn_reference(1);
  type->set_name("mo_exact_decimal");
  auto fn = result.add_extensions()->mutable_extension_function();
  fn->set_function_anchor(21);
  fn->set_extension_urn_reference(1);
  fn->set_name(name);
  auto root = result.add_relations()->mutable_root();
  root->add_names("exact");
  auto project = root->mutable_input()->mutable_project();
  project->mutable_common()->mutable_emit()->add_output_mapping(1);
  auto read = project->mutable_input()->mutable_read();
  read->mutable_base_schema()->add_names("seed");
  read->mutable_base_schema()->mutable_struct_()->add_types()->mutable_i64()->set_nullability(
    substrait::Type::NULLABILITY_REQUIRED);
  read->mutable_virtual_table()->add_expressions()->add_fields()->mutable_literal()->set_i64(1);
  auto scalar = project->add_expressions()->mutable_scalar_function();
  scalar->set_function_reference(21);
  *scalar->mutable_output_type() = std::move(output);
  for (auto& argument : arguments)
    *scalar->add_arguments()->mutable_value() = std::move(argument);
  return result;
}
class importer {
  static std::unique_ptr<duckdb::DBConfig> config()
  {
    auto result                     = std::make_unique<duckdb::DBConfig>();
    result->options.load_extensions = false;
    return result;
  }
  std::unique_ptr<duckdb::DBConfig> config_{config()};

 public:
  duckdb::DuckDB db{nullptr, config_.get()};
  duckdb::Connection connection{db};
  importer()
  {
    db.LoadStaticExtension<duckdb::CoreFunctionsExtension>();
    sirius::mo_decimal::register_scalar_functions(*db.instance);
    sirius::mo_decimal::register_aggregate_functions(*db.instance);
  }
  std::unique_ptr<sirius::ast::node> bind(substrait::Plan const& plan)
  {
    auto bytes = plan.SerializeAsString();
    connection.BeginTransaction();
    try {
      duckdb::SubstraitToDuckDB converter(
        connection.context, bytes, false, false, sirius::mo_decimal::make_import_handler(bytes));
      auto relation = converter.TransformPlan();
      duckdb::Planner planner(*connection.context);
      planner.CreatePlan(duckdb::make_uniq<duckdb::RelationStatement>(relation));
      planner.plan->ResolveOperatorTypes();
      duckdb::ColumnBindingResolver resolver;
      resolver.VisitOperator(*planner.plan);
      std::unique_ptr<sirius::ast::node> found;
      std::function<void(duckdb::Expression const&)> expression;
      expression = [&](duckdb::Expression const& value) {
        if (found) return;
        if (value.GetExpressionClass() == duckdb::ExpressionClass::BOUND_AGGREGATE) {
          auto const& aggregate = value.Cast<duckdb::BoundAggregateExpression>();
          auto id               = sirius::from_duckdb_aggregate_name(aggregate.function.name);
          if (id && sirius::mo_decimal::is_decimal_aggregate(*id)) {
            found = sirius::ast::from_duckdb(value);
            return;
          }
        }
        if (value.GetExpressionClass() == duckdb::ExpressionClass::BOUND_FUNCTION) {
          auto const& function = value.Cast<duckdb::BoundFunctionExpression>();
          auto id              = sirius::from_duckdb_function_name(function.function.name);
          if (id && sirius::mo_decimal::is_decimal_function(*id)) {
            found = sirius::ast::from_duckdb(value);
            return;
          }
        }
        duckdb::ExpressionIterator::EnumerateChildren(value, expression);
      };
      std::function<void(duckdb::LogicalOperator const&)> visit = [&](auto const& op) {
        for (auto const& value : op.expressions)
          expression(*value);
        for (auto const& child : op.children)
          visit(*child);
      };
      visit(*planner.plan);
      if (!found) throw std::runtime_error("numeric function disappeared during import");
      connection.Rollback();
      return found;
    } catch (...) {
      if (connection.HasActiveTransaction()) connection.Rollback();
      throw;
    }
  }
};
}  // namespace decimal_fixture
