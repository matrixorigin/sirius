/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "numeric/decimal_functions.hpp"

#include "expression/ast/node.hpp"
#include "helper/type_conversions.hpp"
#include "numeric/decimal_plan.hpp"
#include "numeric/exact_decimal.hpp"

#include <duckdb/catalog/catalog.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>

namespace sirius::mo_decimal {
bool is_decimal_function(function_id id) noexcept
{
  return id >= function_id::mo_decimal_add && id <= function_id::mo_decimal_greater_equal;
}
decimal_op operation(function_id id)
{
  if (!is_decimal_function(id)) throw std::invalid_argument("not an MO decimal operation");
  return static_cast<decimal_op>(static_cast<uint16_t>(id) -
                                 static_cast<uint16_t>(function_id::mo_decimal_add));
}
function_id function(decimal_op op)
{
  if (op > decimal_op::greater_equal) throw std::invalid_argument("unknown MO decimal operation");
  return static_cast<function_id>(static_cast<uint16_t>(function_id::mo_decimal_add) +
                                  static_cast<uint8_t>(op));
}
bool contains_exact_expression(ast::node const& expression)
{
  if (expression.return_type().is_mo_decimal()) return true;
  return std::visit(
    [](auto const& node) {
      auto child = [](auto const& p) { return p && contains_exact_expression(*p); };
      if constexpr (requires {
                      node.function();
                      node.arguments();
                    }) {
        if constexpr (std::is_same_v<std::decay_t<decltype(node)>, ast::function_call>)
          if (is_decimal_function(node.function())) return true;
        for (auto const& argument : node.arguments())
          if (child(argument)) return true;
      } else if constexpr (requires {
                             node.cases;
                             node.else_;
                           }) {
        if (child(node.else_)) return true;
        for (auto const& entry : node.cases)
          if (child(entry.when_) || child(entry.then_)) return true;
      } else if constexpr (requires { node.children; }) {
        for (auto const& argument : node.children)
          if (child(argument)) return true;
      } else if constexpr (requires { node.child; })
        return child(node.child);
      else if constexpr (requires {
                           node.left;
                           node.right;
                         })
        return child(node.left) || child(node.right);
      else if constexpr (requires {
                           node.input;
                           node.lower;
                           node.upper;
                         })
        return child(node.input) || child(node.lower) || child(node.upper);
      else if constexpr (requires {
                           node.probe;
                           node.values;
                         }) {
        if (child(node.probe)) return true;
        for (auto const& value : node.values)
          if (child(value)) return true;
      }
      return false;
    },
    expression.v);
}
namespace {
bool signed_integer_cast_fits(logical_type const& input, logical_type const& output)
{
  unsigned bits;
  switch (input.id()) {
    case type_id::TINYINT: bits = 8; break;
    case type_id::SMALLINT: bits = 16; break;
    case type_id::INTEGER: bits = 32; break;
    case type_id::BIGINT: bits = 64; break;
    default: return false;
  }
  uint64_t endpoints[]{~uint64_t(0) << (bits - 1), (uint64_t(1) << (bits - 1)) - 1};
  for (auto value : endpoints) {
    auto coefficient = load_coefficient(reinterpret_cast<uint8_t const*>(&value), 8);
    if (evaluate_decimal(decimal_op::cast,
                         coefficient,
                         {64, 18, 0},
                         true,
                         {},
                         {64, 18, 0},
                         true,
                         output.mo_decimal_type())
          .error != decimal_error::none)
      return false;
  }
  return true;
}
struct scalar_data final : duckdb::FunctionData {
  decimal_op op;
  duckdb::vector<duckdb::LogicalType> inputs;
  duckdb::LogicalType output;
  scalar_data(decimal_op o, duckdb::vector<duckdb::LogicalType> i, duckdb::LogicalType r)
    : op(o), inputs(std::move(i)), output(std::move(r))
  {
  }
  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override
  {
    return duckdb::make_uniq<scalar_data>(op, inputs, output);
  }
  bool Equals(duckdb::FunctionData const& other) const override
  {
    auto data = dynamic_cast<scalar_data const*>(&other);
    return data && op == data->op && inputs == data->inputs && output == data->output;
  }
};
void never_execute(duckdb::DataChunk&, duckdb::ExpressionState&, duckdb::Vector&)
{
  throw duckdb::NotImplementedException("MO exact-decimal marker cannot execute in DuckDB");
}
duckdb::unique_ptr<duckdb::FunctionData> bind_scalar(
  duckdb::ClientContext&,
  duckdb::ScalarFunction& fn,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& arguments)
{
  auto id = from_duckdb_function_name(fn.name);
  if (!id || !is_decimal_function(*id) || arguments.empty() ||
      arguments.back()->GetExpressionClass() != duckdb::ExpressionClass::BOUND_CONSTANT)
    throw duckdb::InvalidInputException("invalid MO decimal binding marker");
  auto const& marker = arguments.back()->Cast<duckdb::BoundConstantExpression>().value;
  auto output        = from_duckdb_type(marker.type());
  if (!marker.IsNull() || !output)
    throw duckdb::InvalidInputException("MO decimal binding needs its declared output type");
  std::vector<logical_type> inputs;
  duckdb::vector<duckdb::LogicalType> carriers;
  for (std::size_t i = 0; i + 1 < arguments.size(); ++i) {
    restore_exact_bound_types(arguments[i]);
    inputs.push_back(sirius::from_duckdb(arguments[i]->return_type));
    carriers.push_back(arguments[i]->return_type);
  }
  validate_signature(operation(*id), inputs, *output);
  fn.return_type = marker.type();
  arguments.pop_back();
  fn.arguments = carriers;
  return duckdb::make_uniq<scalar_data>(operation(*id), std::move(carriers), fn.return_type);
}
duckdb::ScalarFunction scalar_function(decimal_op op)
{
  auto arity = op == decimal_op::cast || op == decimal_op::negate ? 1u : 2u;
  duckdb::vector<duckdb::LogicalType> arguments(arity + 1, duckdb::LogicalType::ANY);
  return duckdb::ScalarFunction(std::string(to_duckdb_function_name(function(op))),
                                std::move(arguments),
                                duckdb::LogicalType::ANY,
                                never_execute,
                                bind_scalar,
                                nullptr,
                                nullptr,
                                nullptr,
                                duckdb::LogicalType::INVALID,
                                duckdb::FunctionStability::VOLATILE,
                                duckdb::FunctionNullHandling::SPECIAL_HANDLING);
}
}  // namespace

void validate_signature(decimal_op op,
                        std::vector<logical_type> const& inputs,
                        logical_type const& output)
{
  bool unary = op == decimal_op::cast || op == decimal_op::negate;
  if (op > decimal_op::greater_equal || inputs.size() != (unary ? 1u : 2u))
    throw std::invalid_argument("invalid MO decimal operation arity");
  if (op >= decimal_op::equal || op == decimal_op::integer_divide) {
    if (output.id() != (op >= decimal_op::equal ? type_id::BOOLEAN : type_id::BIGINT) ||
        !output.nullability())
      throw std::invalid_argument("invalid MO decimal comparison or DIV result");
  } else if (!output.is_mo_decimal())
    throw std::invalid_argument("MO decimal operation requires an exact result descriptor");
  if (op == decimal_op::cast && !inputs[0].is_mo_decimal()) {
    if (!signed_integer_cast_fits(inputs[0], output))
      throw std::invalid_argument("MO decimal integer cast does not contain the full input domain");
    return;
  }
  for (auto const& input : inputs)
    if (!input.is_mo_decimal())
      throw std::invalid_argument("MO decimal operands must retain exact type descriptors");
  if (op == decimal_op::negate) {
    auto a = inputs[0].mo_decimal_type(), r = output.mo_decimal_type();
    if (r.bits < a.bits || r.precision < a.precision || r.scale != a.scale)
      throw std::invalid_argument("MO decimal negation is not proven domain-preserving");
  }
}
void register_scalar_functions(duckdb::DatabaseInstance& instance)
{
  auto transaction = duckdb::CatalogTransaction::GetSystemTransaction(instance);
  auto& catalog    = duckdb::Catalog::GetSystemCatalog(instance);
  for (unsigned i = 0; i <= static_cast<unsigned>(decimal_op::greater_equal); ++i) {
    duckdb::CreateScalarFunctionInfo info(scalar_function(static_cast<decimal_op>(i)));
    catalog.CreateFunction(transaction, info);
  }
}
void validate_bound_scalar(duckdb::BoundFunctionExpression const& expression)
{
  auto data = dynamic_cast<scalar_data const*>(expression.bind_info.get());
  auto id   = from_duckdb_function_name(expression.function.name);
  if (!data || !id || !is_decimal_function(*id) || data->op != operation(*id) ||
      data->output != expression.return_type || data->inputs.size() != expression.children.size() ||
      expression.function.GetBindCallback() != bind_scalar)
    throw std::invalid_argument("MO decimal bound function lost its checked signature");
  std::vector<logical_type> inputs;
  for (std::size_t i = 0; i < data->inputs.size(); ++i) {
    if (data->inputs[i] != expression.children[i]->return_type)
      throw std::invalid_argument("MO decimal bound operand type mismatch");
    inputs.push_back(sirius::from_duckdb(data->inputs[i]));
  }
  validate_signature(data->op, inputs, sirius::from_duckdb(data->output));
}
duckdb::unique_ptr<duckdb::Expression> bound_scalar(
  decimal_op op,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children,
  logical_type const& output)
{
  duckdb::vector<duckdb::LogicalType> carriers;
  std::vector<logical_type> inputs;
  for (auto const& child : children) {
    carriers.push_back(child->return_type);
    inputs.push_back(sirius::from_duckdb(child->return_type));
  }
  validate_signature(op, inputs, output);
  auto fn          = scalar_function(op);
  fn.arguments     = carriers;
  fn.return_type   = sirius::to_duckdb(output);
  auto data        = duckdb::make_uniq<scalar_data>(op, std::move(carriers), fn.return_type);
  auto result_type = fn.return_type;
  return duckdb::make_uniq<duckdb::BoundFunctionExpression>(
    std::move(result_type), std::move(fn), std::move(children), std::move(data));
}
}  // namespace sirius::mo_decimal
