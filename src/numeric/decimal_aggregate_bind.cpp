/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "numeric/decimal_aggregate_bind.hpp"

#include "helper/type_conversions.hpp"
#include "numeric/decimal_plan.hpp"

#include <duckdb/catalog/catalog.hpp>
#include <duckdb/function/aggregate_function.hpp>
#include <duckdb/parser/parsed_data/create_aggregate_function_info.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>

namespace sirius::mo_decimal {
bool is_decimal_aggregate(aggregate_id id) noexcept
{
  return id >= aggregate_id::mo_decimal_sum && id <= aggregate_id::mo_decimal_max;
}
aggregate_op aggregate_operation(aggregate_id id)
{
  if (!is_decimal_aggregate(id)) throw std::invalid_argument("not an MO decimal aggregate");
  return static_cast<aggregate_op>(static_cast<uint16_t>(id) -
                                   static_cast<uint16_t>(aggregate_id::mo_decimal_sum));
}
aggregate_id aggregate_function(aggregate_op op)
{
  if (op > aggregate_op::max) throw std::invalid_argument("unknown MO decimal aggregate");
  return static_cast<aggregate_id>(static_cast<uint16_t>(aggregate_id::mo_decimal_sum) +
                                   static_cast<uint8_t>(op));
}
decimal_type aggregate_input_type(logical_type const& input)
{
  if (input.is_mo_decimal()) return input.mo_decimal_type();
  if (input.id() == type_id::BIGINT) return {64, 18, 0};
  throw std::invalid_argument("unsupported MO decimal aggregate operand");
}
void validate_aggregate_signature(aggregate_op op,
                                  logical_type const& input,
                                  logical_type const& output)
{
  if (op > aggregate_op::max || !output.is_mo_decimal())
    throw std::invalid_argument("MO aggregate requires an exact result descriptor");
  auto a = aggregate_input_type(input), r = output.mo_decimal_type();
  if (r.precision > 65) throw std::invalid_argument("MO aggregate result exceeds public precision");
  if (op == aggregate_op::min || op == aggregate_op::max) {
    if (!input.is_mo_decimal() || a.bits != r.bits || a.precision != r.precision ||
        a.scale != r.scale)
      throw std::invalid_argument("MO MIN/MAX must preserve the argument descriptor");
  } else if (r.bits < 128 || r.bits < a.bits || (op == aggregate_op::sum && r.scale != a.scale) ||
             (op == aggregate_op::avg && r.scale < a.scale))
    throw std::invalid_argument("invalid MO SUM/AVG result descriptor");
}
namespace {
struct aggregate_data final : duckdb::FunctionData {
  aggregate_op op;
  duckdb::LogicalType input, output;
  aggregate_data(aggregate_op o, duckdb::LogicalType a, duckdb::LogicalType r)
    : op(o), input(std::move(a)), output(std::move(r))
  {
  }
  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override
  {
    return duckdb::make_uniq<aggregate_data>(op, input, output);
  }
  bool Equals(duckdb::FunctionData const& other) const override
  {
    auto data = dynamic_cast<aggregate_data const*>(&other);
    return data && op == data->op && input == data->input && output == data->output;
  }
};
[[noreturn]] void never_execute()
{
  throw duckdb::NotImplementedException("MO exact-decimal aggregate cannot execute in DuckDB");
}
duckdb::idx_t state_size(duckdb::AggregateFunction const&) { return 1; }
void initialize(duckdb::AggregateFunction const&, duckdb::data_ptr_t) { never_execute(); }
void update(
  duckdb::Vector[], duckdb::AggregateInputData&, duckdb::idx_t, duckdb::Vector&, duckdb::idx_t)
{
  never_execute();
}
void simple_update(
  duckdb::Vector[], duckdb::AggregateInputData&, duckdb::idx_t, duckdb::data_ptr_t, duckdb::idx_t)
{
  never_execute();
}
void combine(duckdb::Vector&, duckdb::Vector&, duckdb::AggregateInputData&, duckdb::idx_t)
{
  never_execute();
}
void finalize(
  duckdb::Vector&, duckdb::AggregateInputData&, duckdb::Vector&, duckdb::idx_t, duckdb::idx_t)
{
  never_execute();
}
duckdb::unique_ptr<duckdb::FunctionData> bind(
  duckdb::ClientContext&,
  duckdb::AggregateFunction& fn,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& arguments)
{
  auto id = from_duckdb_aggregate_name(fn.name);
  if (!id || !is_decimal_aggregate(*id) || arguments.size() != 2 ||
      arguments.back()->GetExpressionClass() != duckdb::ExpressionClass::BOUND_CONSTANT)
    throw duckdb::InvalidInputException("invalid MO aggregate binding marker");
  auto const& marker = arguments.back()->Cast<duckdb::BoundConstantExpression>().value;
  auto output        = from_duckdb_type(marker.type());
  if (!marker.IsNull() || !output)
    throw duckdb::InvalidInputException("MO aggregate lost its output marker");
  restore_exact_bound_types(arguments[0]);
  validate_aggregate_signature(
    aggregate_operation(*id), sirius::from_duckdb(arguments[0]->return_type), *output);
  fn.return_type = marker.type();
  arguments.pop_back();
  fn.arguments = {arguments[0]->return_type};
  return duckdb::make_uniq<aggregate_data>(
    aggregate_operation(*id), fn.arguments[0], fn.return_type);
}
duckdb::AggregateFunction make_function(aggregate_op op)
{
  duckdb::AggregateFunction fn(std::string(to_duckdb_aggregate_name(aggregate_function(op))),
                               {duckdb::LogicalType::ANY, duckdb::LogicalType::ANY},
                               duckdb::LogicalType::ANY,
                               state_size,
                               initialize,
                               update,
                               combine,
                               finalize,
                               duckdb::FunctionNullHandling::SPECIAL_HANDLING,
                               simple_update,
                               bind);
  fn.stability       = duckdb::FunctionStability::VOLATILE;
  fn.order_dependent = duckdb::AggregateOrderDependent::NOT_ORDER_DEPENDENT;
  return fn;
}
}  // namespace
void register_aggregate_functions(duckdb::DatabaseInstance& instance)
{
  auto transaction = duckdb::CatalogTransaction::GetSystemTransaction(instance);
  auto& catalog    = duckdb::Catalog::GetSystemCatalog(instance);
  for (unsigned i = 0; i <= static_cast<unsigned>(aggregate_op::max); ++i) {
    duckdb::CreateAggregateFunctionInfo info(make_function(static_cast<aggregate_op>(i)));
    catalog.CreateFunction(transaction, info);
  }
}
void validate_bound_aggregate(duckdb::BoundAggregateExpression const& expression)
{
  auto data = dynamic_cast<aggregate_data const*>(expression.bind_info.get());
  auto id   = from_duckdb_aggregate_name(expression.function.name);
  if (!data || !id || !is_decimal_aggregate(*id) || data->op != aggregate_operation(*id) ||
      expression.children.size() != 1 || data->input != expression.children[0]->return_type ||
      data->output != expression.return_type || expression.function.bind != bind ||
      expression.IsDistinct() || expression.filter || expression.order_bys)
    throw std::invalid_argument(
      "MO aggregate lost its checked signature or has unsupported modifiers");
  validate_aggregate_signature(
    data->op, sirius::from_duckdb(data->input), sirius::from_duckdb(data->output));
}
duckdb::unique_ptr<duckdb::Expression> bound_aggregate(
  aggregate_op op,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children,
  logical_type const& output)
{
  if (children.size() != 1) throw std::invalid_argument("MO aggregate needs one operand");
  validate_aggregate_signature(op, sirius::from_duckdb(children[0]->return_type), output);
  auto fn        = make_function(op);
  fn.arguments   = {children[0]->return_type};
  fn.return_type = duckdb_type(output);
  auto result    = duckdb::make_uniq<duckdb::BoundAggregateExpression>(
    fn, std::move(children), nullptr, nullptr, duckdb::AggregateType::NON_DISTINCT);
  result->bind_info = duckdb::make_uniq<aggregate_data>(op, fn.arguments[0], fn.return_type);
  return result;
}
}  // namespace sirius::mo_decimal
