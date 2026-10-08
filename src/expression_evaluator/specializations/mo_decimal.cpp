/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "expression_evaluator/expression_evaluator.hpp"
#include "numeric/decimal_error.hpp"
#include "numeric/decimal_functions.hpp"
#include "numeric/exact_decimal_gpu.hpp"
#include "pipeline/gpu_stream_quiescence_error.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

namespace sirius {
expression_evaluator::evaluate_result expression_evaluator::evaluate_decimal_function(
  ast::function_call const& expression, evaluation_mode mode)
{
  using namespace mo_decimal;
  auto op               = operation(expression.function());
  auto const& arguments = expression.arguments();
  std::vector<logical_type> types;
  for (auto const& argument : arguments) {
    if (!argument) throw std::invalid_argument("missing MO decimal argument");
    types.push_back(argument->return_type());
  }
  validate_signature(op, types, expression.return_type());
  if (op == decimal_op::cast && types[0].is_mo_decimal()) {
    auto a = types[0].mo_decimal_type(), b = expression.return_type().mo_decimal_type();
    if (a.bits == b.bits && a.precision == b.precision && a.scale == b.scale)
      return evaluate(*arguments[0], mode);
  }
  struct owners {
    std::vector<evaluate_result> arguments;
    std::vector<std::unique_ptr<cudf::column>> lifted;
    std::optional<decimal_column_result> result;
  };
  auto owner = std::make_unique<owners>();
  owner->arguments.reserve(arguments.size());
  owner->lifted.reserve(2 * arguments.size());
  try {
    std::vector<cudf::column_view> columns;
    std::vector<decimal_type> descriptors;
    columns.reserve(arguments.size());
    descriptors.reserve(arguments.size());
    for (std::size_t i = 0; i < arguments.size(); ++i) {
      owner->arguments.push_back(evaluate(*arguments[i], evaluation_mode::MATERIALIZE));
      auto& value = owner->arguments.back();
      if (value.is_scalar()) {
        owner->lifted.push_back(
          cudf::make_column_from_scalar(value.get_scalar(), _input_table.num_rows(), _stream, _mr));
        columns.push_back(owner->lifted.back()->view());
      } else
        columns.push_back(value.get_column_view());
      if (types[i].is_mo_decimal())
        descriptors.push_back(types[i].mo_decimal_type());
      else {
        if (columns.back().type().id() != cudf::type_id::INT64) {
          owner->lifted.push_back(
            cudf::cast(columns.back(), cudf::data_type{cudf::type_id::INT64}, _stream, _mr));
          columns.back() = owner->lifted.back()->view();
        }
        auto column    = columns.back();
        columns.back() = cudf::column_view(cudf::data_type{cudf::type_id::DECIMAL64, 0},
                                           column.size(),
                                           column.head<uint8_t>(),
                                           column.null_mask(),
                                           column.null_count(),
                                           column.offset());
        descriptors.push_back({64, 18, 0});
      }
    }
    auto output = expression.return_type().is_mo_decimal()
                    ? expression.return_type().mo_decimal_type()
                    : decimal_type{64, 18, 0};
    auto right  = columns.size() == 1 ? 0u : 1u;
    owner->result.emplace(evaluate_decimal_columns(op,
                                                   columns[0],
                                                   descriptors[0],
                                                   columns[right],
                                                   descriptors[right],
                                                   output,
                                                   _mo_active_rows,
                                                   _stream,
                                                   _mr));
    auto error = column_error(*owner->result, _stream, _mr);
    if (error != decimal_error::none) throw numeric_error(error);
    auto result = std::move(owner->result->values);
    return mode == evaluation_mode::AST ? materialize_as_ast_column(std::move(result))
                                        : evaluate_result(std::move(result));
  } catch (...) {
    auto error = std::current_exception();
    if (cudaStreamSynchronize(_stream.get()) != cudaSuccess) {
      (void)owner.release();
      throw pipeline::gpu_stream_quiescence_error(
        "MO decimal expression could not prove quiescence");
    }
    std::rethrow_exception(error);
  }
}
}  // namespace sirius
