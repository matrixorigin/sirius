/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "expression_evaluator/expression_evaluator.hpp"
#include "numeric/decimal_functions.hpp"
#include "numeric/exact_decimal_gpu.hpp"
#include "pipeline/gpu_stream_quiescence_error.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

namespace sirius {
namespace {
using result = expression_evaluator::evaluate_result;
struct active_scope {
  bool const*& target;
  bool const* previous;
  active_scope(bool const*& t, cudf::column_view mask) : target(t), previous(t)
  {
    target = mask.data<bool>();
  }
  ~active_scope() { target = previous; }
};
// At most one branch's temporaries are retained. On a failed synchronization,
// ownership transfers to process quarantine rather than unwinding live buffers.
struct conditional_owners {
  std::unique_ptr<cudf::column> output, remaining, condition, selected, inverse;
  std::unique_ptr<cudf::column> next_output, next_remaining, lifted;
  std::unique_ptr<cudf::scalar> null_scalar, truth, falsity, reduced;
  std::optional<result> predicate, value;

  void retire_branch(logical_type const& type)
  {
    if (next_output && type.is_mo_decimal() && type.mo_decimal_type().bits == 256) {
      // cuDF selection superimposes parent validity on STRUCT children. Inputs
      // were canonical, so these child masks contain no independent NULLs.
      // The caller has synchronized the task stream before retiring them.
      auto rows = next_output->size(), nulls = next_output->null_count();
      auto contents = next_output->release();
      for (auto& child : contents.children) {
        auto child_type = child->type();
        auto count      = child->size();
        auto data       = child->release();
        child =
          std::make_unique<cudf::column>(child_type,
                                         count,
                                         data.data ? std::move(*data.data) : rmm::device_buffer{},
                                         rmm::device_buffer{},
                                         0);
      }
      next_output = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::STRUCT},
        rows,
        rmm::device_buffer{},
        contents.null_mask ? std::move(*contents.null_mask) : rmm::device_buffer{},
        nulls,
        std::move(contents.children));
    }
    if (next_output) output = std::move(next_output);
    if (next_remaining) remaining = std::move(next_remaining);
    predicate.reset();
    value.reset();
    lifted.reset();
    condition.reset();
    selected.reset();
    inverse.reset();
    reduced.reset();
  }
};
void require_canonical(cudf::column_view value, logical_type const& type)
{
  if (type.is_mo_decimal() && !mo_decimal::decimal_column_matches(value, type.mo_decimal_type()))
    throw std::invalid_argument("MO conditional input is not a canonical exact-decimal column");
}
void initialize(conditional_owners& owner,
                logical_type const& type,
                cudf::size_type rows,
                bool const* active,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr)
{
  owner.truth   = std::make_unique<cudf::numeric_scalar<bool>>(true, true, stream, mr);
  owner.falsity = std::make_unique<cudf::numeric_scalar<bool>>(false, true, stream, mr);
  if (type.is_mo_decimal())
    owner.output =
      mo_decimal::make_decimal_literal(type.mo_decimal_type(), {}, false, rows, stream, mr);
  else {
    owner.null_scalar = cudf::make_default_constructed_scalar(get_cudf_type(type), stream, mr);
    owner.null_scalar->set_valid_async(false, stream);
    owner.output = cudf::make_column_from_scalar(*owner.null_scalar, rows, stream, mr);
  }
  if (active)
    owner.remaining = std::make_unique<cudf::column>(
      cudf::column_view(cudf::data_type{cudf::type_id::BOOL8}, rows, active, nullptr, 0),
      stream,
      mr);
  else
    owner.remaining = cudf::make_column_from_scalar(*owner.truth, rows, stream, mr);
}
cudf::column_view column(result& value,
                         conditional_owners& owner,
                         cudf::size_type rows,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
{
  if (!value.is_scalar()) return value.get_column_view();
  owner.lifted = cudf::make_column_from_scalar(value.get_scalar(), rows, stream, mr);
  return owner.lifted->view();
}
bool any(cudf::column_view mask,
         conditional_owners& owner,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr)
{
  if (!mask.size()) return false;
  owner.reduced = cudf::reduce(mask,
                               *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                               cudf::data_type{cudf::type_id::BOOL8},
                               stream,
                               mr);
  return static_cast<cudf::numeric_scalar<bool>&>(*owner.reduced).value(stream);
}
void rethrow_after_quiescence(std::unique_ptr<conditional_owners>& owner,
                              rmm::cuda_stream_view stream)
{
  auto error = std::current_exception();
  if (cudaStreamSynchronize(stream.value()) != cudaSuccess) {
    (void)owner.release();
    throw pipeline::gpu_stream_quiescence_error(
      "MO conditional expression could not prove quiescence");
  }
  std::rethrow_exception(error);
}
}  // namespace

bool expression_evaluator::uses_mo_expressions()
{
  if (!_has_mo_expressions) {
    _has_mo_expressions = false;
    for (auto expression : _ast_expressions)
      if (expression && mo_decimal::contains_exact_expression(*expression)) {
        _has_mo_expressions = true;
        break;
      }
  }
  return *_has_mo_expressions;
}
expression_evaluator::evaluate_result expression_evaluator::evaluate_mo_case(
  ast::case_expr const& expression, evaluation_mode mode)
{
  if (expression.cases.empty() || !expression.else_)
    throw std::invalid_argument("invalid MO numeric CASE expression");
  auto owner = std::make_unique<conditional_owners>();
  auto rows  = _input_table.num_rows();
  try {
    initialize(*owner, expression.return_type(), rows, _mo_active_rows, _stream, _mr);
    for (auto const& branch : expression.cases) {
      if (!branch.when_ || !branch.then_)
        throw std::invalid_argument("missing MO numeric CASE branch");
      if (!any(owner->remaining->view(), *owner, _stream, _mr)) break;
      {
        active_scope active(_mo_active_rows, owner->remaining->view());
        owner->predicate.emplace(evaluate(*branch.when_, evaluation_mode::MATERIALIZE));
      }
      auto predicate   = column(*owner->predicate, *owner, rows, _stream, _mr);
      owner->condition = cudf::replace_nulls(predicate, *owner->falsity, _stream, _mr);
      owner->selected  = cudf::binary_operation(owner->remaining->view(),
                                               owner->condition->view(),
                                               cudf::binary_operator::LOGICAL_AND,
                                               cudf::data_type{cudf::type_id::BOOL8},
                                               _stream,
                                               _mr);
      // Complete any scalar lift before another branch can replace its owner.
      CUDF_CUDA_TRY(cudaStreamSynchronize(_stream.get()));
      owner->lifted.reset();
      if (any(owner->selected->view(), *owner, _stream, _mr)) {
        {
          active_scope active(_mo_active_rows, owner->selected->view());
          owner->value.emplace(evaluate(*branch.then_, evaluation_mode::MATERIALIZE));
        }
        auto value = column(*owner->value, *owner, rows, _stream, _mr);
        require_canonical(value, expression.return_type());
        owner->next_output =
          cudf::copy_if_else(value, owner->output->view(), owner->selected->view(), _stream, _mr);
        owner->inverse =
          cudf::unary_operation(owner->condition->view(), cudf::unary_operator::NOT, _stream, _mr);
        owner->next_remaining = cudf::binary_operation(owner->remaining->view(),
                                                       owner->inverse->view(),
                                                       cudf::binary_operator::LOGICAL_AND,
                                                       cudf::data_type{cudf::type_id::BOOL8},
                                                       _stream,
                                                       _mr);
      }
      CUDF_CUDA_TRY(cudaStreamSynchronize(_stream.get()));
      owner->retire_branch(expression.return_type());
    }
    if (any(owner->remaining->view(), *owner, _stream, _mr)) {
      {
        active_scope active(_mo_active_rows, owner->remaining->view());
        owner->value.emplace(evaluate(*expression.else_, evaluation_mode::MATERIALIZE));
      }
      auto value = column(*owner->value, *owner, rows, _stream, _mr);
      require_canonical(value, expression.return_type());
      owner->next_output =
        cudf::copy_if_else(value, owner->output->view(), owner->remaining->view(), _stream, _mr);
    }
    CUDF_CUDA_TRY(cudaStreamSynchronize(_stream.get()));
    owner->retire_branch(expression.return_type());
    auto output = std::move(owner->output);
    return mode == evaluation_mode::AST ? materialize_as_ast_column(std::move(output))
                                        : evaluate_result(std::move(output));
  } catch (...) {
    rethrow_after_quiescence(owner, _stream);
    throw;
  }
}
expression_evaluator::evaluate_result expression_evaluator::evaluate_mo_coalesce(
  ast::coalesce const& expression, evaluation_mode mode)
{
  if (expression.children.size() < 2) throw std::invalid_argument("invalid MO numeric COALESCE");
  auto owner = std::make_unique<conditional_owners>();
  auto rows  = _input_table.num_rows();
  try {
    initialize(*owner, expression.return_type(), rows, _mo_active_rows, _stream, _mr);
    for (auto const& child : expression.children) {
      if (!child) throw std::invalid_argument("missing MO numeric COALESCE argument");
      if (!any(owner->remaining->view(), *owner, _stream, _mr)) break;
      {
        active_scope active(_mo_active_rows, owner->remaining->view());
        owner->value.emplace(evaluate(*child, evaluation_mode::MATERIALIZE));
      }
      auto value = column(*owner->value, *owner, rows, _stream, _mr);
      require_canonical(value, expression.return_type());
      owner->next_output =
        cudf::copy_if_else(value, owner->output->view(), owner->remaining->view(), _stream, _mr);
      owner->condition      = cudf::is_null(value, _stream, _mr);
      owner->next_remaining = cudf::binary_operation(owner->remaining->view(),
                                                     owner->condition->view(),
                                                     cudf::binary_operator::LOGICAL_AND,
                                                     cudf::data_type{cudf::type_id::BOOL8},
                                                     _stream,
                                                     _mr);
      CUDF_CUDA_TRY(cudaStreamSynchronize(_stream.get()));
      owner->retire_branch(expression.return_type());
    }
    CUDF_CUDA_TRY(cudaStreamSynchronize(_stream.get()));
    auto output = std::move(owner->output);
    return mode == evaluation_mode::AST ? materialize_as_ast_column(std::move(output))
                                        : evaluate_result(std::move(output));
  } catch (...) {
    rethrow_after_quiescence(owner, _stream);
    throw;
  }
}
}  // namespace sirius
