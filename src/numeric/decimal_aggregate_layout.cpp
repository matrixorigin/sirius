/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "numeric/decimal_aggregate_layout.hpp"

#include "cudf/cudf_utils.hpp"
#include "numeric/decimal_aggregate_gpu.hpp"
#include "op/aggregate/group_key_labels.hpp"
#include "pipeline/gpu_stream_quiescence_error.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/sorting.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <numeric>
#include <optional>

namespace sirius::mo_decimal {
namespace {
logical_type ordinary_sum_type(logical_type const& input)
{
  if (input.id() == type_id::FLOAT) return logical_type::make(type_id::DOUBLE);
  if (input.is_integer() && input.fixed_width_byte_size() < 8)
    return logical_type::make(input.id() == type_id::UTINYINT || input.id() == type_id::USMALLINT ||
                                  input.id() == type_id::UINTEGER
                                ? type_id::UBIGINT
                                : type_id::BIGINT);
  return input;
}
}  // namespace
std::shared_ptr<aggregate_layout const> make_aggregate_layout(
  duckdb::vector<std::unique_ptr<ast::node>> const& groups,
  duckdb::vector<std::unique_ptr<ast::node>> const& aggregates)
{
  bool needed = false;
  for (auto const& group : groups)
    needed |= group && group->return_type().is_mo_decimal();
  for (auto const& value : aggregates) {
    auto const& aggr = ast::require_aggregate(value.get(), "MO aggregate layout");
    needed |= is_decimal_aggregate(aggr.function());
    for (auto const& child : aggr.arguments())
      needed |= child && child->return_type().is_mo_decimal();
  }
  if (!needed) return {};
  auto layout = std::make_shared<aggregate_layout>();
  for (auto const& group : groups) {
    auto const& ref = ast::require_reference(group.get(), "MO aggregate group");
    layout->group_indices.push_back(ref.column_index);
    layout->group_types.push_back(ref.return_type());
  }
  size_t next = groups.size();
  for (auto const& value : aggregates) {
    auto const& aggr = ast::require_aggregate(value.get(), "MO aggregate layout");
    aggregate_slot slot{aggr.function(), -1, {}, aggr.return_type(), {}, aggr.distinct(), next};
    auto const& args = aggr.arguments();
    if (args.empty()) {
      if (slot.id != aggregate_id::count_star)
        throw std::invalid_argument("aggregate operand missing");
    } else if (args.size() != 1 || !args[0])
      throw std::invalid_argument("invalid aggregate arity");
    else if (args[0]->is_reference()) {
      slot.input      = args[0]->as_reference().column_index;
      slot.input_type = args[0]->return_type();
    } else if (slot.distinct && slot.id == aggregate_id::count && args[0]->is_function_call() &&
               args[0]->as_function_call().function() == function_id::struct_pack) {
      for (auto const& child : args[0]->as_function_call().arguments())
        slot.struct_inputs.push_back(
          ast::require_reference(child.get(), "COUNT DISTINCT tuple").column_index);
      slot.input_type = logical_type::make(type_id::STRUCT);
    } else
      throw std::invalid_argument("aggregate operand was not materialized");
    if (slot.exact()) {
      if (slot.distinct) throw std::invalid_argument("MO aggregate DISTINCT is unsupported");
      validate_aggregate_signature(aggregate_operation(slot.id), slot.input_type, slot.output_type);
    } else {
      if (slot.input_type.is_mo_decimal() && slot.id != aggregate_id::count)
        throw std::invalid_argument("ordinary aggregate cannot infer an MO decimal signature");
      switch (slot.id) {
        case aggregate_id::sum:
        case aggregate_id::sum_no_overflow:
        case aggregate_id::avg:
        case aggregate_id::min:
        case aggregate_id::max:
        case aggregate_id::count:
        case aggregate_id::count_star: break;
        default: throw std::invalid_argument("unsupported aggregate in MO exact plan");
      }
      if (slot.distinct && slot.id != aggregate_id::count)
        throw std::invalid_argument("unsupported DISTINCT aggregate");
      if (slot.distinct && groups.empty())
        throw std::invalid_argument("ungrouped DISTINCT aggregate is unsupported");
    }
    next += slot.exact() || slot.avg() ? 2 : 1;
    layout->slots.push_back(std::move(slot));
  }
  return layout;
}
duckdb::vector<logical_type> aggregate_layout::local_types() const
{
  auto result = group_types;
  for (auto const& slot : slots) {
    if (slot.exact()) {
      auto input = aggregate_input_type(slot.input_type);
      result.push_back(logical_type::make_mo_decimal(
        {slot.state_bits(), static_cast<uint8_t>(slot.state_bits() == 256 ? 76 : 38), input.scale},
        false));
      result.push_back(logical_type::make(type_id::BIGINT));
    } else if (slot.distinct)
      result.push_back(logical_type::make(type_id::LIST));
    else {
      result.push_back(slot.avg() ? ordinary_sum_type(slot.input_type) : slot.output_type);
      if (slot.avg()) result.push_back(logical_type::make(type_id::BIGINT));
    }
  }
  return result;
}
namespace {
struct owners {
  std::unique_ptr<cudf::table> combined;
  std::optional<op::detail::group_key_labels> keys;
  std::unique_ptr<cudf::column> order, sorted_labels;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> offsets;
  std::unique_ptr<cudf::groupby::groupby> grouping;
  std::vector<std::unique_ptr<cudf::column>> columns, temporaries, finalized;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  std::vector<std::unique_ptr<cudf::table>> intermediates;
  std::optional<aggregate_columns> pending;
  cudf::size_type ungrouped_offsets[2]{};
};
template <class Base>
std::unique_ptr<Base> aggregation(cudf::aggregation::Kind kind)
{
  switch (kind) {
    case cudf::aggregation::Kind::SUM: return cudf::make_sum_aggregation<Base>();
    case cudf::aggregation::Kind::MIN: return cudf::make_min_aggregation<Base>();
    case cudf::aggregation::Kind::MAX: return cudf::make_max_aggregation<Base>();
    case cudf::aggregation::Kind::COUNT_ALL:
      return cudf::make_count_aggregation<Base>(cudf::null_policy::INCLUDE);
    case cudf::aggregation::Kind::COUNT_VALID:
      return cudf::make_count_aggregation<Base>(cudf::null_policy::EXCLUDE);
    case cudf::aggregation::Kind::COLLECT_SET:
      return cudf::make_collect_set_aggregation<Base>(cudf::null_policy::EXCLUDE);
    case cudf::aggregation::Kind::MERGE_SETS: return cudf::make_merge_sets_aggregation<Base>();
    default: throw std::invalid_argument("unsupported ordinary aggregate kind");
  }
}
std::unique_ptr<cudf::column> ordinary(owners& owner,
                                       cudf::column_view input,
                                       cudf::aggregation::Kind kind,
                                       cudf::data_type result_type,
                                       bool grouped,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  std::unique_ptr<cudf::column> result;
  if (kind == cudf::aggregation::Kind::SUM && input.type() != result_type &&
      cudf::is_fixed_width(input.type()) && cudf::is_fixed_width(result_type)) {
    owner.temporaries.push_back(cudf::cast(input, result_type, stream, mr));
    input = owner.temporaries.back()->view();
  }
  if (grouped) {
    auto gathered = cudf::gather(cudf::table_view{{input}},
                                 owner.order->view(),
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 mr)
                      ->release();
    owner.temporaries.push_back(std::move(gathered[0]));
    std::vector<cudf::groupby::aggregation_request> requests(1);
    requests[0].values = owner.temporaries.back()->view();
    requests[0].aggregations.push_back(aggregation<cudf::groupby_aggregation>(kind));
    auto out = owner.grouping->aggregate(requests, stream, mr);
    owner.intermediates.push_back(std::move(out.first));
    result = std::move(out.second[0].results[0]);
  } else {
    owner.scalars.push_back(
      cudf::reduce(input, *aggregation<cudf::reduce_aggregation>(kind), result_type, stream, mr));
    result = cudf::make_column_from_scalar(*owner.scalars.back(), 1, stream, mr);
  }
  if (result->type() != result_type && kind != cudf::aggregation::Kind::COLLECT_SET &&
      kind != cudf::aggregation::Kind::MERGE_SETS) {
    owner.temporaries.push_back(std::move(result));
    result = cudf::cast(owner.temporaries.back()->view(), result_type, stream, mr);
  }
  return result;
}
std::unique_ptr<cudf::table> finish(cudf::table_view input,
                                    aggregate_layout const& layout,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr,
                                    std::unique_ptr<owners> owner = std::make_unique<owners>())
{
  try {
    owner->finalized.reserve(layout.group_types.size() + layout.slots.size());
    owner->temporaries.reserve(2 * layout.slots.size());
    for (size_t i = 0; i < layout.group_types.size(); ++i) {
      owner->finalized.push_back(std::make_unique<cudf::column>(input.column(i), stream, mr));
      if (layout.group_types[i].is_mo_decimal())
        owner->finalized.back() = restore_decimal_validity(
          std::move(owner->finalized.back()), layout.group_types[i].mo_decimal_type(), stream, mr);
    }
    for (auto const& slot : layout.slots) {
      auto source = input.column(slot.state_index);
      if (slot.exact())
        owner->finalized.push_back(finalize_aggregate_columns(aggregate_operation(slot.id),
                                                              source,
                                                              input.column(slot.state_index + 1),
                                                              aggregate_input_type(slot.input_type),
                                                              slot.output_type.mo_decimal_type(),
                                                              stream,
                                                              mr));
      else if (slot.avg())
        owner->finalized.push_back(cudf::binary_operation(source,
                                                          input.column(slot.state_index + 1),
                                                          cudf::binary_operator::DIV,
                                                          get_cudf_type(slot.output_type),
                                                          stream,
                                                          mr));
      else if (slot.distinct) {
        owner->temporaries.push_back(
          cudf::lists::count_elements(cudf::lists_column_view(source), stream, mr));
        owner->finalized.push_back(cudf::cast(
          owner->temporaries.back()->view(), get_cudf_type(slot.output_type), stream, mr));
      } else
        owner->finalized.push_back(std::make_unique<cudf::column>(source, stream, mr));
    }
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    return std::make_unique<cudf::table>(std::move(owner->finalized));
  } catch (...) {
    auto error = std::current_exception();
    if (cudaStreamSynchronize(stream.value()) != cudaSuccess) {
      (void)owner.release();
      throw pipeline::gpu_stream_quiescence_error(
        "MO aggregate finalization could not prove quiescence");
    }
    std::rethrow_exception(error);
  }
}
std::unique_ptr<cudf::table> compute(cudf::table_view input,
                                     aggregate_layout const& layout,
                                     bool merging,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr,
                                     std::unique_ptr<owners> owner = std::make_unique<owners>())
{
  try {
    owner->columns.reserve(layout.group_types.size() + 2 * layout.slots.size());
    owner->temporaries.reserve(4 * layout.slots.size());
    owner->scalars.reserve(2 * layout.slots.size());
    owner->intermediates.reserve(2 * layout.slots.size());
    bool grouped           = !layout.group_indices.empty();
    cudf::size_type groups = 1;
    std::vector<cudf::column_view> group_views;
    for (size_t i = 0; i < layout.group_indices.size(); ++i) {
      auto key = input.column(merging ? i : layout.group_indices[i]);
      if (layout.group_types[i].is_mo_decimal())
        key = canonical_decimal_view(key, layout.group_types[i].mo_decimal_type(), stream, mr);
      group_views.push_back(key);
    }
    if (grouped) {
      owner->keys.emplace(
        op::detail::make_group_key_labels(cudf::table_view(group_views), stream, mr));
      groups = owner->keys->sorted_unique_keys->num_rows();
      owner->order =
        cudf::sorted_order(cudf::table_view{{owner->keys->labels->view()}}, {}, {}, stream, mr);
      auto sorted = cudf::gather(cudf::table_view{{owner->keys->labels->view()}},
                                 owner->order->view(),
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 mr)
                      ->release();
      owner->sorted_labels = std::move(sorted[0]);
      owner->grouping =
        std::make_unique<cudf::groupby::groupby>(cudf::table_view{{owner->sorted_labels->view()}},
                                                 cudf::null_policy::INCLUDE,
                                                 cudf::sorted::YES);
      owner->offsets =
        std::make_unique<rmm::device_uvector<cudf::size_type>>(groups + 1, stream, mr);
      make_group_offsets(
        owner->keys->labels->view(), owner->order->view(), owner->offsets->data(), groups, stream);
      auto keys = owner->keys->sorted_unique_keys->release();
      for (size_t i = 0; i < keys.size(); ++i) {
        if (layout.group_types[i].is_mo_decimal())
          keys[i] = restore_decimal_validity(
            std::move(keys[i]), layout.group_types[i].mo_decimal_type(), stream, mr);
        owner->columns.push_back(std::move(keys[i]));
      }
    } else {
      owner->offsets = std::make_unique<rmm::device_uvector<cudf::size_type>>(2, stream, mr);
      owner->ungrouped_offsets[1] = input.num_rows();
      CUDF_CUDA_TRY(cudaMemcpyAsync(owner->offsets->data(),
                                    owner->ungrouped_offsets,
                                    sizeof(owner->ungrouped_offsets),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
      CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    }
    for (auto const& slot : layout.slots) {
      auto source_index = merging ? static_cast<int>(slot.state_index) : slot.input;
      cudf::column_view source;
      if (!merging && slot.id == aggregate_id::count_star)
        source = input.column(0);
      else if (source_index >= 0)
        source = input.column(source_index);
      else {
        std::vector<std::unique_ptr<cudf::column>> children;
        for (int index : slot.struct_inputs)
          children.push_back(std::make_unique<cudf::column>(input.column(index), stream, mr));
        owner->temporaries.push_back(cudf::make_structs_column(
          input.num_rows(), std::move(children), 0, rmm::device_buffer{}, stream, mr));
        source = owner->temporaries.back()->view();
      }
      if (slot.exact()) {
        auto original   = aggregate_input_type(slot.input_type);
        auto descriptor = merging
                            ? decimal_type{slot.state_bits(),
                                           static_cast<uint8_t>(slot.state_bits() == 256 ? 76 : 38),
                                           original.scale}
                            : original;
        if (!merging && !slot.input_type.is_mo_decimal())
          source = cudf::column_view(cudf::data_type{cudf::type_id::DECIMAL64, 0},
                                     source.size(),
                                     source.head<uint8_t>(),
                                     source.null_mask(),
                                     source.null_count(),
                                     source.offset());
        owner->pending.emplace(reduce_aggregate_columns(
          aggregate_operation(slot.id),
          source,
          descriptor,
          slot.state_bits(),
          merging ? input.column(slot.state_index + 1) : cudf::column_view{},
          owner->order ? owner->order->view() : cudf::column_view{},
          owner->offsets->data(),
          groups,
          stream,
          mr));
        owner->columns.push_back(std::move(owner->pending->values));
        owner->columns.push_back(std::move(owner->pending->counts));
        owner->pending.reset();
        continue;
      }
      auto type = get_cudf_type(slot.output_type);
      if (slot.distinct) {
        owner->columns.push_back(ordinary(
          *owner,
          source,
          merging ? cudf::aggregation::Kind::MERGE_SETS : cudf::aggregation::Kind::COLLECT_SET,
          cudf::data_type{cudf::type_id::LIST},
          grouped,
          stream,
          mr));
      } else if (slot.avg()) {
        auto sum_type = get_cudf_type(ordinary_sum_type(slot.input_type));
        owner->columns.push_back(
          ordinary(*owner, source, cudf::aggregation::Kind::SUM, sum_type, grouped, stream, mr));
        owner->columns.push_back(
          ordinary(*owner,
                   merging ? input.column(slot.state_index + 1) : source,
                   merging ? cudf::aggregation::Kind::SUM : cudf::aggregation::Kind::COUNT_VALID,
                   cudf::data_type{cudf::type_id::INT64},
                   grouped,
                   stream,
                   mr));
      } else {
        auto kind =
          slot.id == aggregate_id::min                      ? cudf::aggregation::Kind::MIN
          : slot.id == aggregate_id::max                    ? cudf::aggregation::Kind::MAX
          : slot.id == aggregate_id::count_star && !merging ? cudf::aggregation::Kind::COUNT_ALL
          : slot.id == aggregate_id::count && !merging      ? cudf::aggregation::Kind::COUNT_VALID
                                                            : cudf::aggregation::Kind::SUM;
        owner->columns.push_back(ordinary(*owner, source, kind, type, grouped, stream, mr));
      }
    }
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    if (merging) {
      owner->combined = std::make_unique<cudf::table>(std::move(owner->columns));
      auto states     = owner->combined->view();
      return finish(states, layout, stream, mr, std::move(owner));
    }
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    return std::make_unique<cudf::table>(std::move(owner->columns));
  } catch (...) {
    auto error = std::current_exception();
    if (cudaStreamSynchronize(stream.value()) != cudaSuccess) {
      (void)owner.release();
      throw pipeline::gpu_stream_quiescence_error("MO aggregate layout could not prove quiescence");
    }
    std::rethrow_exception(error);
  }
}
}  // namespace
std::unique_ptr<cudf::table> local_aggregate_table(cudf::table_view input,
                                                   aggregate_layout const& layout,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  return compute(input, layout, false, stream, mr);
}
std::unique_ptr<cudf::table> merge_aggregate_tables(std::vector<cudf::table_view> const& inputs,
                                                    aggregate_layout const& layout,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  if (inputs.empty())
    throw std::invalid_argument("MO aggregate merge requires a schema-bearing partial batch");
  // A single local partial already has one row per group. It still needs
  // final precision/NULL publication, but no second grouping or reduction.
  if (inputs.size() == 1) return finish(inputs[0], layout, stream, mr);
  auto owner = std::make_unique<owners>();
  try {
    owner->combined = cudf::concatenate(inputs, stream, mr);
    // Concatenation can superimpose validity on Decimal256 group keys. The
    // accumulator coefficients themselves are always nonnullable raw states.
    owner->columns = owner->combined->release();
    for (size_t i = 0; i < layout.group_types.size(); ++i)
      if (layout.group_types[i].is_mo_decimal())
        owner->columns[i] = restore_decimal_validity(
          std::move(owner->columns[i]), layout.group_types[i].mo_decimal_type(), stream, mr);
    owner->combined = std::make_unique<cudf::table>(std::move(owner->columns));
    auto view       = owner->combined->view();
    return compute(view, layout, true, stream, mr, std::move(owner));
  } catch (...) {
    auto error = std::current_exception();
    if (cudaStreamSynchronize(stream.value()) != cudaSuccess) {
      (void)owner.release();
      throw pipeline::gpu_stream_quiescence_error(
        "MO aggregate concatenation could not prove quiescence");
    }
    std::rethrow_exception(error);
  }
}
}  // namespace sirius::mo_decimal
