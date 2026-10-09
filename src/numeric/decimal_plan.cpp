/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "numeric/decimal_plan.hpp"

#include "helper/type_conversions.hpp"
#include "numeric/decimal_aggregate_bind.hpp"
#include "numeric/decimal_functions.hpp"
#include "numeric/decimal_import.hpp"
#include "substrait/plan.pb.h"

#include <duckdb/planner/column_binding_map.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_case_expression.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/logical_operator_visitor.hpp>
#include <duckdb/planner/operator/logical_any_join.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace sirius::mo_decimal {
namespace {
bool same_decimal(logical_type const& a, logical_type const& b)
{
  if (!a.is_mo_decimal() || !b.is_mo_decimal()) return false;
  auto x = a.mo_decimal_type(), y = b.mo_decimal_type();
  return x.bits == y.bits && x.precision == y.precision && x.scale == y.scale;
}
struct wire_validator {
  substrait::Plan plan;
  std::vector<uint32_t> anchors;
  std::unordered_map<uint32_t, std::string> functions;
  std::unordered_set<uint32_t> exact_functions;
  uint32_t cast_anchor{}, uri{};
  std::vector<std::vector<logical_type>> relations;
  std::size_t schema_limit{}, schema_bytes{};
  void charge(std::size_t cells)
  {
    if (cells > (schema_limit - schema_bytes) / sizeof(logical_type)) throw std::bad_alloc();
    schema_bytes += cells * sizeof(logical_type);
  }
  logical_type type(substrait::Type const& t)
  {
    if (t.has_user_defined()) return import_result_type(t, anchors);
    using K = substrait::Type::KindCase;
    type_id id;
    bool nullable = false;
    switch (t.kind_case()) {
      case K::kBool:
        id       = type_id::BOOLEAN;
        nullable = t.bool_().nullability() == substrait::Type::NULLABILITY_NULLABLE;
        break;
      case K::kI8:
        id       = type_id::TINYINT;
        nullable = t.i8().nullability() == substrait::Type::NULLABILITY_NULLABLE;
        break;
      case K::kI16:
        id       = type_id::SMALLINT;
        nullable = t.i16().nullability() == substrait::Type::NULLABILITY_NULLABLE;
        break;
      case K::kI32:
        id       = type_id::INTEGER;
        nullable = t.i32().nullability() == substrait::Type::NULLABILITY_NULLABLE;
        break;
      case K::kI64:
        id       = type_id::BIGINT;
        nullable = t.i64().nullability() == substrait::Type::NULLABILITY_NULLABLE;
        break;
      case K::kFp32:
        id       = type_id::FLOAT;
        nullable = t.fp32().nullability() == substrait::Type::NULLABILITY_NULLABLE;
        break;
      case K::kFp64:
        id       = type_id::DOUBLE;
        nullable = t.fp64().nullability() == substrait::Type::NULLABILITY_NULLABLE;
        break;
      case K::kString:
        id       = type_id::VARCHAR;
        nullable = t.string().nullability() == substrait::Type::NULLABILITY_NULLABLE;
        break;
      case K::kVarchar:
        id       = type_id::VARCHAR;
        nullable = t.varchar().nullability() == substrait::Type::NULLABILITY_NULLABLE;
        break;
      case K::kBinary:
        id       = type_id::VARCHAR;
        nullable = t.binary().nullability() == substrait::Type::NULLABILITY_NULLABLE;
        break;
      case K::kDate:
        id       = type_id::DATE;
        nullable = t.date().nullability() == substrait::Type::NULLABILITY_NULLABLE;
        break;
      case K::kPrecisionTimestamp:
        id       = type_id::TIMESTAMP;
        nullable = t.precision_timestamp().nullability() == substrait::Type::NULLABILITY_NULLABLE;
        break;
      default: throw std::invalid_argument("unsupported type in exact-decimal closure");
    }
    return logical_type::make(id).with_nullability(nullable);
  }
  substrait::Type wire_type(logical_type const& logical)
  {
    substrait::Type t;
    auto user = t.mutable_user_defined();
    user->set_type_reference(anchors.front());
    auto d = logical.mo_decimal_type();
    for (auto value : {int(d.bits), int(d.precision), int(d.scale)})
      user->add_type_parameters()->set_integer(value);
    user->set_nullability(logical.nullability() == 2 ? substrait::Type::NULLABILITY_NULLABLE
                                                     : substrait::Type::NULLABILITY_REQUIRED);
    return t;
  }
  void annotate(substrait::Expression& expression, logical_type const& logical)
  {
    substrait::Expression original;
    original.Swap(&expression);
    auto function = expression.mutable_scalar_function();
    function->set_function_reference(cast_anchor);
    *function->mutable_output_type() = wire_type(logical);
    function->add_arguments()->mutable_value()->Swap(&original);
  }
  logical_type expression(substrait::Expression& e, std::vector<logical_type> const& input)
  {
    if (e.has_selection()) {
      auto const& s = e.selection();
      if (!s.has_direct_reference() || !s.direct_reference().has_struct_field() ||
          s.direct_reference().struct_field().has_child() || !s.has_root_reference())
        throw std::invalid_argument("unsupported exact-plan field reference");
      auto index = s.direct_reference().struct_field().field();
      if (index < 0 || static_cast<size_t>(index) >= input.size())
        throw std::invalid_argument("exact-plan field outside schema");
      auto result = input[index];
      if (result.is_mo_decimal()) annotate(e, result);
      return result;
    }
    if (e.has_literal()) {
      auto const& l = e.literal();
      if (l.has_null()) return type(l.null());
      if (l.has_user_defined()) {
        substrait::Type t;
        auto u = t.mutable_user_defined();
        u->set_type_reference(l.user_defined().type_reference());
        *u->mutable_type_parameters() = l.user_defined().type_parameters();
        u->set_nullability(l.nullable() ? substrait::Type::NULLABILITY_NULLABLE
                                        : substrait::Type::NULLABILITY_REQUIRED);
        return type(t);
      }
      type_id id = l.has_boolean() ? type_id::BOOLEAN
                   : l.has_i8()    ? type_id::TINYINT
                   : l.has_i16()   ? type_id::SMALLINT
                   : l.has_i32()   ? type_id::INTEGER
                   : l.has_i64()   ? type_id::BIGINT
                   : l.has_fp32()  ? type_id::FLOAT
                   : l.has_fp64()  ? type_id::DOUBLE
                   : l.has_date()  ? type_id::DATE
                                   : type_id::VARCHAR;
      return logical_type::make(id).with_nullability(l.nullable());
    }
    if (e.has_scalar_function()) {
      auto function = e.mutable_scalar_function();
      std::vector<logical_type> args;
      charge(function->arguments_size());
      for (auto& argument : *function->mutable_arguments())
        if (argument.has_value()) args.push_back(expression(*argument.mutable_value(), input));
      auto result = type(function->output_type());
      auto found  = functions.find(function->function_reference());
      if (found == functions.end())
        throw std::invalid_argument("unknown exact-plan function anchor");
      if (exact_functions.contains(function->function_reference())) {
        auto id = from_duckdb_function_name("__sirius_" + found->second);
        if (!id || !is_decimal_function(*id))
          throw std::invalid_argument("invalid MO scalar identity");
        validate_signature(operation(*id), args, result);
      } else {
        bool exact = result.is_mo_decimal();
        for (auto const& a : args)
          exact |= a.is_mo_decimal();
        auto name = found->second.substr(0, found->second.find(':'));
        if (exact && name != "coalesce" && name != "is_null" && name != "is_not_null")
          throw std::invalid_argument("ordinary function cannot coerce an MO exact decimal");
        if (name == "coalesce" && result.is_mo_decimal())
          for (auto const& a : args)
            if (!same_decimal(a, result))
              throw std::invalid_argument("MO COALESCE descriptors differ");
        if (name == "coalesce" && result.is_mo_decimal()) annotate(e, result);
      }
      return result;
    }
    if (e.has_if_then()) {
      auto branch = e.mutable_if_then();
      std::optional<logical_type> result;
      bool nullable = false;
      for (auto& pair : *branch->mutable_ifs()) {
        if (expression(*pair.mutable_if_(), input).id() != type_id::BOOLEAN)
          throw std::invalid_argument("CASE condition is not boolean");
        auto value = expression(*pair.mutable_then(), input);
        nullable |= value.nullability() == 2;
        if (result && (result->is_mo_decimal() || value.is_mo_decimal()) &&
            !same_decimal(*result, value))
          throw std::invalid_argument("MO CASE branches require matching exact descriptors");
        result = value;
      }
      if (!result || !branch->has_else_())
        throw std::invalid_argument("CASE requires typed branches and ELSE");
      auto other = expression(*branch->mutable_else_(), input);
      nullable |= other.nullability() == 2;
      if ((result->is_mo_decimal() || other.is_mo_decimal()) && !same_decimal(*result, other))
        throw std::invalid_argument("MO CASE ELSE descriptor mismatch");
      auto output = result->with_nullability(nullable);
      if (output.is_mo_decimal()) annotate(e, output);
      return output;
    }
    if (e.has_cast()) {
      auto child  = expression(*e.mutable_cast()->mutable_input(), input),
           target = type(e.cast().type());
      if (child.is_mo_decimal() || target.is_mo_decimal())
        throw std::invalid_argument("MO casts require the checked versioned function");
      return target;
    }
    if (e.has_singular_or_list()) {
      auto list  = e.mutable_singular_or_list();
      auto probe = expression(*list->mutable_value(), input);
      if (probe.is_mo_decimal())
        throw std::invalid_argument("exact decimal IN lists are unsupported");
      for (auto& value : *list->mutable_options())
        if (expression(value, input).is_mo_decimal())
          throw std::invalid_argument("mixed exact IN list");
      return logical_type::make(type_id::BOOLEAN).with_nullability(true);
    }
    if (e.has_nested()) {
      if (!e.nested().has_struct_())
        throw std::invalid_argument("unsupported nested exact-plan value");
      for (auto& field : *e.mutable_nested()->mutable_struct_()->mutable_fields())
        if (expression(field, input).is_mo_decimal())
          throw std::invalid_argument("nested exact-decimal tuple is unsupported");
      return logical_type::make(type_id::STRUCT);
    }
    throw std::invalid_argument("unsupported expression in exact-plan closure");
  }
  std::vector<logical_type> relation(substrait::Rel& rel)
  {
    std::vector<logical_type> result;
    substrait::RelCommon const* common = nullptr;
    if (rel.has_read()) {
      auto read = rel.mutable_read();
      common    = &read->common();
      charge(read->base_schema().struct_().types_size());
      for (auto const& t : read->base_schema().struct_().types())
        result.push_back(type(t));
      if (read->has_filter() &&
          expression(*read->mutable_filter(), result).id() != type_id::BOOLEAN)
        throw std::invalid_argument("read filter is not boolean");
      if (read->has_best_effort_filter())
        throw std::invalid_argument("exact-plan best-effort read filters are unsupported");
      if (read->has_projection()) {
        if (!read->projection().has_select() || !read->projection().select().struct_items_size())
          throw std::invalid_argument("exact-plan read projection requires a flat selection");
        charge(read->projection().select().struct_items_size());
        std::vector<logical_type> projected;
        for (auto const& item : read->projection().select().struct_items()) {
          auto index = item.field();
          if (item.has_child() || index < 0 || static_cast<size_t>(index) >= result.size())
            throw std::invalid_argument("invalid exact-plan read projection field");
          projected.push_back(result[index]);
        }
        result = std::move(projected);
      }
    } else if (rel.has_project()) {
      auto p = rel.mutable_project();
      common = &p->common();
      result = relation(*p->mutable_input());
      charge(result.size() + p->expressions_size());
      auto input = result;
      for (auto& e : *p->mutable_expressions())
        result.push_back(expression(e, input));
    } else if (rel.has_filter()) {
      auto p = rel.mutable_filter();
      common = &p->common();
      result = relation(*p->mutable_input());
      if (expression(*p->mutable_condition(), result).id() != type_id::BOOLEAN)
        throw std::invalid_argument("filter is not boolean");
    } else if (rel.has_fetch()) {
      auto p = rel.mutable_fetch();
      common = &p->common();
      result = relation(*p->mutable_input());
    } else if (rel.has_sort()) {
      auto p = rel.mutable_sort();
      common = &p->common();
      result = relation(*p->mutable_input());
      for (auto& field : *p->mutable_sorts())
        expression(*field.mutable_expr(), result);
    } else if (rel.has_aggregate()) {
      auto p     = rel.mutable_aggregate();
      common     = &p->common();
      auto input = relation(*p->mutable_input());
      charge(p->grouping_expressions_size() + p->measures_size());
      if (p->groupings_size() > 1)
        throw std::invalid_argument("exact-plan grouping sets are unsupported");
      if (p->grouping_expressions_size()) {
        if (p->groupings_size() != 1 ||
            p->groupings(0).expression_references_size() != p->grouping_expressions_size())
          throw std::invalid_argument("exact-plan grouping set must cover every key");
        std::unordered_set<uint32_t> keys;
        for (auto index : p->groupings(0).expression_references())
          if (index >= static_cast<uint32_t>(p->grouping_expressions_size()) ||
              !keys.insert(index).second)
            throw std::invalid_argument("invalid exact-plan grouping key reference");
      }
      for (auto& e : *p->mutable_grouping_expressions())
        result.push_back(expression(e, input));
      for (auto& measure : *p->mutable_measures()) {
        auto f = measure.mutable_measure();
        if (measure.has_filter() || f->sorts_size() || f->options_size() ||
            f->phase() != substrait::AGGREGATION_PHASE_INITIAL_TO_RESULT ||
            (f->invocation() != substrait::AggregateFunction::AGGREGATION_INVOCATION_ALL &&
             f->invocation() != substrait::AggregateFunction::AGGREGATION_INVOCATION_DISTINCT))
          throw std::invalid_argument("unsupported exact-plan aggregate modifiers");
        std::vector<logical_type> args;
        charge(f->arguments_size());
        for (auto& arg : *f->mutable_arguments()) {
          if (!arg.has_value()) throw std::invalid_argument("invalid aggregate argument");
          args.push_back(expression(*arg.mutable_value(), input));
        }
        auto output = type(f->output_type());
        auto found  = functions.find(f->function_reference());
        if (found == functions.end()) throw std::invalid_argument("missing aggregate anchor");
        if (exact_functions.contains(f->function_reference())) {
          auto id = from_duckdb_aggregate_name("__sirius_" + found->second);
          if (!id || !is_decimal_aggregate(*id) || args.size() != 1 || measure.has_filter())
            throw std::invalid_argument("invalid exact aggregate");
          validate_aggregate_signature(aggregate_operation(*id), args[0], output);
        } else {
          auto name  = found->second.substr(0, found->second.find(':'));
          bool exact = output.is_mo_decimal();
          for (auto const& a : args)
            exact |= a.is_mo_decimal();
          if (exact && name != "count")
            throw std::invalid_argument("ordinary aggregate cannot infer an exact descriptor");
        }
        result.push_back(output);
      }
    } else if (rel.has_join()) {
      auto p    = rel.mutable_join();
      common    = &p->common();
      auto left = relation(*p->mutable_left()), right = relation(*p->mutable_right());
      charge(2 * (left.size() + right.size()) + 1);
      auto both = left;
      both.insert(both.end(), right.begin(), right.end());
      if (expression(*p->mutable_expression(), both).id() != type_id::BOOLEAN)
        throw std::invalid_argument("join condition is not boolean");
      auto kind = p->type();
      if (kind == substrait::JoinRel::JOIN_TYPE_LEFT ||
          kind == substrait::JoinRel::JOIN_TYPE_LEFT_SINGLE ||
          kind == substrait::JoinRel::JOIN_TYPE_OUTER)
        for (auto& t : right)
          t = t.with_nullability(true);
      if (kind == substrait::JoinRel::JOIN_TYPE_RIGHT ||
          kind == substrait::JoinRel::JOIN_TYPE_OUTER)
        for (auto& t : left)
          t = t.with_nullability(true);
      if (kind == substrait::JoinRel::JOIN_TYPE_RIGHT_SEMI ||
          kind == substrait::JoinRel::JOIN_TYPE_RIGHT_ANTI)
        result = right;
      else {
        result = left;
        if (kind == substrait::JoinRel::JOIN_TYPE_LEFT_MARK)
          result.push_back(logical_type::make(type_id::BOOLEAN).with_nullability(true));
        else if (kind != substrait::JoinRel::JOIN_TYPE_LEFT_SEMI &&
                 kind != substrait::JoinRel::JOIN_TYPE_LEFT_ANTI)
          result.insert(result.end(), right.begin(), right.end());
      }
    } else if (rel.has_cross()) {
      auto p     = rel.mutable_cross();
      common     = &p->common();
      result     = relation(*p->mutable_left());
      auto right = relation(*p->mutable_right());
      charge(right.size());
      result.insert(result.end(), right.begin(), right.end());
    } else if (rel.has_reference()) {
      auto index = rel.reference().subtree_ordinal();
      if (index < 0 || static_cast<size_t>(index) >= relations.size())
        throw std::invalid_argument("invalid exact-plan relation reference");
      charge(relations[index].size());
      result = relations[index];
    } else if (rel.has_set()) {
      auto p = rel.mutable_set();
      common = &p->common();
      for (auto& input : *p->mutable_inputs()) {
        auto schema = relation(input);
        if (result.empty())
          result = std::move(schema);
        else {
          if (result.size() != schema.size())
            throw std::invalid_argument("set schema width differs");
          for (size_t i = 0; i < result.size(); ++i) {
            if ((result[i].is_mo_decimal() || schema[i].is_mo_decimal()) && result[i] != schema[i])
              throw std::invalid_argument("set exact descriptors differ");
            result[i] = result[i].with_nullability(result[i].nullability() == 2 ||
                                                   schema[i].nullability() == 2);
          }
        }
      }
    } else
      throw std::invalid_argument("unsupported exact-plan relation");
    if (result.empty())
      throw std::invalid_argument("exact-plan relations require a nonempty row schema");
    if (common && common->has_emit()) {
      charge(common->emit().output_mapping_size());
      std::vector<logical_type> emitted;
      for (int index : common->emit().output_mapping()) {
        if (index < 0 || static_cast<size_t>(index) >= result.size())
          throw std::invalid_argument("exact-plan emit outside schema");
        emitted.push_back(result[index]);
      }
      if (emitted.empty())
        throw std::invalid_argument("exact-plan emit requires a nonempty row schema");
      return emitted;
    }
    return result;
  }
};
}  // namespace
std::string normalize_exact_substrait(std::string_view bytes, std::size_t schema_bytes_limit)
{
  if (bytes.size() > (16u << 20)) throw std::invalid_argument("exact plan exceeds 16 MiB");
  wire_validator v;
  v.schema_limit = schema_bytes_limit;
  if (!v.plan.ParseFromArray(bytes.data(), static_cast<int>(bytes.size())))
    throw std::invalid_argument("invalid exact plan");
  std::unordered_set<uint32_t> uris;
  uint32_t maximum = 0;
  for (auto const& urn : v.plan.extension_urns())
    if (urn.urn() == extension_uri) {
      uris.insert(urn.extension_urn_anchor());
      v.uri = urn.extension_urn_anchor();
    }
  for (auto const& ext : v.plan.extensions()) {
    if (ext.has_extension_type() && uris.contains(ext.extension_type().extension_urn_reference()) &&
        ext.extension_type().name() == "mo_exact_decimal")
      v.anchors.push_back(ext.extension_type().type_anchor());
    if (ext.has_extension_function()) {
      auto const& f = ext.extension_function();
      maximum       = std::max(maximum, f.function_anchor());
      if (!v.functions.emplace(f.function_anchor(), f.name()).second)
        throw std::invalid_argument("duplicate function anchor");
      if (uris.contains(f.extension_urn_reference())) {
        v.exact_functions.insert(f.function_anchor());
        if (f.name() == "mo_decimal_cast") v.cast_anchor = f.function_anchor();
      }
    }
  }
  std::sort(v.anchors.begin(), v.anchors.end());
  if (v.anchors.empty())
    throw std::invalid_argument("exact plan has no canonical type declaration");
  if (!v.cast_anchor) {
    if (maximum == UINT32_MAX) throw std::invalid_argument("exact plan function anchor overflow");
    v.cast_anchor = maximum + 1;
    auto f        = v.plan.add_extensions()->mutable_extension_function();
    f->set_function_anchor(v.cast_anchor);
    f->set_extension_urn_reference(v.uri);
    f->set_name("mo_decimal_cast");
  }
  for (auto& relation : *v.plan.mutable_relations())
    v.relations.push_back(v.relation(relation.has_root() ? *relation.mutable_root()->mutable_input()
                                                         : *relation.mutable_rel()));
  if (v.plan.ByteSizeLong() > (16u << 20))
    throw std::invalid_argument("normalized exact plan exceeds 16 MiB");
  return v.plan.SerializeAsString();
}
void restore_exact_bound_types(duckdb::unique_ptr<duckdb::Expression>& expression)
{
  duckdb::ExpressionIterator::EnumerateChildren(
    *expression,
    [](duckdb::unique_ptr<duckdb::Expression>& child) { restore_exact_bound_types(child); });
  if (expression->GetExpressionClass() == duckdb::ExpressionClass::BOUND_CAST) {
    auto& cast  = expression->Cast<duckdb::BoundCastExpression>();
    auto source = from_duckdb_type(cast.child->return_type);
    if (source) {
      auto plain = duckdb_type(*source);
      plain.SetAlias("");
      auto target = from_duckdb_type(cast.return_type);
      // All wire-level MO casts were checked versioned functions. These
      // BoundCast nodes are DuckDB's conditional alias/nullability coercions;
      // restore the original declared branch rather than a generic STRUCT cast.
      if (plain == cast.return_type || cast.child->return_type == cast.return_type ||
          (target && same_decimal(*source, *target)))
        expression = std::move(cast.child);
    }
  }
  std::vector<duckdb::Expression*> branches;
  bool coalesce = false;
  if (expression->GetExpressionClass() == duckdb::ExpressionClass::BOUND_CASE) {
    auto& c = expression->Cast<duckdb::BoundCaseExpression>();
    for (auto& branch : c.case_checks)
      branches.push_back(branch.then_expr.get());
    branches.push_back(c.else_expr.get());
  } else if (expression->GetExpressionClass() == duckdb::ExpressionClass::BOUND_OPERATOR &&
             expression->GetExpressionType() == duckdb::ExpressionType::OPERATOR_COALESCE) {
    coalesce = true;
    for (auto& child : expression->Cast<duckdb::BoundOperatorExpression>().children)
      branches.push_back(child.get());
  }
  std::optional<logical_type> result;
  bool nullable = coalesce;
  for (auto branch : branches) {
    auto t = from_duckdb_type(branch->return_type);
    if (!t || !t->is_mo_decimal()) {
      result.reset();
      break;
    }
    if (result && !same_decimal(*result, *t))
      throw std::invalid_argument("bound conditional changed an exact descriptor");
    result = t;
    if (coalesce)
      nullable &= t->nullability() == 2;
    else
      nullable |= t->nullability() == 2;
  }
  if (result) expression->return_type = duckdb_type(result->with_nullability(nullable));
}
namespace {
duckdb::ExpressionType comparison_type(decimal_op op)
{
  using T = duckdb::ExpressionType;
  switch (op) {
    case decimal_op::equal: return T::COMPARE_EQUAL;
    case decimal_op::not_equal: return T::COMPARE_NOTEQUAL;
    case decimal_op::less: return T::COMPARE_LESSTHAN;
    case decimal_op::less_equal: return T::COMPARE_LESSTHANOREQUALTO;
    case decimal_op::greater: return T::COMPARE_GREATERTHAN;
    case decimal_op::greater_equal: return T::COMPARE_GREATERTHANOREQUALTO;
    default: throw std::invalid_argument("not an exact comparison");
  }
}
void rewrite_expression(duckdb::unique_ptr<duckdb::Expression>& expression, bool comparisons)
{
  duckdb::ExpressionIterator::EnumerateChildren(
    *expression,
    [&](duckdb::unique_ptr<duckdb::Expression>& child) { rewrite_expression(child, comparisons); });
  if (expression->GetExpressionClass() == duckdb::ExpressionClass::BOUND_FUNCTION) {
    auto& fn = expression->Cast<duckdb::BoundFunctionExpression>();
    auto id  = from_duckdb_function_name(fn.function.name);
    if (id && is_decimal_function(*id)) {
      validate_bound_scalar(fn);
      if (operation(*id) == decimal_op::cast &&
          sirius::from_duckdb(fn.children[0]->return_type) == sirius::from_duckdb(fn.return_type)) {
        expression = std::move(fn.children[0]);
      } else if (comparisons && operation(*id) >= decimal_op::equal) {
        auto output = fn.return_type;
        expression  = duckdb::make_uniq<duckdb::BoundComparisonExpression>(
          comparison_type(operation(*id)), std::move(fn.children[0]), std::move(fn.children[1]));
        expression->return_type = std::move(output);
      }
    }
  } else if (expression->GetExpressionClass() == duckdb::ExpressionClass::BOUND_AGGREGATE) {
    auto& agg = expression->Cast<duckdb::BoundAggregateExpression>();
    auto id   = from_duckdb_aggregate_name(agg.function.name);
    if (id && is_decimal_aggregate(*id)) validate_bound_aggregate(agg);
  }
}
}  // namespace
void rewrite_exact_comparisons(duckdb::unique_ptr<duckdb::LogicalOperator>& plan,
                               duckdb::ClientContext& context)
{
  for (auto& child : plan->children)
    rewrite_exact_comparisons(child, context);
  duckdb::column_binding_map_t<duckdb::LogicalType> bindings;
  for (auto& child : plan->children) {
    auto columns = child->GetColumnBindings();
    if (columns.size() != child->types.size())
      throw std::invalid_argument("exact-plan child schema lost bindings");
    for (size_t i = 0; i < columns.size(); ++i)
      bindings.emplace(columns[i], child->types[i]);
  }
  std::function<void(duckdb::unique_ptr<duckdb::Expression>&)> restore_references;
  restore_references = [&](duckdb::unique_ptr<duckdb::Expression>& expression) {
    if (expression->GetExpressionClass() == duckdb::ExpressionClass::BOUND_COLUMN_REF &&
        !from_duckdb_type(expression->return_type)) {
      auto& ref  = expression->Cast<duckdb::BoundColumnRefExpression>();
      auto found = bindings.find(ref.binding);
      if (found != bindings.end() && from_duckdb_type(found->second)) {
        auto plain = duckdb_type(*from_duckdb_type(found->second));
        plain.SetAlias("");
        if (plain == ref.return_type) ref.return_type = found->second;
      }
    }
    duckdb::ExpressionIterator::EnumerateChildren(*expression, restore_references);
  };
  duckdb::LogicalOperatorVisitor::EnumerateExpressions(
    *plan,
    [&](duckdb::unique_ptr<duckdb::Expression>* expression) { restore_references(*expression); });
  bool comparisons = plan->type == duckdb::LogicalOperatorType::LOGICAL_ANY_JOIN;
  duckdb::LogicalOperatorVisitor::EnumerateExpressions(
    *plan, [&](duckdb::unique_ptr<duckdb::Expression>* expression) {
      restore_exact_bound_types(*expression);
      rewrite_expression(*expression, comparisons);
    });
  if (plan->type == duckdb::LogicalOperatorType::LOGICAL_ANY_JOIN) {
    auto& join = plan->Cast<duckdb::LogicalAnyJoin>();
    if (join.condition->GetExpressionClass() == duckdb::ExpressionClass::BOUND_CONSTANT &&
        join.condition->return_type.id() == duckdb::LogicalTypeId::BOOLEAN &&
        !join.condition->Cast<duckdb::BoundConstantExpression>().value.IsNull() &&
        join.condition->Cast<duckdb::BoundConstantExpression>().value.GetValue<bool>()) {
      // Exact plans skip DuckDB's ordinary optimizer. JOIN ON true otherwise
      // remains an unsupported ANY_JOIN. Equal constant keys use the existing
      // GPU join, retaining multiplicity and outer NULLs even for empty sides.
      auto replacement = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(join.join_type);
      duckdb::JoinCondition condition;
      condition.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
      condition.left =
        duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::TINYINT(1));
      condition.right =
        duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::TINYINT(1));
      replacement->conditions.push_back(std::move(condition));
      replacement->children             = std::move(join.children);
      replacement->left_projection_map  = std::move(join.left_projection_map);
      replacement->right_projection_map = std::move(join.right_projection_map);
      replacement->mark_index           = join.mark_index;
      plan                              = std::move(replacement);
    } else {
      plan = duckdb::LogicalComparisonJoin::CreateJoin(context,
                                                       join.join_type,
                                                       duckdb::JoinRefType::REGULAR,
                                                       std::move(join.children[0]),
                                                       std::move(join.children[1]),
                                                       std::move(join.condition));
    }
  }
  if (plan->type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    auto& join = plan->Cast<duckdb::LogicalComparisonJoin>();
    for (auto const& condition : join.conditions) {
      auto left  = from_duckdb_type(condition.left->return_type),
           right = from_duckdb_type(condition.right->return_type);
      if (((left && left->is_mo_decimal()) || (right && right->is_mo_decimal())) &&
          condition.comparison != duckdb::ExpressionType::COMPARE_EQUAL)
        throw std::invalid_argument("MO decimal joins require exact equality keys");
    }
  }
  plan->ResolveOperatorTypes();
}
}  // namespace sirius::mo_decimal
