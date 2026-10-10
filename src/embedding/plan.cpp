/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/control.hpp"
#include "numeric/decimal_aggregate_bind.hpp"
#include "numeric/decimal_functions.hpp"
#include "numeric/decimal_import.hpp"
#include "substrait/algebra.pb.h"
#include "substrait/plan.pb.h"

#include <algorithm>
#include <charconv>
#include <limits>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace sirius::embedding {
namespace {
constexpr std::string_view prefix = "__sirius_embedded_v1";
struct numeric_profile {
  bool exact{};
  std::vector<uint32_t> types;
  std::unordered_set<uint32_t> functions;
};

uint64_t binding_id(std::string const& text)
{
  uint64_t id{};
  auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), id);
  if (ec != std::errc{} || end != text.data() + text.size() || id == 0 ||
      id > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) || text != std::to_string(id))
    throw failure(SIRIUS_INVALID_ARGUMENT, "invalid embedded read binding id");
  return id;
}

std::string normalized_function(std::string name)
{
  auto separator = name.find(':');
  if (separator != std::string::npos) name.resize(separator);
  return name;
}

bool supported_scalar_function(std::string const& name)
{
  static const std::unordered_set<std::string> supported{"add",
                                                         "subtract",
                                                         "multiply",
                                                         "divide",
                                                         "modulus",
                                                         "and",
                                                         "or",
                                                         "not",
                                                         "lt",
                                                         "lte",
                                                         "gt",
                                                         "gte",
                                                         "equal",
                                                         "not_equal",
                                                         "is_null",
                                                         "is_not_null",
                                                         "is_not_distinct_from",
                                                         "between",
                                                         "coalesce",
                                                         "like",
                                                         "extract",
                                                         "year",
                                                         "month",
                                                         "day",
                                                         "abs",
                                                         "sqrt",
                                                         "power",
                                                         "ceil",
                                                         "floor",
                                                         "lower",
                                                         "upper",
                                                         "length",
                                                         "substring",
                                                         "trim",
                                                         "date_add",
                                                         "date_diff",
                                                         "date_part"};
  return supported.contains(normalized_function(name));
}

bool supported_aggregate_function(std::string const& name)
{
  static const std::unordered_set<std::string> supported{
    "count", "sum", "min", "max", "avg", "grouping"};
  return supported.contains(normalized_function(name));
}

bool read_type_matches(substrait::Type const& type,
                       owned_column const& column,
                       numeric_profile const& profile)
{
  if (profile.exact && (column.oid == 32 || column.oid == 33 || column.oid == 34)) {
    if (!type.has_user_defined()) return false;
    auto logical = mo_decimal::import_result_type(type, profile.types);
    auto decimal = logical.mo_decimal_type();
    return decimal.bits == (column.oid == 32   ? 64
                            : column.oid == 33 ? 128
                                               : 256) &&
           decimal.precision == column.width && decimal.scale == column.scale;
  }
  using Kind = substrait::Type::KindCase;
  switch (column.oid) {
    case 10: return type.kind_case() == Kind::kBool;
    case 20: return type.kind_case() == Kind::kI8;
    case 21: return type.kind_case() == Kind::kI16;
    case 22: return type.kind_case() == Kind::kI32;
    case 23: return type.kind_case() == Kind::kI64;
    case 30: return type.kind_case() == Kind::kFp32;
    case 31: return type.kind_case() == Kind::kFp64;
    case 32:
    case 33:
      return type.kind_case() == Kind::kDecimal && type.decimal().precision() == column.width &&
             type.decimal().scale() == column.scale;
    case 50: return type.kind_case() == Kind::kDate;
    case 52:
      return type.kind_case() == Kind::kPrecisionTimestamp &&
             type.precision_timestamp().precision() == 6;
    case 60:
    case 61:
    case 71: return type.kind_case() == Kind::kVarchar || type.kind_case() == Kind::kString;
    case 70: return type.kind_case() == Kind::kBinary;
    default: return false;  // Includes unsigned widths and Decimal256 unsupported by this plan ABI.
  }
}

bool supported_embedded_type(substrait::Type const& type, numeric_profile const& profile)
{
  if (type.has_user_defined()) {
    return profile.exact && mo_decimal::import_result_type(type, profile.types).is_mo_decimal();
  }
  using Kind = substrait::Type::KindCase;
  switch (type.kind_case()) {
    case Kind::kBool:
    case Kind::kI8:
    case Kind::kI16:
    case Kind::kI32:
    case Kind::kI64:
    case Kind::kFp32:
    case Kind::kFp64:
    case Kind::kString:
    case Kind::kVarchar:
    case Kind::kBinary:
    case Kind::kDate:
    case Kind::kPrecisionTimestamp:
      return type.kind_case() != Kind::kPrecisionTimestamp ||
             type.precision_timestamp().precision() == 6;
    case Kind::kDecimal:
      return !profile.exact && type.decimal().precision() > 0 && type.decimal().precision() <= 38 &&
             type.decimal().scale() >= 0 && type.decimal().scale() <= type.decimal().precision();
    default: return false;
  }
}

bool supported_literal(substrait::Expression_Literal const& literal, numeric_profile const& profile)
{
  if (literal.has_user_defined())
    return profile.exact &&
           std::binary_search(
             profile.types.begin(), profile.types.end(), literal.user_defined().type_reference());
  using Literal = substrait::Expression_Literal;
  switch (literal.literal_type_case()) {
    case Literal::kNull:
    case Literal::kBoolean:
    case Literal::kI8:
    case Literal::kI32:
    case Literal::kI64:
    case Literal::kFp64:
    case Literal::kString:
    case Literal::kDate:
    case Literal::kVarChar: return true;
    case Literal::kDecimal:
      return !profile.exact && literal.decimal().value().size() == 16 &&
             literal.decimal().precision() > 0 && literal.decimal().precision() <= 38 &&
             literal.decimal().scale() >= 0 &&
             literal.decimal().scale() <= literal.decimal().precision();
    default: return false;
  }
}

int read_nullability(substrait::Type const& type)
{
  auto const* reflection = type.GetReflection();
  std::vector<const duckdb::google::protobuf::FieldDescriptor*> fields;
  reflection->ListFields(type, &fields);
  for (auto const* field : fields) {
    if (field->cpp_type() != duckdb::google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE) continue;
    auto const& concrete    = reflection->GetMessage(type, field);
    auto const* nullability = concrete.GetDescriptor()->FindFieldByName("nullability");
    if (nullability != nullptr)
      return concrete.GetReflection()->GetEnumValue(concrete, nullability);
  }
  return substrait::Type::NULLABILITY_UNSPECIFIED;
}

void inspect_message(duckdb::google::protobuf::Message const& message,
                     int relation_ordinal,
                     query_state const& query,
                     std::set<uint64_t>& reads,
                     std::unordered_map<uint32_t, std::string> const& functions,
                     numeric_profile const& profile)
{
  auto const name = message.GetDescriptor()->full_name();
  if (name == "substrait.Type") {
    if (!supported_embedded_type(static_cast<substrait::Type const&>(message), profile))
      throw failure(SIRIUS_UNSUPPORTED, "unsupported Substrait type in embedded plan");
  } else if (name == "substrait.Expression.Literal") {
    if (!supported_literal(static_cast<substrait::Expression_Literal const&>(message), profile))
      throw failure(SIRIUS_UNSUPPORTED, "unsupported Substrait literal in embedded plan");
  } else if (name == "substrait.Rel") {
    auto const& rel = static_cast<substrait::Rel const&>(message);
    if (rel.has_aggregate())
      for (auto const& measure : rel.aggregate().measures())
        if (measure.has_filter() &&
            profile.functions.contains(measure.measure().function_reference()))
          throw failure(SIRIUS_UNSUPPORTED, "MO aggregate filters are unsupported");
    switch (rel.rel_type_case()) {
      case substrait::Rel::kRead:
      case substrait::Rel::kFilter:
      case substrait::Rel::kFetch:
      case substrait::Rel::kAggregate:
      case substrait::Rel::kSort:
      case substrait::Rel::kJoin:
      case substrait::Rel::kProject:
      case substrait::Rel::kSet:
      case substrait::Rel::kCross:
      case substrait::Rel::kReference: break;
      default: throw failure(SIRIUS_UNSUPPORTED, "unsupported Substrait relation");
    }
    if (rel.has_join()) {
      if (!rel.join().has_left() || !rel.join().has_right() || !rel.join().has_expression() ||
          rel.join().has_post_join_filter())
        throw failure(SIRIUS_UNSUPPORTED, "unsupported Substrait join shape");
      switch (rel.join().type()) {
        case substrait::JoinRel::JOIN_TYPE_INNER:
        case substrait::JoinRel::JOIN_TYPE_LEFT:
        case substrait::JoinRel::JOIN_TYPE_RIGHT:
        case substrait::JoinRel::JOIN_TYPE_LEFT_SINGLE:
        case substrait::JoinRel::JOIN_TYPE_LEFT_SEMI:
        case substrait::JoinRel::JOIN_TYPE_RIGHT_SEMI:
        case substrait::JoinRel::JOIN_TYPE_LEFT_ANTI:
        case substrait::JoinRel::JOIN_TYPE_RIGHT_ANTI:
        case substrait::JoinRel::JOIN_TYPE_LEFT_MARK:
        case substrait::JoinRel::JOIN_TYPE_OUTER: break;
        default: throw failure(SIRIUS_UNSUPPORTED, "unsupported Substrait join type");
      }
    }
    if (rel.has_fetch()) {
      auto const& fetch = rel.fetch();
      if (fetch.count_mode_case() != substrait::FetchRel::kCount ||
          fetch.offset_mode_case() != substrait::FetchRel::kOffset)
        throw failure(SIRIUS_UNSUPPORTED,
                      "expression-based Substrait fetch bounds are unsupported");
      auto const* descriptor = fetch.GetDescriptor();
      auto const* count_field =
        descriptor->FindFieldByNumber(substrait::FetchRel::kCountFieldNumber);
      auto const* offset_field =
        descriptor->FindFieldByNumber(substrait::FetchRel::kOffsetFieldNumber);
      auto count  = fetch.GetReflection()->GetInt64(fetch, count_field);
      auto offset = fetch.GetReflection()->GetInt64(fetch, offset_field);
      if (count < -1 || offset < 0)
        throw failure(SIRIUS_INVALID_ARGUMENT, "invalid Substrait fetch bounds");
    }
  } else if (name == "substrait.Expression") {
    auto const& expression = static_cast<substrait::Expression const&>(message);
    switch (expression.rex_type_case()) {
      case substrait::Expression::kLiteral:
      case substrait::Expression::kSelection:
      case substrait::Expression::kScalarFunction:
      case substrait::Expression::kIfThen:
      case substrait::Expression::kCast:
      case substrait::Expression::kSingularOrList:
      case substrait::Expression::kNested: break;
      default: throw failure(SIRIUS_UNSUPPORTED, "unsupported Substrait expression");
    }
  } else if (name == "substrait.Expression.ScalarFunction") {
    auto const& function = static_cast<substrait::Expression_ScalarFunction const&>(message);
    auto found           = functions.find(function.function_reference());
    auto exact           = found == functions.end() ? std::optional<function_id>{}
                                                    : from_duckdb_function_name("__sirius_" + found->second);
    bool numeric         = exact && mo_decimal::is_decimal_function(*exact) &&
                   profile.functions.contains(function.function_reference());
    if (found == functions.end() || (!numeric && !supported_scalar_function(found->second)))
      throw failure(SIRIUS_UNSUPPORTED, "unsupported scalar function in embedded plan");
    auto const normalized = normalized_function(found->second);
    static const std::unordered_set<std::string> extract_fields{"year",
                                                                "month",
                                                                "day",
                                                                "decade",
                                                                "century",
                                                                "millennium",
                                                                "quarter",
                                                                "microsecond",
                                                                "milliseconds",
                                                                "second",
                                                                "minute",
                                                                "hour"};
    for (auto const& argument : function.arguments()) {
      if (argument.has_type() || (argument.has_enum_() && normalized != "extract") ||
          (argument.has_enum_() && !extract_fields.contains(argument.enum_())))
        throw failure(SIRIUS_UNSUPPORTED, "unsupported scalar function argument");
    }
  } else if (name == "substrait.AggregateFunction") {
    auto const& function = static_cast<substrait::AggregateFunction const&>(message);
    auto found           = functions.find(function.function_reference());
    auto exact           = found == functions.end() ? std::optional<aggregate_id>{}
                                                    : from_duckdb_aggregate_name("__sirius_" + found->second);
    bool numeric         = exact && mo_decimal::is_decimal_aggregate(*exact) &&
                   profile.functions.contains(function.function_reference());
    if (found == functions.end() || (!numeric && !supported_aggregate_function(found->second)))
      throw failure(SIRIUS_UNSUPPORTED, "unsupported aggregate function in embedded plan");
    for (auto const& argument : function.arguments())
      if (!argument.has_value())
        throw failure(SIRIUS_UNSUPPORTED, "unsupported aggregate function argument");
  }
  if (name == "substrait.ReadRel") {
    auto const& read = static_cast<substrait::ReadRel const&>(message);
    if (!read.has_named_table() || read.named_table().names_size() != 2 ||
        read.named_table().names(0) != prefix)
      throw failure(SIRIUS_UNSUPPORTED, "embedded plans may read only registered named tables");
    auto id    = binding_id(read.named_table().names(1));
    auto found = std::find_if(query.bindings.begin(), query.bindings.end(), [id](auto const& b) {
      return b.binding_id == id;
    });
    if (found == query.bindings.end())
      throw failure(SIRIUS_INVALID_ARGUMENT, "plan references an unregistered read binding");
    if (!reads.insert(id).second)
      throw failure(SIRIUS_INVALID_ARGUMENT,
                    "a read binding may identify only one scan occurrence");
    if (!read.has_base_schema() ||
        read.base_schema().names_size() != static_cast<int>(found->columns.size()) ||
        !read.base_schema().has_struct_() ||
        read.base_schema().struct_().types_size() != static_cast<int>(found->columns.size()))
      throw failure(SIRIUS_INVALID_ARGUMENT, "plan read schema does not match its binding");
    for (int i = 0; i < read.base_schema().names_size(); ++i) {
      if (read.base_schema().names(i) != found->columns[static_cast<std::size_t>(i)].logical.name)
        throw failure(SIRIUS_INVALID_ARGUMENT, "plan read column name does not match its binding");
      auto const& expected      = found->columns[static_cast<std::size_t>(i)].logical;
      auto const& actual        = read.base_schema().struct_().types(i);
      auto actual_nullability   = read_nullability(actual);
      auto expected_nullability = expected.nullable ? substrait::Type::NULLABILITY_NULLABLE
                                                    : substrait::Type::NULLABILITY_REQUIRED;
      if (!read_type_matches(actual, expected, profile) ||
          actual_nullability != expected_nullability)
        throw failure(SIRIUS_INVALID_ARGUMENT, "plan read column type does not match its binding");
    }
  } else if (name == "substrait.ReferenceRel") {
    auto ordinal = static_cast<substrait::ReferenceRel const&>(message).subtree_ordinal();
    if (ordinal < 0 || ordinal >= relation_ordinal)
      throw failure(SIRIUS_INVALID_ARGUMENT, "Substrait reference must point backward");
  } else if (name == "substrait.ExtensionLeafRel" || name == "substrait.ExtensionSingleRel" ||
             name == "substrait.ExtensionMultiRel" || name == "substrait.WriteRel" ||
             name == "substrait.DdlRel" || name == "substrait.UpdateRel" ||
             name == "substrait.ExchangeRel") {
    throw failure(SIRIUS_UNSUPPORTED, "unsupported relation in embedded plan");
  }

  auto const* reflection = message.GetReflection();
  std::vector<const duckdb::google::protobuf::FieldDescriptor*> fields;
  reflection->ListFields(message, &fields);
  for (auto const* field : fields) {
    if (field->cpp_type() != duckdb::google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE) continue;
    if (field->is_repeated()) {
      for (int i = 0; i < reflection->FieldSize(message, field); ++i)
        inspect_message(reflection->GetRepeatedMessage(message, field, i),
                        relation_ordinal,
                        query,
                        reads,
                        functions,
                        profile);
    } else {
      inspect_message(
        reflection->GetMessage(message, field), relation_ordinal, query, reads, functions, profile);
    }
  }
}
// ReferenceRel is inlined by the importer. A single wire ReadRel can therefore
// become several consumers of the same destructive MO input. Count reachable
// consumption rather than only declarations, before any native reader starts.
void inspect_consumption(duckdb::google::protobuf::Message const& message,
                         substrait::Plan const& plan,
                         int ordinal,
                         query_state const& query,
                         std::set<uint64_t>& consumed,
                         std::size_t& visited,
                         int depth)
{
  if (++visited > 65536 || depth > 256)
    throw failure(SIRIUS_UNSUPPORTED, "embedded reference expansion exceeds its bound");
  auto const name = message.GetDescriptor()->full_name();
  if (name == "substrait.ReferenceRel") {
    auto const source = static_cast<substrait::ReferenceRel const&>(message).subtree_ordinal();
    if (source < 0 || source >= ordinal || !plan.relations(source).has_rel())
      throw failure(SIRIUS_INVALID_ARGUMENT, "invalid embedded subtree reference");
    inspect_consumption(
      plan.relations(source).rel(), plan, source, query, consumed, visited, depth + 1);
    return;
  }
  if (name == "substrait.ReadRel") {
    auto const& read = static_cast<substrait::ReadRel const&>(message);
    auto const id    = binding_id(read.named_table().names(1));
    auto binding = std::find_if(query.bindings.begin(), query.bindings.end(), [id](auto const& b) {
      return b.binding_id == id;
    });
    if (binding != query.bindings.end() && binding->source_kind == SIRIUS_READ_MO &&
        !consumed.insert(id).second)
      throw failure(SIRIUS_INVALID_ARGUMENT,
                    "a destructive MO read binding cannot be consumed through shared references");
  }
  auto const* reflection = message.GetReflection();
  std::vector<const duckdb::google::protobuf::FieldDescriptor*> fields;
  reflection->ListFields(message, &fields);
  for (auto const* field : fields) {
    if (field->cpp_type() != duckdb::google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE) continue;
    if (field->is_repeated()) {
      for (int i = 0; i < reflection->FieldSize(message, field); ++i)
        inspect_consumption(reflection->GetRepeatedMessage(message, field, i),
                            plan,
                            ordinal,
                            query,
                            consumed,
                            visited,
                            depth + 1);
    } else {
      inspect_consumption(
        reflection->GetMessage(message, field), plan, ordinal, query, consumed, visited, depth + 1);
    }
  }
}
}  // namespace

void validate_embedded_plan(std::string_view bytes, query_state const& query)
{
  if (!query.contract) throw failure(SIRIUS_INVALID_ARGUMENT, "missing native query contract");
  substrait::Plan plan;
  if (!plan.ParseFromArray(bytes.data(), static_cast<int>(bytes.size())))
    throw failure(SIRIUS_INVALID_ARGUMENT, "invalid serialized Substrait plan");
  if (!plan.has_version() || plan.version().major_number() != 0 ||
      plan.version().minor_number() != 78)
    throw failure(SIRIUS_UNSUPPORTED, "embedded plans require Substrait version 0.78");
  std::unordered_map<uint32_t, std::string> functions;
  numeric_profile profile;
  std::unordered_set<uint32_t> numeric_uris;
  for (auto const& uri : plan.extension_urns())
    if (uri.urn() == mo_decimal::extension_uri) {
      profile.exact = true;
      numeric_uris.insert(uri.extension_urn_anchor());
    }
  for (auto const& extension : plan.extensions())
    if (extension.has_extension_type()) {
      auto const& type = extension.extension_type();
      if (numeric_uris.contains(type.extension_urn_reference()) &&
          type.name() == "mo_exact_decimal")
        profile.types.push_back(type.type_anchor());
    }
  std::sort(profile.types.begin(), profile.types.end());
  if (profile.exact)
    for (auto const& binding : query.bindings)
      if (binding.source_kind != SIRIUS_READ_MO)
        throw failure(SIRIUS_UNSUPPORTED, "exact decimal v1 requires MO readers");
  if (profile.exact)
    for (auto const& output : query.contract->outputs)
      if (output.oid == 34 && output.width > 65)
        throw failure(SIRIUS_UNSUPPORTED, "public Decimal256 precision exceeds 65");
  for (auto const& extension : plan.extensions()) {
    if (!extension.has_extension_function()) continue;
    auto const& mapping = extension.extension_function();
    if (numeric_uris.contains(mapping.extension_urn_reference()))
      profile.functions.insert(mapping.function_anchor());
    if (!functions.emplace(mapping.function_anchor(), mapping.name()).second)
      throw failure(SIRIUS_INVALID_ARGUMENT, "duplicate Substrait function anchor");
  }
  if (plan.relations_size() == 0 || !plan.relations(plan.relations_size() - 1).has_root())
    throw failure(SIRIUS_INVALID_ARGUMENT, "embedded plan requires one final root relation");
  for (int i = 0; i + 1 < plan.relations_size(); ++i)
    if (!plan.relations(i).has_rel())
      throw failure(SIRIUS_INVALID_ARGUMENT, "only the final plan relation may be a root");
  auto const& root = plan.relations(plan.relations_size() - 1).root();
  if (!root.has_input() || root.input().rel_type_case() == substrait::Rel::REL_TYPE_NOT_SET)
    throw failure(SIRIUS_INVALID_ARGUMENT, "embedded root has no input relation");
  if (root.names_size() != static_cast<int>(query.contract->outputs.size()))
    throw failure(SIRIUS_INVALID_ARGUMENT, "plan output count does not match query contract");
  for (int i = 0; i < root.names_size(); ++i)
    if (root.names(i) != query.contract->outputs[static_cast<std::size_t>(i)].name)
      throw failure(SIRIUS_INVALID_ARGUMENT, "plan output name does not match query contract");
  std::set<uint64_t> reads;
  for (int i = 0; i < plan.relations_size(); ++i)
    inspect_message(plan.relations(i), i, query, reads, functions, profile);
  if (reads.size() != query.bindings.size())
    throw failure(SIRIUS_INVALID_ARGUMENT, "registered and planned read binding sets differ");
  std::set<uint64_t> consumed;
  std::size_t visited = 0;
  inspect_consumption(root.input(), plan, plan.relations_size() - 1, query, consumed, visited, 0);
  for (auto const& binding : query.bindings)
    if (binding.source_kind == SIRIUS_READ_MO && !consumed.contains(binding.binding_id))
      throw failure(SIRIUS_INVALID_ARGUMENT,
                    "registered MO read binding has no reachable consumer");
}
}  // namespace sirius::embedding
