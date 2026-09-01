/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "offload/substrait_execution.hpp"

#include "from_substrait.hpp"
#include "planner/sirius_prepare.hpp"
#include "sirius_interface.hpp"
#include "substrait/plan.pb.h"

#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/prepared_statement_data.hpp>
#include <duckdb/main/relation.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/planner/binder.hpp>
#include <duckdb/planner/bound_statement.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace sirius::offload {

namespace {

using protobuf_message = ::duckdb::google::protobuf::Message;

[[noreturn]] void fail(substrait_error_code code, const std::string& message)
{
  throw substrait_execution_error(code, message);
}

[[noreturn]] void invalid(const std::string& message)
{
  fail(substrait_error_code::INVALID_PLAN, message);
}

[[noreturn]] void unsupported(const std::string& message)
{
  fail(substrait_error_code::UNSUPPORTED_PLAN, message);
}

std::uint64_t system_now_unix_ms()
{
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

void validate_message_tree(const protobuf_message& message)
{
  const auto* reflection = message.GetReflection();
  if (reflection->GetUnknownFields(message).field_count() != 0) {
    invalid("Substrait plan contains unknown protobuf fields in " +
            message.GetDescriptor()->full_name());
  }

  std::vector<const ::duckdb::google::protobuf::FieldDescriptor*> fields;
  reflection->ListFields(message, &fields);
  for (const auto* field : fields) {
    if (field->name() == "type_variation_reference") {
      if (field->cpp_type() == ::duckdb::google::protobuf::FieldDescriptor::CPPTYPE_UINT32 &&
          reflection->GetUInt32(message, field) != 0) {
        unsupported("Substrait type variations are not supported");
      }
    }
    if (field->cpp_type() == ::duckdb::google::protobuf::FieldDescriptor::CPPTYPE_ENUM) {
      const auto enum_is_known = [&](int value) {
        return field->enum_type()->FindValueByNumber(value) != nullptr;
      };
      if (field->is_repeated()) {
        for (int i = 0; i < reflection->FieldSize(message, field); ++i) {
          if (!enum_is_known(reflection->GetRepeatedEnumValue(message, field, i))) {
            invalid("Substrait plan contains an unknown enum value in " + field->full_name());
          }
        }
      } else if (!enum_is_known(reflection->GetEnumValue(message, field))) {
        invalid("Substrait plan contains an unknown enum value in " + field->full_name());
      }
    }
    if (field->cpp_type() != ::duckdb::google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE) {
      continue;
    }

    const auto validate_child = [](const protobuf_message& child) {
      if (child.GetDescriptor()->full_name() == "substrait.extensions.AdvancedExtension" &&
          child.ByteSizeLong() != 0) {
        unsupported("Substrait advanced extensions are not supported");
      }
      validate_message_tree(child);
    };
    if (field->is_repeated()) {
      for (int i = 0; i < reflection->FieldSize(message, field); ++i) {
        validate_child(reflection->GetRepeatedMessage(message, field, i));
      }
    } else {
      validate_child(reflection->GetMessage(message, field));
    }
  }
}

class wire_reader {
 public:
  explicit wire_reader(std::string_view bytes) : bytes_(bytes) {}

  bool done() const noexcept { return offset_ == bytes_.size(); }

  std::uint64_t varint()
  {
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift < 70; shift += 7) {
      if (offset_ == bytes_.size()) { invalid("truncated extension-read varint"); }
      const auto byte = static_cast<std::uint8_t>(bytes_[offset_++]);
      if (shift == 63 && byte > 1) { invalid("overflowing extension-read varint"); }
      value |= static_cast<std::uint64_t>(byte & 0x7fU) << shift;
      if ((byte & 0x80U) == 0) { return value; }
    }
    invalid("overflowing extension-read varint");
  }

  std::string bytes()
  {
    const auto length = varint();
    if (length > bytes_.size() - offset_) { invalid("truncated extension-read bytes field"); }
    std::string result(bytes_.substr(offset_, static_cast<std::size_t>(length)));
    offset_ += static_cast<std::size_t>(length);
    return result;
  }

 private:
  std::string_view bytes_;
  std::size_t offset_ = 0;
};

tae_read parse_tae_read(std::string_view bytes)
{
  wire_reader reader(bytes);
  tae_read result;
  std::uint16_t seen = 0;
  while (!reader.done()) {
    const auto tag       = reader.varint();
    const auto field     = static_cast<unsigned>(tag >> 3U);
    const auto wire_type = static_cast<unsigned>(tag & 7U);
    if (field == 0 || field > 12) { invalid("TaeRead contains an unknown field"); }
    const auto mask = static_cast<std::uint16_t>(1U << (field - 1U));
    if ((seen & mask) != 0) { invalid("TaeRead contains a duplicate field"); }
    seen |= mask;

    const bool is_varint =
      field == 1 || field == 2 || field == 5 || field == 6 || field == 11 || field == 12;
    if (wire_type != (is_varint ? 0U : 2U)) { invalid("TaeRead field has the wrong wire type"); }
    switch (field) {
      case 1: {
        const auto value = reader.varint();
        if (value > std::numeric_limits<std::uint32_t>::max()) {
          invalid("TaeRead protocol_version overflows uint32");
        }
        result.protocol_version = static_cast<std::uint32_t>(value);
        break;
      }
      case 2: result.feature_bits = reader.varint(); break;
      case 3: result.read_ref = reader.bytes(); break;
      case 4: result.query_id = reader.bytes(); break;
      case 5: result.account_id = reader.varint(); break;
      case 6: result.table_id = reader.varint(); break;
      case 7: result.snapshot_ts = reader.bytes(); break;
      case 8: result.schema_digest = reader.bytes(); break;
      case 9: result.manifest_sha256 = reader.bytes(); break;
      case 10: result.capability_hash = reader.bytes(); break;
      case 11: result.expires_at_unix_ms = reader.varint(); break;
      case 12: result.database_id = reader.varint(); break;
      default: invalid("TaeRead contains an unknown field");
    }
  }
  constexpr std::uint16_t required_fields =
    static_cast<std::uint16_t>(((1U << 12U) - 1U) & ~(1U << 1U));
  if ((seen & required_fields) != required_fields) {
    invalid("TaeRead is missing a required field");
  }
  return result;
}

void validate_tae_read(const tae_read& request, std::uint64_t now_unix_ms)
{
  if (request.protocol_version != k_tae_read_protocol_version) {
    unsupported("unsupported TaeRead protocol version");
  }
  if (request.feature_bits != k_tae_read_feature_bits) {
    unsupported("unsupported TaeRead feature bits");
  }
  if (request.read_ref.empty() || request.read_ref.size() > k_max_read_ref_bytes) {
    invalid("TaeRead read_ref is empty or too large");
  }
  if (request.query_id.empty() || request.database_id == 0 || request.table_id == 0) {
    invalid("TaeRead identity fields are incomplete");
  }
  if (request.snapshot_ts.size() != 12 || request.schema_digest.size() != 32 ||
      request.manifest_sha256.size() != 32 || request.capability_hash.size() != 32) {
    invalid("TaeRead snapshot or digest fields have invalid lengths");
  }
  if (request.expires_at_unix_ms == 0 || request.expires_at_unix_ms <= now_unix_ms) {
    fail(substrait_error_code::AUTHENTICATION_FAILED, "TaeRead capability has expired");
  }
}

stream_read parse_stream_read(std::string_view bytes)
{
  wire_reader reader(bytes);
  stream_read result;
  std::uint16_t seen = 0;
  while (!reader.done()) {
    const auto tag       = reader.varint();
    const auto field     = static_cast<unsigned>(tag >> 3U);
    const auto wire_type = static_cast<unsigned>(tag & 7U);
    if (field == 0 || field > 9) { invalid("StreamRead contains an unknown field"); }
    const auto mask = static_cast<std::uint16_t>(1U << (field - 1U));
    if ((seen & mask) != 0) { invalid("StreamRead contains a duplicate field"); }
    seen |= mask;

    const bool is_varint = field == 1 || field == 2 || field == 5 || field == 9;
    if (wire_type != (is_varint ? 0U : 2U)) { invalid("StreamRead field has the wrong wire type"); }
    switch (field) {
      case 1: {
        const auto value = reader.varint();
        if (value > std::numeric_limits<std::uint32_t>::max()) {
          invalid("StreamRead protocol_version overflows uint32");
        }
        result.protocol_version = static_cast<std::uint32_t>(value);
        break;
      }
      case 2: result.feature_bits = reader.varint(); break;
      case 3: result.stream_ref = reader.bytes(); break;
      case 4: result.query_id = reader.bytes(); break;
      case 5: result.account_id = reader.varint(); break;
      case 6: result.snapshot_ts = reader.bytes(); break;
      case 7: result.schema_digest = reader.bytes(); break;
      case 8: result.capability_hash = reader.bytes(); break;
      case 9: result.expires_at_unix_ms = reader.varint(); break;
      default: invalid("StreamRead contains an unknown field");
    }
  }
  constexpr std::uint16_t required_fields =
    static_cast<std::uint16_t>(((1U << 9U) - 1U) & ~(1U << 1U));
  if ((seen & required_fields) != required_fields) {
    invalid("StreamRead is missing a required field");
  }
  return result;
}

void validate_stream_read(const stream_read& request, std::uint64_t now_unix_ms)
{
  if (request.protocol_version != k_stream_read_protocol_version) {
    unsupported("unsupported StreamRead protocol version");
  }
  if (request.feature_bits != k_stream_read_feature_bits) {
    unsupported("unsupported StreamRead feature bits");
  }
  if (request.stream_ref.size() != 32 || request.query_id.size() != 16) {
    invalid("StreamRead identity fields have invalid lengths");
  }
  if (request.snapshot_ts.size() != 12 || request.schema_digest.size() != 32 ||
      request.capability_hash.size() != 32) {
    invalid("StreamRead snapshot or digest fields have invalid lengths");
  }
  if (request.expires_at_unix_ms == 0 || request.expires_at_unix_ms <= now_unix_ms) {
    fail(substrait_error_code::AUTHENTICATION_FAILED, "StreamRead capability has expired");
  }
}

void validate_type(const ::substrait::Type& type)
{
  switch (type.kind_case()) {
    case ::substrait::Type::kBool:
    case ::substrait::Type::kI8:
    case ::substrait::Type::kI16:
    case ::substrait::Type::kI32:
    case ::substrait::Type::kI64:
    case ::substrait::Type::kFp32:
    case ::substrait::Type::kFp64:
    case ::substrait::Type::kString:
    case ::substrait::Type::kDate:
    case ::substrait::Type::kVarchar: break;
    case ::substrait::Type::kDecimal:
      if (type.decimal().precision() <= 0 || type.decimal().precision() > 38 ||
          type.decimal().scale() < 0 || type.decimal().scale() > type.decimal().precision()) {
        unsupported("Substrait decimal precision or scale is outside the Sirius range");
      }
      break;
    case ::substrait::Type::kPrecisionTimestamp:
      if (type.precision_timestamp().precision() != 6) {
        unsupported("only microsecond Substrait timestamps are supported");
      }
      break;
    default: unsupported("Substrait type is outside the Sirius v1 type subset");
  }
}

void validate_schema(const ::substrait::NamedStruct& schema)
{
  if (!schema.has_struct_()) { invalid("read relation is missing its struct schema"); }
  if (schema.names_size() != schema.struct_().types_size()) {
    invalid("Substrait schema name/type counts differ");
  }
  std::unordered_set<std::string> names;
  for (int i = 0; i < schema.names_size(); ++i) {
    if (schema.names(i).empty() || !names.emplace(schema.names(i)).second) {
      invalid("Substrait schema has an empty or duplicate field name");
    }
    validate_type(schema.struct_().types(i));
  }
}

std::string base_function_name(const std::string& name) { return name.substr(0, name.find(':')); }

class plan_validator {
 public:
  plan_validator(::substrait::Plan& plan, tae_read_resolver& resolver, std::uint64_t now_unix_ms)
    : plan_(plan), resolver_(resolver), now_unix_ms_(now_unix_ms)
  {
  }

  std::vector<std::unique_ptr<resolved_read>> validate()
  {
    if (plan_.relations_size() == 0 || !plan_.relations(plan_.relations_size() - 1).has_root()) {
      invalid("Substrait plan requires one final root relation");
    }
    if (plan_.parameter_bindings_size() != 0 || plan_.type_aliases_size() != 0 ||
        plan_.has_execution_behavior()) {
      unsupported("dynamic parameters, type aliases, and execution behaviors are not supported");
    }
    for (const auto& type_url : plan_.expected_type_urls()) {
      if (type_url != k_tae_read_type_url && type_url != k_stream_read_type_url) {
        unsupported("plan declares an unsupported protobuf Any type");
      }
    }
    validate_function_declarations();

    const auto producer_count = plan_.relations_size() - 1;
    for (int i = 0; i < producer_count; ++i) {
      if (!plan_.relations(i).has_rel()) {
        invalid("only the final Substrait PlanRel may be a root");
      }
      current_relation_ordinal_ = static_cast<std::uint32_t>(i);
      validate_rel(plan_.mutable_relations(i)->mutable_rel());
    }
    current_relation_ordinal_ = static_cast<std::uint32_t>(producer_count);
    auto* root                = plan_.mutable_relations(producer_count)->mutable_root();
    if (!root->has_input()) { invalid("Substrait root relation has no input"); }
    validate_rel(root->mutable_input());
    return std::move(resolutions_);
  }

 private:
  void validate_function_declarations()
  {
    std::unordered_set<std::uint32_t> urn_anchors;
    for (const auto& urn : plan_.extension_urns()) {
      if (urn.urn().empty() || !urn_anchors.emplace(urn.extension_urn_anchor()).second) {
        invalid("Substrait extension URN is empty or has a duplicate anchor");
      }
    }

    static const std::unordered_set<std::string> scalar_allowlist = {
      "and",  "or",          "not",       "equal",   "not_equal",   "lt",
      "lte",  "gt",          "gte",       "is_null", "is_not_null", "is_not_distinct_from",
      "add",  "subtract",    "multiply",  "divide",  "modulus",     "between",
      "like", "starts_with", "substring", "extract"};
    static const std::unordered_set<std::string> aggregate_allowlist = {
      "count", "sum", "min", "max", "avg"};

    for (const auto& declaration : plan_.extensions()) {
      if (!declaration.has_extension_function()) {
        unsupported("only Substrait function extensions are supported");
      }
      const auto& function = declaration.extension_function();
      if (function.name().empty() ||
          !functions_.emplace(function.function_anchor(), base_function_name(function.name()))
             .second) {
        invalid("Substrait function declaration is empty or has a duplicate anchor");
      }
      if (function.extension_urn_reference() != 0 &&
          urn_anchors.count(function.extension_urn_reference()) == 0) {
        invalid("Substrait function references an undeclared extension URN");
      }
      const auto& name = functions_.at(function.function_anchor());
      if (scalar_allowlist.count(name) == 0 && aggregate_allowlist.count(name) == 0) {
        unsupported("Substrait function is outside the Sirius v1 function subset: " + name);
      }
    }
  }

  const std::string& function_name(std::uint32_t anchor)
  {
    const auto entry = functions_.find(anchor);
    if (entry == functions_.end()) { invalid("expression references an undeclared function"); }
    return entry->second;
  }

  void validate_expression(const ::substrait::Expression& expression)
  {
    switch (expression.rex_type_case()) {
      case ::substrait::Expression::kLiteral: validate_literal(expression.literal()); break;
      case ::substrait::Expression::kSelection: {
        const auto& selection = expression.selection();
        if (!selection.has_direct_reference() || !selection.has_root_reference() ||
            !selection.direct_reference().has_struct_field() ||
            selection.direct_reference().struct_field().has_child() ||
            selection.direct_reference().struct_field().field() < 0) {
          unsupported("only direct root struct-field selections are supported");
        }
        break;
      }
      case ::substrait::Expression::kScalarFunction: {
        static const std::unordered_set<std::string> allowed = {
          "and",  "or",          "not",       "equal",   "not_equal",   "lt",
          "lte",  "gt",          "gte",       "is_null", "is_not_null", "is_not_distinct_from",
          "add",  "subtract",    "multiply",  "divide",  "modulus",     "between",
          "like", "starts_with", "substring", "extract"};
        const auto& function = expression.scalar_function();
        const auto& name     = function_name(function.function_reference());
        if (allowed.count(name) == 0) {
          unsupported("aggregate function used as a scalar expression");
        }
        if (function.args_size() != 0 || function.options_size() != 0 ||
            !function.has_output_type()) {
          unsupported(
            "deprecated arguments, options, or missing scalar output types are unsupported");
        }
        validate_type(function.output_type());
        if (name == "extract") {
          // Keep this exact allowlist aligned with the pinned Substrait importer's
          // valid_extract_subfields. "millenium" is its compatibility spelling.
          static const std::unordered_set<std::string> units = {"year",
                                                                "month",
                                                                "day",
                                                                "decade",
                                                                "century",
                                                                "millenium",
                                                                "quarter",
                                                                "microsecond",
                                                                "milliseconds",
                                                                "second",
                                                                "minute",
                                                                "hour"};
          if (function.arguments_size() != 2 || !function.arguments(0).has_enum_() ||
              units.count(function.arguments(0).enum_()) == 0 ||
              !function.arguments(1).has_value()) {
            unsupported("extract requires one supported enum and one value argument");
          }
          validate_expression(function.arguments(1).value());
          break;
        }
        for (const auto& argument : function.arguments()) {
          if (!argument.has_value()) { unsupported("only value function arguments are supported"); }
          validate_expression(argument.value());
        }
        break;
      }
      case ::substrait::Expression::kCast: {
        const auto& cast = expression.cast();
        if (!cast.has_input() || !cast.has_type() ||
            cast.failure_behavior() == ::substrait::Expression_Cast::FAILURE_BEHAVIOR_RETURN_NULL) {
          unsupported("only throwing Substrait casts with explicit types are supported");
        }
        validate_expression(cast.input());
        validate_type(cast.type());
        break;
      }
      case ::substrait::Expression::kIfThen: {
        const auto& if_then = expression.if_then();
        if (if_then.ifs_size() == 0 || !if_then.has_else_()) {
          invalid("Substrait IfThen requires a branch and an else expression");
        }
        for (const auto& branch : if_then.ifs()) {
          if (!branch.has_if_() || !branch.has_then()) {
            invalid("Substrait IfThen branch is incomplete");
          }
          validate_expression(branch.if_());
          validate_expression(branch.then());
        }
        validate_expression(if_then.else_());
        break;
      }
      case ::substrait::Expression::kSingularOrList: {
        const auto& in = expression.singular_or_list();
        if (!in.has_value() || in.options_size() == 0) {
          invalid("Substrait SingularOrList requires a value and options");
        }
        validate_expression(in.value());
        for (const auto& option : in.options()) {
          validate_expression(option);
        }
        break;
      }
      default: unsupported("Substrait expression is outside the Sirius v1 expression subset");
    }
  }

  void validate_literal(const ::substrait::Expression_Literal& literal)
  {
    switch (literal.literal_type_case()) {
      case ::substrait::Expression_Literal::kBoolean:
      case ::substrait::Expression_Literal::kI8:
      case ::substrait::Expression_Literal::kI16:
      case ::substrait::Expression_Literal::kI32:
      case ::substrait::Expression_Literal::kI64:
      case ::substrait::Expression_Literal::kFp32:
      case ::substrait::Expression_Literal::kFp64:
      case ::substrait::Expression_Literal::kString:
      case ::substrait::Expression_Literal::kDate:
      case ::substrait::Expression_Literal::kVarChar: break;
      case ::substrait::Expression_Literal::kDecimal:
        if (literal.decimal().value().size() != 16 || literal.decimal().precision() <= 0 ||
            literal.decimal().precision() > 38 || literal.decimal().scale() < 0 ||
            literal.decimal().scale() > literal.decimal().precision()) {
          unsupported("Substrait decimal literal is outside the Sirius range");
        }
        break;
      case ::substrait::Expression_Literal::kPrecisionTimestamp:
        if (literal.precision_timestamp().precision() != 6) {
          unsupported("only microsecond Substrait timestamp literals are supported");
        }
        break;
      case ::substrait::Expression_Literal::kNull: validate_type(literal.null()); break;
      default: unsupported("Substrait literal is outside the Sirius v1 literal subset");
    }
  }

  void validate_sort_field(const ::substrait::SortField& sort)
  {
    if (!sort.has_expr() || sort.has_comparison_function_reference()) {
      unsupported("custom Substrait sort comparisons are not supported");
    }
    switch (sort.direction()) {
      case ::substrait::SortField::SORT_DIRECTION_ASC_NULLS_FIRST:
      case ::substrait::SortField::SORT_DIRECTION_ASC_NULLS_LAST:
      case ::substrait::SortField::SORT_DIRECTION_DESC_NULLS_FIRST:
      case ::substrait::SortField::SORT_DIRECTION_DESC_NULLS_LAST: break;
      default: unsupported("Substrait sort direction is unsupported");
    }
    validate_expression(sort.expr());
  }

  void validate_aggregate(::substrait::AggregateRel* aggregate)
  {
    if (!aggregate->has_input()) { invalid("aggregate relation has no input"); }
    validate_rel(aggregate->mutable_input());
    for (const auto& expression : aggregate->grouping_expressions()) {
      validate_expression(expression);
    }
    for (const auto& grouping : aggregate->groupings()) {
      for (const auto reference : grouping.expression_references()) {
        if (reference >= static_cast<std::uint32_t>(aggregate->grouping_expressions_size())) {
          invalid("aggregate grouping reference is out of range");
        }
      }
    }
    static const std::unordered_set<std::string> allowed = {"count", "sum", "min", "max", "avg"};
    for (const auto& measure_wrapper : aggregate->measures()) {
      if (!measure_wrapper.has_measure()) { invalid("aggregate measure is missing"); }
      const auto& measure = measure_wrapper.measure();
      if (allowed.count(function_name(measure.function_reference())) == 0) {
        unsupported("scalar function used as an aggregate measure");
      }
      if (measure.args_size() != 0 || measure.sorts_size() != 0 || measure.options_size() != 0 ||
          !measure.has_output_type()) {
        unsupported(
          "aggregate deprecated arguments, sorts, options, or missing output types are "
          "unsupported");
      }
      if (measure.phase() != ::substrait::AGGREGATION_PHASE_UNSPECIFIED &&
          measure.phase() != ::substrait::AGGREGATION_PHASE_INITIAL_TO_RESULT) {
        unsupported("multi-phase aggregates are not supported");
      }
      if (measure.invocation() !=
            ::substrait::AggregateFunction::AGGREGATION_INVOCATION_UNSPECIFIED &&
          measure.invocation() != ::substrait::AggregateFunction::AGGREGATION_INVOCATION_ALL) {
        unsupported("only non-distinct aggregate invocation is supported in Sirius v1 offload");
      }
      validate_type(measure.output_type());
      for (const auto& argument : measure.arguments()) {
        if (!argument.has_value()) { unsupported("only value aggregate arguments are supported"); }
        validate_expression(argument.value());
      }
      if (measure_wrapper.has_filter()) { validate_expression(measure_wrapper.filter()); }
    }
  }

  void validate_read(::substrait::ReadRel* read)
  {
    if (!read->has_extension_table() || !read->extension_table().has_detail()) {
      unsupported("Sirius offload reads must use an authenticated ExtensionTable read");
    }
    if (!read->has_base_schema()) { invalid("extension read relation has no base schema"); }
    if (read->has_filter() || read->has_best_effort_filter() || read->has_projection()) {
      unsupported("read-level filters and projections are not supported; use FilterRel/ProjectRel");
    }
    validate_schema(read->base_schema());
    const auto detail = read->extension_table().detail();
    if (detail.type_url() == k_tae_read_type_url) {
      validate_tae_relation(read, detail.value());
      return;
    }
    if (detail.type_url() == k_stream_read_type_url) {
      validate_stream_relation(read, detail.value());
      return;
    }
    unsupported("ExtensionTable detail is outside the MatrixOne read contract");
  }

  void rewrite_read(::substrait::ReadRel* read, resolved_read& resolution)
  {
    if (resolution.relation_name().empty()) {
      fail(substrait_error_code::READ_RESOLUTION_FAILED,
           "read resolver returned no query-local relation");
    }
    validate_message_tree(resolution.canonical_schema());
    validate_schema(resolution.canonical_schema());
    read->mutable_base_schema()->CopyFrom(resolution.canonical_schema());
    auto* named = read->mutable_named_table();
    named->clear_names();
    named->add_names(resolution.relation_name());
  }

  void validate_tae_relation(::substrait::ReadRel* read, std::string_view value)
  {
    auto request = parse_tae_read(value);
    validate_tae_read(request, now_unix_ms_);

    std::unique_ptr<resolved_tae_read> resolution;
    try {
      resolution = resolver_.resolve(request, read->base_schema());
    } catch (const substrait_execution_error&) {
      throw;
    } catch (const std::exception& error) {
      fail(substrait_error_code::READ_RESOLUTION_FAILED,
           std::string("TaeRead resolver failed: ") + error.what());
    } catch (...) {
      fail(substrait_error_code::READ_RESOLUTION_FAILED, "TaeRead resolver failed");
    }
    if (!resolution) {
      fail(substrait_error_code::READ_RESOLUTION_FAILED, "TaeRead resolver returned no resolution");
    }
    const bool authenticated =
      resolution->read_ref() == request.read_ref && resolution->query_id() == request.query_id &&
      resolution->account_id() == request.account_id &&
      resolution->database_id() == request.database_id &&
      resolution->table_id() == request.table_id &&
      resolution->snapshot_ts() == request.snapshot_ts &&
      resolution->schema_digest() == request.schema_digest &&
      resolution->manifest_sha256() == request.manifest_sha256 &&
      resolution->capability_hash() == request.capability_hash &&
      resolution->expires_at_unix_ms() == request.expires_at_unix_ms &&
      resolution->canonical_schema().SerializeAsString() == read->base_schema().SerializeAsString();
    if (!authenticated) {
      fail(substrait_error_code::AUTHENTICATION_FAILED,
           "TaeRead resolver metadata does not match the signed request");
    }

    rewrite_read(read, *resolution);
    resolutions_.push_back(std::move(resolution));
  }

  void validate_stream_relation(::substrait::ReadRel* read, std::string_view value)
  {
    auto request = parse_stream_read(value);
    validate_stream_read(request, now_unix_ms_);

    std::unique_ptr<resolved_stream_read> resolution;
    try {
      resolution = resolver_.resolve(request, read->base_schema());
    } catch (const substrait_execution_error&) {
      throw;
    } catch (const std::exception& error) {
      fail(substrait_error_code::READ_RESOLUTION_FAILED,
           std::string("StreamRead resolver failed: ") + error.what());
    } catch (...) {
      fail(substrait_error_code::READ_RESOLUTION_FAILED, "StreamRead resolver failed");
    }
    if (!resolution) {
      fail(substrait_error_code::READ_RESOLUTION_FAILED,
           "StreamRead resolver returned no resolution");
    }
    const bool authenticated =
      resolution->stream_ref() == request.stream_ref &&
      resolution->query_id() == request.query_id &&
      resolution->account_id() == request.account_id &&
      resolution->snapshot_ts() == request.snapshot_ts &&
      resolution->schema_digest() == request.schema_digest &&
      resolution->capability_hash() == request.capability_hash &&
      resolution->expires_at_unix_ms() == request.expires_at_unix_ms &&
      resolution->canonical_schema().SerializeAsString() == read->base_schema().SerializeAsString();
    if (!authenticated) {
      fail(substrait_error_code::AUTHENTICATION_FAILED,
           "StreamRead resolver metadata does not match the signed request");
    }

    rewrite_read(read, *resolution);
    resolutions_.push_back(std::move(resolution));
  }

  void validate_rel(::substrait::Rel* rel)
  {
    switch (rel->rel_type_case()) {
      case ::substrait::Rel::kRead: validate_read(rel->mutable_read()); break;
      case ::substrait::Rel::kFilter:
        if (!rel->filter().has_input() || !rel->filter().has_condition()) {
          invalid("filter relation is incomplete");
        }
        validate_rel(rel->mutable_filter()->mutable_input());
        validate_expression(rel->filter().condition());
        break;
      case ::substrait::Rel::kProject:
        if (!rel->project().has_input()) { invalid("project relation has no input"); }
        validate_rel(rel->mutable_project()->mutable_input());
        for (const auto& expression : rel->project().expressions()) {
          validate_expression(expression);
        }
        break;
      case ::substrait::Rel::kAggregate: validate_aggregate(rel->mutable_aggregate()); break;
      case ::substrait::Rel::kSort:
        if (!rel->sort().has_input()) { invalid("sort relation has no input"); }
        validate_rel(rel->mutable_sort()->mutable_input());
        for (const auto& sort : rel->sort().sorts()) {
          validate_sort_field(sort);
        }
        break;
      case ::substrait::Rel::kFetch:
        if (!rel->fetch().has_input()) { invalid("fetch relation has no input"); }
        if (rel->fetch().has_offset_expr() || rel->fetch().has_count_expr() ||
            rel->fetch().offset() < 0 || rel->fetch().count() < -1) {
          unsupported("only non-negative integer FetchRel offset/count values are supported");
        }
        validate_rel(rel->mutable_fetch()->mutable_input());
        break;
      case ::substrait::Rel::kJoin: {
        const auto& join = rel->join();
        switch (join.type()) {
          case ::substrait::JoinRel::JOIN_TYPE_INNER:
          case ::substrait::JoinRel::JOIN_TYPE_LEFT:
          case ::substrait::JoinRel::JOIN_TYPE_RIGHT:
          case ::substrait::JoinRel::JOIN_TYPE_LEFT_SEMI:
          case ::substrait::JoinRel::JOIN_TYPE_LEFT_ANTI:
          case ::substrait::JoinRel::JOIN_TYPE_RIGHT_SEMI:
          case ::substrait::JoinRel::JOIN_TYPE_RIGHT_ANTI: break;
          default: unsupported("Substrait join type is outside the Sirius TPC-H subset");
        }
        if (!join.has_left() || !join.has_right() || !join.has_expression()) {
          invalid("Substrait join relation is incomplete");
        }
        if (join.has_post_join_filter()) {
          unsupported("Substrait post-join filters are not supported; use FilterRel");
        }
        validate_rel(rel->mutable_join()->mutable_left());
        validate_rel(rel->mutable_join()->mutable_right());
        validate_expression(join.expression());
        break;
      }
      case ::substrait::Rel::kReference:
        if (rel->reference().subtree_ordinal() >= current_relation_ordinal_) {
          invalid("Substrait ReferenceRel must refer to an earlier top-level relation");
        }
        break;
      default: unsupported("Substrait relation is outside the Sirius v1 operator subset");
    }
  }

  ::substrait::Plan& plan_;
  tae_read_resolver& resolver_;
  std::uint64_t now_unix_ms_;
  std::unordered_map<std::uint32_t, std::string> functions_;
  std::vector<std::unique_ptr<resolved_read>> resolutions_;
  std::uint32_t current_relation_ordinal_ = 0;
};

}  // namespace

substrait_execution_error::substrait_execution_error(substrait_error_code code, std::string message)
  : std::runtime_error(std::move(message)), code_(code)
{
}

substrait_execution::substrait_execution(
  duckdb::ClientContext& context,
  duckdb::shared_ptr<sirius_prepared_statement_data> prepared,
  execution_schema schema,
  std::shared_ptr<execution_evidence> evidence,
  std::vector<std::unique_ptr<resolved_read>> resolutions)
  : context_(context),
    prepared_(std::move(prepared)),
    schema_(std::move(schema)),
    evidence_(std::move(evidence)),
    resolutions_(std::move(resolutions))
{
}

bool substrait_execution::transition(execution_state expected, execution_state desired) noexcept
{
  return state_.compare_exchange_strong(expected, desired);
}

void substrait_execution::release_resolutions() noexcept
{
  std::lock_guard<std::mutex> guard(resolutions_mutex_);
  resolutions_.clear();
}

void substrait_execution::cancel() noexcept
{
  cancel_requested_.store(true);
  if (transition(execution_state::PREPARED, execution_state::CANCELLED)) { release_resolutions(); }
}

void substrait_execution::run(const chunk_consumer& consumer)
{
  if (!consumer) { invalid("Substrait execution requires a chunk consumer"); }
  if (!transition(execution_state::PREPARED, execution_state::RUNNING)) {
    if (state() == execution_state::CANCELLED) {
      fail(substrait_error_code::CANCELLED, "Substrait execution was cancelled before start");
    }
    fail(substrait_error_code::EXECUTION_FAILED, "Substrait execution handles are single-use");
  }

  try {
    if (cancel_requested_.load()) {
      fail(substrait_error_code::CANCELLED, "Substrait execution was cancelled before start");
    }
    sirius_interface interface(context_, evidence_);
    interface.sirius_execute_streaming(
      context_, "[matrixone substrait offload]", prepared_, [this, &consumer](const auto& chunk) {
        if (cancel_requested_.load()) { return false; }
        if (consumer(chunk) == chunk_action::CANCEL) {
          cancel_requested_.store(true);
          return false;
        }
        return !cancel_requested_.load();
      });
    if (cancel_requested_.load()) {
      fail(substrait_error_code::CANCELLED, "Substrait execution was cancelled");
    }
    (void)transition(execution_state::RUNNING, execution_state::SUCCEEDED);
    release_resolutions();
  } catch (const substrait_execution_error& error) {
    (void)transition(execution_state::RUNNING,
                     error.code() == substrait_error_code::CANCELLED ? execution_state::CANCELLED
                                                                     : execution_state::FAILED);
    release_resolutions();
    throw;
  } catch (const std::exception& error) {
    const bool cancelled = cancel_requested_.load();
    (void)transition(execution_state::RUNNING,
                     cancelled ? execution_state::CANCELLED : execution_state::FAILED);
    release_resolutions();
    fail(cancelled ? substrait_error_code::CANCELLED : substrait_error_code::EXECUTION_FAILED,
         cancelled ? "Substrait execution was cancelled"
                   : std::string("Sirius execution failed: ") + error.what());
  } catch (...) {
    const bool cancelled = cancel_requested_.load();
    (void)transition(execution_state::RUNNING,
                     cancelled ? execution_state::CANCELLED : execution_state::FAILED);
    release_resolutions();
    fail(cancelled ? substrait_error_code::CANCELLED : substrait_error_code::EXECUTION_FAILED,
         cancelled ? "Substrait execution was cancelled" : "Sirius execution failed");
  }
}

void substrait_execution::run_batches(const batch_consumer& consumer)
{
  if (!consumer) { invalid("Substrait execution requires a batch consumer"); }
  if (!transition(execution_state::PREPARED, execution_state::RUNNING)) {
    if (state() == execution_state::CANCELLED) {
      fail(substrait_error_code::CANCELLED, "Substrait execution was cancelled before start");
    }
    fail(substrait_error_code::EXECUTION_FAILED, "Substrait execution handles are single-use");
  }
  try {
    if (cancel_requested_.load()) {
      fail(substrait_error_code::CANCELLED, "Substrait execution was cancelled before start");
    }
    sirius_interface interface(context_, evidence_);
    interface.sirius_execute_batch_streaming(
      context_,
      "[matrixone substrait offload]",
      prepared_,
      [this, &consumer](const auto& batch, auto stream) {
        if (cancel_requested_.load()) { return false; }
        if (consumer(batch, stream) == chunk_action::CANCEL) {
          cancel_requested_.store(true);
          return false;
        }
        return !cancel_requested_.load();
      });
    if (cancel_requested_.load()) {
      fail(substrait_error_code::CANCELLED, "Substrait execution was cancelled");
    }
    (void)transition(execution_state::RUNNING, execution_state::SUCCEEDED);
    release_resolutions();
  } catch (const substrait_execution_error& error) {
    (void)transition(execution_state::RUNNING,
                     error.code() == substrait_error_code::CANCELLED ? execution_state::CANCELLED
                                                                     : execution_state::FAILED);
    release_resolutions();
    throw;
  } catch (const std::exception& error) {
    const bool cancelled = cancel_requested_.load();
    (void)transition(execution_state::RUNNING,
                     cancelled ? execution_state::CANCELLED : execution_state::FAILED);
    release_resolutions();
    fail(cancelled ? substrait_error_code::CANCELLED : substrait_error_code::EXECUTION_FAILED,
         cancelled ? "Substrait execution was cancelled"
                   : std::string("Sirius execution failed: ") + error.what());
  } catch (...) {
    const bool cancelled = cancel_requested_.load();
    (void)transition(execution_state::RUNNING,
                     cancelled ? execution_state::CANCELLED : execution_state::FAILED);
    release_resolutions();
    fail(cancelled ? substrait_error_code::CANCELLED : substrait_error_code::EXECUTION_FAILED,
         cancelled ? "Substrait execution was cancelled" : "Sirius execution failed");
  }
}

namespace detail {

validated_substrait_plan validate_and_resolve_substrait(std::string_view binary_plan,
                                                        tae_read_resolver& resolver,
                                                        std::uint64_t now_unix_ms)
{
  if (binary_plan.empty() || binary_plan.size() > k_max_substrait_plan_bytes) {
    invalid("Substrait plan is empty or exceeds the 16 MiB limit");
  }
  ::substrait::Plan plan;
  if (!plan.ParseFromArray(binary_plan.data(), static_cast<int>(binary_plan.size()))) {
    invalid("Substrait plan protobuf is malformed");
  }
  validate_message_tree(plan);
  plan_validator validator(plan, resolver, now_unix_ms == 0 ? system_now_unix_ms() : now_unix_ms);
  auto resolutions = validator.validate();
  std::string serialized;
  if (!plan.SerializeToString(&serialized)) {
    invalid("failed to serialize validated Substrait plan");
  }
  return {std::move(serialized), std::move(resolutions)};
}

}  // namespace detail

std::unique_ptr<substrait_execution> prepare_substrait(duckdb::ClientContext& context,
                                                       std::string_view binary_plan,
                                                       tae_read_resolver& resolver,
                                                       std::shared_ptr<execution_evidence> evidence)
{
  auto validated = detail::validate_and_resolve_substrait(binary_plan, resolver);
  try {
    auto context_ptr = context.shared_from_this();
    duckdb::SubstraitToDuckDB transformer(context_ptr, validated.serialized, false, true);
    auto relation = transformer.TransformPlan();
    auto binder   = duckdb::Binder::CreateBinder(context);
    auto bound    = relation->Bind(*binder);
    if (!bound.plan || bound.names.size() != bound.types.size()) {
      invalid("Substrait relation binding produced an invalid result schema");
    }
    if (context.config.enable_optimizer) {
      duckdb::Optimizer optimizer(*binder, context);
      bound.plan = optimizer.Optimize(std::move(bound.plan));
    }
    bound.plan->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver binding_resolver;
    duckdb::ColumnBindingResolver::Verify(*bound.plan);
    binding_resolver.VisitOperator(*bound.plan);

    auto prepared = duckdb::make_shared_ptr<duckdb::PreparedStatementData>(
      duckdb::StatementType::SELECT_STATEMENT);
    prepared->names = bound.names;
    prepared->types = bound.types;
    execution_schema schema{bound.names, bound.types};
    auto sirius_prepared =
      prepare_sirius_statement(context, std::move(prepared), std::move(bound.plan));
    return std::make_unique<substrait_execution>(context,
                                                 std::move(sirius_prepared),
                                                 std::move(schema),
                                                 std::move(evidence),
                                                 std::move(validated.resolutions));
  } catch (const substrait_execution_error&) {
    throw;
  } catch (const duckdb::NotImplementedException& error) {
    unsupported(std::string("Substrait plan is not supported by Sirius: ") + error.what());
  } catch (const std::exception& error) {
    invalid(std::string("Substrait plan could not be bound for Sirius: ") + error.what());
  }
}

}  // namespace sirius::offload
