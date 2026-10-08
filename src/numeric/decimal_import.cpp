/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "numeric/decimal_import.hpp"

#include "from_substrait.hpp"
#include "numeric/decimal_aggregate_bind.hpp"
#include "numeric/decimal_functions.hpp"
#include "numeric/exact_decimal.hpp"

#include <duckdb/parser/expression/constant_expression.hpp>
#include <duckdb/parser/expression/function_expression.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite.h>

#include <algorithm>
#include <array>

namespace sirius::mo_decimal {
namespace {
bool nullable(substrait::Type_Nullability value)
{
  if (value != substrait::Type::NULLABILITY_NULLABLE &&
      value != substrait::Type::NULLABILITY_REQUIRED)
    throw std::invalid_argument("MO numeric type requires explicit nullability");
  return value == substrait::Type::NULLABILITY_NULLABLE;
}
template <class UserType>
decimal_type descriptor(UserType const& user)
{
  if (user.type_parameters_size() != 3)
    throw std::invalid_argument("MO numeric type requires width precision and scale");
  int64_t parameters[3]{};
  for (int i = 0; i < 3; ++i) {
    if (!user.type_parameters(i).has_integer())
      throw std::invalid_argument("MO numeric type parameters must be integers");
    parameters[i] = user.type_parameters(i).integer();
  }
  if ((parameters[0] != 64 && parameters[0] != 128 && parameters[0] != 256) || parameters[1] < 1 ||
      parameters[1] > 76 || parameters[2] < 0 || parameters[2] > 76)
    throw std::invalid_argument("invalid MO numeric type domain");
  decimal_type result{static_cast<uint16_t>(parameters[0]),
                      static_cast<uint8_t>(parameters[1]),
                      static_cast<uint8_t>(parameters[2])};
  if (!result.valid()) throw std::invalid_argument("invalid MO numeric precision or scale");
  return result;
}
logical_type result_type(substrait::Type const& type, std::vector<uint32_t> const& anchors)
{
  if (type.has_user_defined()) {
    if (!std::binary_search(anchors.begin(), anchors.end(), type.user_defined().type_reference()))
      throw std::invalid_argument("MO numeric result references another type identity");
    return logical_type::make_mo_decimal(descriptor(type.user_defined()),
                                         nullable(type.user_defined().nullability()));
  }
  if (type.has_bool_())
    return logical_type::make(type_id::BOOLEAN)
      .with_nullability(nullable(type.bool_().nullability()));
  if (type.has_i64())
    return logical_type::make(type_id::BIGINT).with_nullability(nullable(type.i64().nullability()));
  throw std::invalid_argument("unsupported MO decimal result type");
}
class importer final : public duckdb::SubstraitExtensionHandler {
  std::vector<uint32_t> anchors_;

 public:
  explicit importer(std::vector<uint32_t> anchors) : anchors_(std::move(anchors)) {}
  bool IsOpaqueType(duckdb::LogicalType const& type) const override
  {
    auto exact = from_duckdb_type(type);
    return exact && exact->is_mo_decimal();
  }
  bool Handles(duckdb::SubstraitExtensionIdentity const& identity) const override
  {
    return identity.urn == extension_uri;
  }
  duckdb::LogicalType Type(duckdb::ClientContext&,
                           duckdb::SubstraitExtensionIdentity const& identity,
                           substrait::Type const& type) const override
  {
    if (!Handles(identity) || identity.name != "mo_exact_decimal" || !type.has_user_defined())
      throw std::invalid_argument("unhandled MO numeric type identity");
    return duckdb_type(result_type(type, anchors_));
  }
  duckdb::unique_ptr<duckdb::ParsedExpression> Literal(
    duckdb::ClientContext&,
    duckdb::SubstraitExtensionIdentity const& identity,
    substrait::Expression_Literal const& literal) const override
  {
    if (!Handles(identity) || identity.name != "mo_exact_decimal" || !literal.has_user_defined())
      throw std::invalid_argument("unhandled MO numeric literal identity");
    auto const& user    = literal.user_defined();
    auto type           = descriptor(user);
    auto const& any     = user.value();
    auto const& payload = any.value();
    if (any.type_url() != literal_type_url || payload.size() > (16u << 20))
      throw std::invalid_argument("invalid MO exact-decimal literal encoding");
    // Parse the approved protobuf using the pinned reader, including ordinary
    // unknown-field and singular-field semantics. Only the final coefficient
    // is retained, in bounded stack storage; no numeric strings are parsed.
    duckdb::google::protobuf::io::CodedInputStream input(
      reinterpret_cast<uint8_t const*>(payload.data()), static_cast<int>(payload.size()));
    std::array<uint8_t, 32> coefficient{};
    bool found = false, width_matches = false;
    while (auto tag = input.ReadTag()) {
      if (!(tag >> 3)) throw std::invalid_argument("invalid MO decimal literal field");
      if (tag == 10) {
        uint32_t bytes;
        if (!input.ReadVarint32(&bytes) || bytes > payload.size())
          throw std::invalid_argument("invalid MO decimal literal length");
        found         = true;
        width_matches = bytes == type.bytes();
        if (!(width_matches ? input.ReadRaw(coefficient.data(), bytes)
                            : input.Skip(static_cast<int>(bytes))))
          throw std::invalid_argument("truncated MO decimal literal coefficient");
      } else if (!duckdb::google::protobuf::internal::WireFormatLite::SkipField(&input, tag))
        throw std::invalid_argument("invalid MO decimal literal payload");
    }
    if (!found || !width_matches || !input.ConsumedEntireMessage())
      throw std::invalid_argument("MO decimal literal coefficient width mismatch");
    auto value = load_coefficient(coefficient.data(), type.bytes());
    return duckdb::make_uniq<duckdb::ConstantExpression>(
      duckdb_value(value, logical_type::make_mo_decimal(type, literal.nullable())));
  }
  duckdb::unique_ptr<duckdb::ParsedExpression> Scalar(
    duckdb::ClientContext&,
    duckdb::SubstraitExtensionIdentity const& identity,
    substrait::Expression_ScalarFunction const& scalar,
    duckdb::vector<duckdb::unique_ptr<duckdb::ParsedExpression>> children) const override
  {
    if (!Handles(identity) || !scalar.has_output_type() ||
        scalar.arguments_size() != static_cast<int>(children.size()) || scalar.options_size())
      throw std::invalid_argument("invalid MO exact-decimal scalar signature");
    auto id = from_duckdb_function_name("__sirius_" + identity.name);
    if (!id || !is_decimal_function(*id))
      throw std::invalid_argument("unknown MO exact-decimal scalar identity");
    auto op = operation(*id);
    if (children.size() != (op == decimal_op::cast || op == decimal_op::negate ? 1u : 2u))
      throw std::invalid_argument("invalid MO exact-decimal scalar arity");
    auto output = result_type(scalar.output_type(), anchors_);
    children.push_back(
      duckdb::make_uniq<duckdb::ConstantExpression>(duckdb::Value(duckdb_type(output))));
    return duckdb::make_uniq<duckdb::FunctionExpression>(std::string(to_duckdb_function_name(*id)),
                                                         std::move(children));
  }
  duckdb::unique_ptr<duckdb::ParsedExpression> Aggregate(
    duckdb::ClientContext&,
    duckdb::SubstraitExtensionIdentity const& identity,
    substrait::AggregateFunction const& aggregate,
    duckdb::vector<duckdb::unique_ptr<duckdb::ParsedExpression>> children) const override
  {
    if (!Handles(identity) || !aggregate.has_output_type() || children.size() != 1 ||
        aggregate.arguments_size() != 1 || aggregate.options_size() || aggregate.sorts_size() ||
        aggregate.invocation() != substrait::AggregateFunction::AGGREGATION_INVOCATION_ALL ||
        aggregate.phase() != substrait::AGGREGATION_PHASE_INITIAL_TO_RESULT)
      throw std::invalid_argument("unsupported MO exact-decimal aggregate signature or modifiers");
    auto id = from_duckdb_aggregate_name("__sirius_" + identity.name);
    if (!id || !is_decimal_aggregate(*id))
      throw std::invalid_argument("unknown MO aggregate identity");
    auto output = result_type(aggregate.output_type(), anchors_);
    children.push_back(
      duckdb::make_uniq<duckdb::ConstantExpression>(duckdb::Value(duckdb_type(output))));
    return duckdb::make_uniq<duckdb::FunctionExpression>(std::string(to_duckdb_aggregate_name(*id)),
                                                         std::move(children));
  }
};
}  // namespace
logical_type import_result_type(substrait::Type const& type, std::vector<uint32_t> const& anchors)
{
  return result_type(type, anchors);
}
bool uses_exact_decimal(std::string_view bytes)
{
  if (bytes.size() > (16u << 20)) throw std::invalid_argument("MO numeric plan exceeds 16 MiB");
  substrait::Plan plan;
  if (!plan.ParseFromArray(bytes.data(), static_cast<int>(bytes.size())))
    throw std::invalid_argument("invalid MO numeric plan");
  for (auto const& uri : plan.extension_urns())
    if (uri.urn() == extension_uri) return true;
  return false;
}
duckdb::shared_ptr<duckdb::SubstraitExtensionHandler> make_import_handler(std::string_view bytes)
{
  if (bytes.size() > (16u << 20)) throw std::invalid_argument("MO numeric plan exceeds 16 MiB");
  substrait::Plan plan;
  if (!plan.ParseFromArray(bytes.data(), static_cast<int>(bytes.size())))
    throw std::invalid_argument("invalid MO numeric plan");
  // Retain only this extension's anchors, not copies of unrelated URI strings.
  // The converter still validates all URI/type/function declarations.
  std::vector<uint32_t> urns;
  for (auto const& entry : plan.extension_urns())
    if (entry.urn() == extension_uri) urns.push_back(entry.extension_urn_anchor());
  std::sort(urns.begin(), urns.end());
  if (std::adjacent_find(urns.begin(), urns.end()) != urns.end())
    throw std::invalid_argument("duplicate MO numeric extension URI anchor");
  std::vector<uint32_t> anchors;
  for (auto const& extension : plan.extensions()) {
    if (!extension.has_extension_type()) continue;
    auto const& type = extension.extension_type();
    if (std::binary_search(urns.begin(), urns.end(), type.extension_urn_reference()) &&
        type.name() == "mo_exact_decimal")
      anchors.push_back(type.type_anchor());
  }
  std::sort(anchors.begin(), anchors.end());
  if (std::adjacent_find(anchors.begin(), anchors.end()) != anchors.end())
    throw std::invalid_argument("duplicate MO numeric type anchor");
  return duckdb::make_shared_ptr<importer>(std::move(anchors));
}
}  // namespace sirius::mo_decimal
