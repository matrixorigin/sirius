/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "numeric/decimal_types.hpp"

#include "numeric/exact_decimal.hpp"

#include <charconv>
#include <cstring>
#include <string_view>

namespace sirius::mo_decimal {
namespace {
constexpr std::string_view prefix = "__sirius_mo_exact_v1_";
std::string alias(logical_type const& type)
{
  auto descriptor = type.is_mo_decimal() ? type.mo_decimal_type() : decimal_type{};
  auto tag        = type.is_mo_decimal() ? "d" : type.id() == type_id::BOOLEAN ? "b" : "i";
  return std::string(prefix) + tag + "_" + std::to_string(descriptor.bits) + "_" +
         std::to_string(descriptor.precision) + "_" + std::to_string(descriptor.scale) + "_" +
         std::to_string(type.nullability());
}
duckdb::LogicalType carrier(logical_type const& type)
{
  if (type.id() == type_id::BOOLEAN) return duckdb::LogicalType::BOOLEAN;
  if (type.id() == type_id::BIGINT) return duckdb::LogicalType::BIGINT;
  auto bits = type.mo_decimal_type().bits;
  if (bits == 64) return duckdb::LogicalType::BIGINT;
  if (bits == 128) return duckdb::LogicalType::HUGEINT;
  return duckdb::LogicalType::STRUCT({{"high", duckdb::LogicalType::BIGINT},
                                      {"mid_high", duckdb::LogicalType::UBIGINT},
                                      {"mid_low", duckdb::LogicalType::UBIGINT},
                                      {"low", duckdb::LogicalType::UBIGINT}});
}
}  // namespace

duckdb::LogicalType duckdb_type(logical_type const& type)
{
  if (!type.nullability() ||
      (!type.is_mo_decimal() && type.id() != type_id::BOOLEAN && type.id() != type_id::BIGINT))
    throw std::invalid_argument("invalid MO exact-decimal carrier");
  auto result = carrier(type);
  result.SetAlias(alias(type));
  return result;
}
std::optional<logical_type> from_duckdb_type(duckdb::LogicalType const& type)
{
  auto const& name = type.GetAlias();
  if (!std::string_view(name).starts_with(prefix)) return std::nullopt;
  auto tail = std::string_view(name).substr(prefix.size());
  if (tail.size() < 3 || tail[1] != '_')
    throw std::invalid_argument("invalid MO exact-decimal carrier alias");
  auto tag = tail[0];
  tail.remove_prefix(2);
  uint32_t fields[4]{};
  for (int i = 0; i < 4; ++i) {
    auto end = i == 3 ? tail.size() : tail.find('_');
    if (end == std::string_view::npos || end == 0)
      throw std::invalid_argument("invalid MO exact-decimal carrier parameters");
    auto text          = tail.substr(0, end);
    auto [last, error] = std::from_chars(text.data(), text.data() + text.size(), fields[i]);
    if (error != std::errc{} || last != text.data() + text.size())
      throw std::invalid_argument("invalid MO exact-decimal carrier parameter");
    tail.remove_prefix(end + (i == 3 ? 0 : 1));
  }
  if (!tail.empty() || (fields[3] != 1 && fields[3] != 2) || fields[0] > 256 || fields[1] > 76 ||
      fields[2] > 76)
    throw std::invalid_argument("invalid MO exact-decimal carrier domain");
  logical_type result;
  if (tag == 'd')
    result = logical_type::make_mo_decimal({static_cast<uint16_t>(fields[0]),
                                            static_cast<uint8_t>(fields[1]),
                                            static_cast<uint8_t>(fields[2])},
                                           fields[3] == 2);
  else if ((tag == 'b' || tag == 'i') && !fields[0] && !fields[1] && !fields[2])
    result = logical_type::make(tag == 'b' ? type_id::BOOLEAN : type_id::BIGINT)
               .with_nullability(fields[3] == 2);
  else
    throw std::invalid_argument("invalid MO exact-decimal carrier kind");
  if (duckdb_type(result) != type)
    throw std::invalid_argument("noncanonical MO exact-decimal carrier");
  return result;
}
duckdb::Value duckdb_value(coefficient const& value, logical_type const& type)
{
  auto descriptor = type.mo_decimal_type();
  auto checked =
    evaluate_decimal(decimal_op::cast, value, descriptor, true, {}, descriptor, true, descriptor);
  if (checked.error != decimal_error::none)
    throw std::invalid_argument("MO exact-decimal literal exceeds its declared domain");
  uint64_t limbs[4]{};
  for (int i = 0; i < 4; ++i)
    limbs[i] = uint64_t(value.words[2 * i]) | uint64_t(value.words[2 * i + 1]) << 32;
  duckdb::Value result;
  if (descriptor.bits == 64) {
    int64_t signed_value;
    std::memcpy(&signed_value, limbs, 8);
    result = duckdb::Value::BIGINT(signed_value);
  } else if (descriptor.bits == 128) {
    int64_t high;
    std::memcpy(&high, limbs + 1, 8);
    result = duckdb::Value::HUGEINT(duckdb::hugeint_t{high, limbs[0]});
  } else {
    int64_t high;
    std::memcpy(&high, limbs + 3, 8);
    result = duckdb::Value::STRUCT(carrier(type),
                                   {duckdb::Value::BIGINT(high),
                                    duckdb::Value::UBIGINT(limbs[2]),
                                    duckdb::Value::UBIGINT(limbs[1]),
                                    duckdb::Value::UBIGINT(limbs[0])});
  }
  result.Reinterpret(duckdb_type(type));
  return result;
}
coefficient from_duckdb_value(duckdb::Value const& value, logical_type const& type)
{
  if (value.IsNull() || value.type() != duckdb_type(type))
    throw std::invalid_argument("MO exact-decimal literal carrier mismatch");
  auto descriptor = type.mo_decimal_type();
  uint64_t limbs[4]{};
  if (descriptor.bits == 64) {
    auto signed_value = value.GetValueUnsafe<int64_t>();
    std::memcpy(limbs, &signed_value, 8);
  } else if (descriptor.bits == 128) {
    auto big = value.GetValueUnsafe<duckdb::hugeint_t>();
    limbs[0] = big.lower;
    std::memcpy(limbs + 1, &big.upper, 8);
  } else {
    auto const& children = duckdb::StructValue::GetChildren(value);
    for (int i = 0; i < 4; ++i) {
      if (children[i].IsNull()) throw std::invalid_argument("NULL MO decimal coefficient limb");
      if (i == 0) {
        auto high = children[i].GetValueUnsafe<int64_t>();
        std::memcpy(limbs + 3, &high, 8);
      } else
        limbs[3 - i] = children[i].GetValueUnsafe<uint64_t>();
    }
  }
  return load_coefficient(reinterpret_cast<uint8_t const*>(limbs), descriptor.bytes());
}
}  // namespace sirius::mo_decimal
