/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/input.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>

namespace sirius::embedding {
static_assert(std::endian::native == std::endian::little);
namespace {
void require(bool value, const char* message)
{
  if (!value) throw failure(SIRIUS_INVALID_ARGUMENT, message);
}
void range(std::size_t offset, std::size_t size, std::size_t capacity)
{
  require(offset <= capacity && size <= capacity - offset, "native input range outside lease");
}
template <class T>
T load(input_storage const& data, std::size_t offset)
{
  T value{};
  data.read(offset, {reinterpret_cast<std::byte*>(&value), sizeof(value)});
  return value;
}
}  // namespace
void input_storage::write(std::size_t offset, std::span<const std::byte> bytes)
{
  range(offset, bytes.size(), size());
  auto const end = offset + bytes.size();
  visit([&](std::size_t start, std::span<std::byte> block) {
    auto lo = std::max(start, offset), hi = std::min(start + block.size(), end);
    if (lo < hi) std::memcpy(block.data() + lo - start, bytes.data() + lo - offset, hi - lo);
  });
}
void input_storage::read(std::size_t offset, std::span<std::byte> bytes) const
{
  range(offset, bytes.size(), size());
  auto const end = offset + bytes.size();
  const_cast<input_storage*>(this)->visit([&](std::size_t start, std::span<std::byte> block) {
    auto lo = std::max(start, offset), hi = std::min(start + block.size(), end);
    if (lo < hi) std::memcpy(bytes.data() + lo - offset, block.data() + lo - start, hi - lo);
  });
}
bool input_string_type(uint32_t oid) { return oid == 60 || oid == 61 || oid == 70 || oid == 71; }
std::size_t input_element_size(uint32_t oid)
{
  switch (oid) {
    case 10:
    case 20:
    case 25: return 1;
    case 21:
    case 26: return 2;
    case 22:
    case 27:
    case 30:
    case 50: return 4;
    case 23:
    case 28:
    case 31:
    case 32:
    case 52: return 8;
    case 33: return 16;
    default:
      if (input_string_type(oid)) return 24;
  }
  throw failure(SIRIUS_UNSUPPORTED, "unsupported native MO column type");
}
void validate_input_schema(std::span<const sirius_input_column> columns)
{
  require(columns.size() <= input_columns_limit, "too many native input columns");
  for (auto const& c : columns) {
    (void)input_element_size(c.oid);
    require(c.nullable <= 1, "invalid native column nullability");
    if (c.oid == 32 || c.oid == 33)
      require(
        c.width >= 1 && c.width <= (c.oid == 32 ? 18 : 38) && c.scale >= 0 && c.scale <= c.width,
        "invalid native decimal precision or scale");
  }
}
input_batch::~input_batch()
{
  // Return pool blocks before announcing capacity to a blocked producer.
  storage.reset();
  credit.reset();
  if (owner && !published) owner->release_filling();
}
void input_batch::write(std::size_t offset, std::span<const std::byte> bytes)
{
  require(!published, "published input is immutable");
  range(offset, bytes.size(), payload_bytes);
  storage->write(offset, bytes);
}
bool input_batch::is_null(std::size_t column, uint32_t row) const
{
  auto const& c = columns[column];
  if (c.vector_class == SIRIUS_VECTOR_NULL) return true;
  if (c.vector_class == SIRIUS_VECTOR_CONSTANT) row = 0;
  auto byte = row / 8;
  return byte < c.null_bytes && (load<uint8_t>(*storage, c.null_offset + byte) & (1u << (row % 8)));
}
std::pair<std::size_t, std::size_t> input_batch::string_range(std::size_t column,
                                                              uint32_t row) const
{
  if (is_null(column, row)) return {0, 0};
  auto const& c = columns[column];
  if (c.vector_class == SIRIUS_VECTOR_CONSTANT) row = 0;
  auto start  = c.data_offset + static_cast<std::size_t>(row) * 24;
  auto length = load<uint8_t>(*storage, start);
  if (length <= 23) return {start + 1, length};
  require(load<uint32_t>(*storage, start) == UINT32_MAX, "invalid native varlena tag");
  auto offset = load<uint32_t>(*storage, start + 4);
  auto size   = load<uint32_t>(*storage, start + 8);
  range(offset, size, c.area_bytes);
  return {c.area_offset + offset, size};
}
native_input::native_input(uint64_t binding,
                           std::vector<sirius_input_column> columns,
                           std::stop_token stop,
                           clock::time_point deadline,
                           std::size_t bytes,
                           std::size_t count)
  : schema(std::move(columns)),
    id(binding),
    capacity(bytes),
    stop_(stop),
    deadline_(deadline),
    budget_(bytes, count),
    cancel_(stop, [this] { this->stop(); })
{
  validate_input_schema(schema);
  require(bytes <= input_window && count <= 128, "native input limits exceed hard bounds");
}
void native_input::activate(std::shared_ptr<input_pool> pool)
{
  std::lock_guard lock(mutex_);
  check_open();
  require(pool && !pool_, "native input activation must occur once");
  pool_ = std::move(pool);
}
void native_input::check_open() const
{
  if (error_.code) throw failure(error_.code, error_.message);
  if (stop_.stop_requested()) throw failure(SIRIUS_CANCELLED, "native query cancelled");
  if (clock::now() >= deadline_) throw failure(SIRIUS_TIMEOUT, "native query deadline expired");
  if (eos_) throw failure(SIRIUS_INVALID_STATE, "native input already finished");
}
std::shared_ptr<input_batch> native_input::acquire(std::size_t bytes, clock::time_point until)
{
  std::shared_ptr<input_pool> pool;
  {
    std::lock_guard lock(mutex_);
    check_open();
    if (!pool_) throw failure(SIRIUS_INVALID_STATE, "native input is not activated");
    pool = pool_;
  }
  require(bytes > 0 && bytes <= capacity, "native input lease size outside window");
  auto allocated = pool->rounded(bytes);
  // Charge retained descriptors as well as physical pool blocks, before either
  // allocation. Constant/NULL vectors cannot hide metadata outside the window.
  auto charged = allocated + schema.size() * sizeof(sirius_input_vector);
  buffer_budget::lease credit;
  auto status = budget_.acquire(charged, stop_, clock::now(), credit);
  if (status == SIRIUS_TIMEOUT && until > clock::now()) {
    ++blocked_;
    status = budget_.acquire(charged, stop_, std::min(until, deadline_), credit);
  }
  if (status != SIRIUS_OK) {
    std::lock_guard lock(mutex_);
    check_open();
    throw failure(status, "native input capacity unavailable");
  }
  auto batch           = std::make_shared<input_batch>();
  batch->credit        = std::move(credit);
  batch->pool          = pool;
  batch->storage       = pool->allocate(allocated);
  batch->payload_bytes = bytes;
  require(batch->storage && batch->storage->size() == allocated, "invalid native pool allocation");
  {
    std::lock_guard lock(mutex_);
    check_open();
    ++filling_;
    batch->owner = shared_from_this();
  }
  return batch;
}
void native_input::publish(std::shared_ptr<input_batch> const& batch,
                           uint32_t rows,
                           std::span<const sirius_input_vector> columns)
{
  require(batch && batch->owner.get() == this && !batch->published, "wrong native input lease");
  require(rows <= INT32_MAX && columns.size() == schema.size(), "native input schema/row mismatch");
  // Caller has exclusive ownership of an unpublished batch. Validation and
  // potentially large scans never hold the queue or coordinator mutex.
  batch->columns.assign(columns.begin(), columns.end());
  batch->rows = rows;
  for (std::size_t k = 0; k < columns.size(); ++k) {
    auto const& c = columns[k];
    require(c.reserved == 0 && c.vector_class <= SIRIUS_VECTOR_NULL, "invalid native vector class");
    range(c.data_offset, c.data_bytes, batch->payload_bytes);
    range(c.area_offset, c.area_bytes, batch->payload_bytes);
    range(c.null_offset, c.null_bytes, batch->payload_bytes);
    auto physical_rows = c.vector_class == SIRIUS_VECTOR_FLAT ? rows : (rows ? 1u : 0u);
    require(
      c.vector_class == SIRIUS_VECTOR_NULL ||
        c.data_bytes >= static_cast<std::size_t>(physical_rows) * input_element_size(schema[k].oid),
      "truncated native column values");
    require(c.null_bytes % 8 == 0 &&
              c.null_bytes <= ((static_cast<uint64_t>(physical_rows) + 63) / 64) * 8,
            "invalid native null bitmap length");
    for (uint32_t r = 0; r < physical_rows; ++r) {
      require(schema[k].nullable || !batch->is_null(k, r), "NULL in nonnullable native column");
      if (input_string_type(schema[k].oid)) {
        auto [offset, size] = batch->string_range(k, r);
        if (size > input_window - 8)
          throw failure(SIRIUS_RESOURCE_EXHAUSTED, "native row exceeds expanded limit");
      }
    }
  }
  {
    std::lock_guard lock(mutex_);
    check_open();
    if (rows) queue_.push_back(batch);  // allocation failure preserves filling ownership
    had_rows_        = had_rows_ || rows != 0;
    batch->published = true;
    --filling_;
  }
  notify();
}
void native_input::finish()
{
  {
    std::lock_guard lock(mutex_);
    if (eos_ && !error_.code) return;
    check_open();
    if (filling_) throw failure(SIRIUS_BUSY, "publish or release filling batches before EOS");
    eos_ = true;
  }
  budget_.close();
  notify();
}
void native_input::stop(sirius_status code, const char* message)
{
  {
    std::lock_guard lock(mutex_);
    if (!error_.code) assign_error(error_, code, message);
  }
  budget_.close();
  notify();
}
void native_input::discard()
{
  std::deque<std::shared_ptr<input_batch>> dropped;
  std::shared_ptr<input_pool> pool;
  {
    std::lock_guard lock(mutex_);
    dropped.swap(queue_);
    pool = std::move(pool_);
  }
  dropped.clear();
}
void native_input::subscribe(std::shared_ptr<capacity_waker> wake)
{
  {
    std::lock_guard lock(mutex_);
    waker_ = wake;
  }
  expanded_.set_waker(std::move(wake));
  notify();  // covers publication before subscription
}
void native_input::notify() const
{
  std::shared_ptr<capacity_waker> wake;
  {
    std::lock_guard lock(mutex_);
    wake = waker_;
  }
  if (wake) wake->wake();
}
bool native_input::ready() const
{
  std::lock_guard lock(mutex_);
  return error_.code || (eos_ && queue_.empty()) ||
         (!queue_.empty() && expanded_.inspect().bytes < input_window);
}
bool native_input::exhausted() const
{
  std::lock_guard lock(mutex_);
  return !error_.code && eos_ && queue_.empty() && (had_rows_ || empty_claimed_);
}
bool native_input::finished() const
{
  std::lock_guard lock(mutex_);
  return eos_;
}
sirius_error native_input::outcome() const
{
  std::lock_guard lock(mutex_);
  return error_;
}
void native_input::release_filling()
{
  std::lock_guard lock(mutex_);
  --filling_;
}
sirius_input_stats native_input::inspect() const
{
  std::lock_guard lock(mutex_);
  auto b = budget_.inspect();
  return {sizeof(sirius_input_stats),
          SIRIUS_ABI_VERSION,
          b.bytes,
          b.peak,
          b.leases,
          queue_.size(),
          filling_,
          blocked_.load(),
          expanded_.inspect().leases};
}
std::unique_ptr<input_unit> native_input::claim()
{
  std::unique_lock serial(claim_mutex_, std::try_to_lock);
  if (!serial.owns_lock()) return nullptr;
  std::vector<std::shared_ptr<input_batch>> pending;
  uint32_t front = 0;
  {
    std::lock_guard lock(mutex_);
    if (error_.code) {
      if (error_.code == SIRIUS_NOT_NEEDED) return nullptr;
      throw failure(error_.code, error_.message);
    }
    if (queue_.empty()) {
      if (!eos_ || had_rows_ || empty_claimed_) return nullptr;
      auto empty = std::make_unique<input_unit>();
      empty->chars.resize(schema.size());
      empty->nulls.resize(schema.size());
      empty_claimed_ = true;
      return empty;
    }
    pending.assign(queue_.begin(), queue_.end());
    front = front_row_;
  }
  auto unit = std::make_unique<input_unit>();
  unit->chars.resize(schema.size());
  unit->nulls.resize(schema.size());
  unit->slices.reserve(128);
  auto available = input_window - expanded_.inspect().bytes;
  auto limit     = std::min(input_target, available);
  // Bound per-column validity padding and string offset sentinels as well as
  // row-dependent data. GPU reservations additionally include upload/scratch.
  unit->bytes = schema.size() * 68;
  if (available <= unit->bytes) return nullptr;
  // First plan without mutating the queue; allocation/credit failures cannot
  // lose rows. Bounded descriptor fan-in, no eager list of all source units.
  for (auto const& batch : pending) {
    uint32_t begin = unit->slices.empty() ? front : 0;
    uint32_t end   = begin;
    for (; end < batch->rows; ++end) {
      if ((end & 4095u) == 0) {
        auto terminal = outcome();
        if (terminal.code == SIRIUS_NOT_NEEDED) return nullptr;
        if (terminal.code) throw failure(terminal.code, terminal.message);
        if (stop_.stop_requested()) throw failure(SIRIUS_CANCELLED, "native query cancelled");
        if (clock::now() >= deadline_)
          throw failure(SIRIUS_TIMEOUT, "native query deadline expired");
      }
      std::size_t cost = schema.empty() ? 1 : 0;
      for (std::size_t k = 0; k < schema.size(); ++k)
        cost += input_string_type(schema[k].oid) ? 5 + batch->string_range(k, end).second
                                                 : 1 + input_element_size(schema[k].oid);
      if (cost > input_window - schema.size() * 68)
        throw failure(SIRIUS_RESOURCE_EXHAUSTED, "native row exceeds expanded limit");
      if (cost > available - unit->bytes ||
          (unit->rows && cost > limit - std::min(limit, unit->bytes)))
        break;
      for (std::size_t k = 0; k < schema.size(); ++k) {
        if (input_string_type(schema[k].oid)) unit->chars[k] += batch->string_range(k, end).second;
        unit->nulls[k] += batch->is_null(k, end);
      }
      unit->bytes += cost;
      ++unit->rows;
      if (unit->rows == INT32_MAX) {
        ++end;
        break;
      }
    }
    if (end > begin) unit->slices.push_back({batch, begin, end - begin});
    if (end < batch->rows || unit->slices.size() == 128 || unit->bytes >= limit ||
        unit->rows == INT32_MAX)
      break;
  }
  if (unit->slices.empty()) return nullptr;
  if (expanded_.acquire(unit->bytes, {}, clock::now(), unit->expanded_credit) != SIRIUS_OK)
    return nullptr;
  {
    std::lock_guard lock(mutex_);
    if (error_.code == SIRIUS_NOT_NEEDED) return nullptr;
    if (error_.code) throw failure(error_.code, error_.message);
    for (auto const& slice : unit->slices) {
      front_row_ = slice.begin + slice.rows;
      if (front_row_ == queue_.front()->rows) {
        queue_.pop_front();
        front_row_ = 0;
      }
    }
  }
  return unit;
}
void input_registry::stop()
{
  for (auto const& read : reads)
    read->stop();
}
void input_registry::discard()
{
  for (auto const& read : reads)
    read->discard();
}
sirius_error input_registry::outcome() const
{
  for (auto const& read : reads) {
    auto error = read->outcome();
    if (error.code && error.code != SIRIUS_NOT_NEEDED && error.code != SIRIUS_CANCELLED)
      return error;
  }
  return {};
}
}  // namespace sirius::embedding
