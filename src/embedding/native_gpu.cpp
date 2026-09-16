/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/native_gpu.hpp"

#include "op/scan/sirius_gpu_scan_operator_data.hpp"

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <cstring>
#include <numeric>

namespace sirius::embedding {
namespace {
using host_resource = cucascade::memory::fixed_size_host_memory_resource;
struct pool_storage final : input_storage {
  host_resource::fixed_multiple_blocks_allocation blocks;
  explicit pool_storage(host_resource::fixed_multiple_blocks_allocation p) : blocks(std::move(p))
  {
    // Never expose bytes from a previous query when a caller leaves a field unwritten.
    for (auto* block : blocks->get_blocks())
      std::memset(block, 0, blocks->block_size());
  }
  std::size_t size() const override { return blocks->size_bytes(); }
  void read(std::size_t offset, std::span<std::byte> bytes) const override
  {
    if (offset > size() || bytes.size() > size() - offset)
      throw failure(SIRIUS_INVALID_ARGUMENT, "native pool read outside lease");
    while (!bytes.empty()) {
      auto block  = (*blocks)[offset / blocks->block_size()];
      auto within = offset % blocks->block_size();
      auto count  = std::min(bytes.size(), block.size() - within);
      std::memcpy(bytes.data(), block.data() + within, count);
      offset += count;
      bytes = bytes.subspan(count);
    }
  }
  void visit(std::function<void(std::size_t, std::span<std::byte>)> const& fn) override
  {
    std::size_t offset = 0;
    for (auto* block : blocks->get_blocks()) {
      fn(offset, {block, blocks->block_size()});
      offset += blocks->block_size();
    }
  }
};
struct reserved_pool final : input_pool {
  std::unique_ptr<cucascade::memory::reservation> reservation;
  host_resource* resource;
  reserved_pool(cucascade::memory::memory_space& host, std::size_t bytes)
    : reservation(host.make_reservation_or_null(bytes)),
      resource(host.get_memory_resource_of<cucascade::memory::Tier::HOST>())
  {
    if (!reservation)
      throw failure(SIRIUS_RESOURCE_EXHAUSTED,
                    "native input progress window does not fit host capacity");
    if (!resource) throw failure(SIRIUS_INVALID_STATE, "native input needs a pinned host pool");
  }
  std::size_t rounded(std::size_t bytes) const override
  {
    auto block = resource->get_block_size();
    return (bytes + block - 1) / block * block;
  }
  std::unique_ptr<input_storage> allocate(std::size_t bytes) override
  {
    return std::make_unique<pool_storage>(
      resource->allocate_multiple_blocks(bytes, reservation.get()));
  }
};
struct native_info final : op::scan::ingestible_table_info {
  std::span<const std::string> column_names() const override { return {}; }
  std::span<const std::string> file_paths() const override { return {}; }
  std::string display_name() const override { return "native_mo_input"; }
};
struct native_split final : op::scan::scan_info {
  std::unique_ptr<input_unit> unit;
  explicit native_split(std::unique_ptr<input_unit> data) : unit(std::move(data)) {}
  std::size_t estimated_bytes() const noexcept override
  {
    return std::max<std::size_t>(1, unit->bytes);
  }
  std::size_t estimated_working_set_bytes() const noexcept override
  {
    std::size_t bytes = estimated_bytes();
    for (auto const& slice : unit->slices)
      bytes += slice.batch->payload_bytes;
    return bytes;
  }
};
class native_ingestible final : public op::scan::gpu_ingestible {
 public:
  explicit native_ingestible(std::shared_ptr<native_input> p) : input_(std::move(p)) {}
  bool is_live() const noexcept override { return true; }
  bool live_ready() const override { return input_->ready(); }
  bool live_exhausted() const override
  {
    return input_->exhausted() || input_->outcome().code == SIRIUS_NOT_NEEDED;
  }
  void live_subscribe(std::shared_ptr<capacity_waker> wake) override
  {
    input_->subscribe(std::move(wake));
  }
  void live_stop() override { input_->stop(SIRIUS_NOT_NEEDED, "native source no longer needed"); }
  std::unique_ptr<op::operator_data> live_claim() override
  {
    auto unit = input_->claim();
    if (!unit) return nullptr;
    return std::make_unique<op::scan::scan_operator_input>(
      std::make_unique<native_split>(std::move(unit)));
  }
  std::unique_ptr<op::scan::batch_coalescer> create_batch_coalescer() const override
  {
    throw failure(SIRIUS_INVALID_STATE, "live inputs cannot enumerate file metadata");
  }
  bool has_processed_all_metadata() const override { return live_exhausted(); }
  metadata_scan_task_t next_split_provider(io::ioctx_resolver) override
  {
    throw failure(SIRIUS_INVALID_STATE, "live inputs cannot enumerate file metadata");
  }
  op::scan::filtered_table materialize_metadata_to_table(
    op::scan::scan_info const& info,
    cucascade::memory::memory_space const& space,
    rmm::cuda_stream_view stream,
    bool,
    std::shared_ptr<const like_multiliteral_cache>) override
  {
    auto const& split = dynamic_cast<native_split const&>(info);
    return {
      op::scan::owning_table_view(convert_native_input(*split.unit, input_->schema, space, stream)),
      op::scan::filter_state::ROW_FILTERED_AND_PROJECTED};
  }
  std::unique_ptr<cudf::table> post_filter_and_project(
    op::scan::filtered_table&& table,
    cucascade::memory::memory_space const&,
    rmm::cuda_stream_view stream,
    bool,
    std::shared_ptr<const like_multiliteral_cache>,
    std::unique_ptr<cudf::column>*,
    std::span<const std::size_t>) override
  {
    return table.table.release(stream);
  }
  op::scan::ingestible_table_info const& table_info() const noexcept override { return info_; }
  std::vector<std::size_t> materialized_column_order() const override
  {
    std::vector<std::size_t> order(input_->schema.size());
    std::iota(order.begin(), order.end(), 0);
    return order;
  }

 private:
  std::shared_ptr<native_input> input_;
  native_info info_;
};
}  // namespace
std::shared_ptr<input_pool> make_native_input_pool(cucascade::memory::memory_space& host,
                                                   std::size_t bytes)
{
  return std::make_shared<reserved_pool>(host, bytes);
}
void activate_native_inputs(input_registry& inputs, cucascade::memory::memory_space& host)
{
  std::vector<std::shared_ptr<input_pool>> pools;
  for (auto const& read : inputs.reads)
    pools.push_back(make_native_input_pool(host, read->capacity));
  for (std::size_t i = 0; i < pools.size(); ++i)
    inputs.reads[i]->activate(std::move(pools[i]));
}
std::shared_ptr<op::scan::gpu_ingestible> make_native_ingestible(
  std::shared_ptr<native_input> input)
{
  return std::make_shared<native_ingestible>(std::move(input));
}
}  // namespace sirius::embedding
