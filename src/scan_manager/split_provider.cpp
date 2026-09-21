/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "scan_manager/split_provider.hpp"

#include "exec/completion_controller.hpp"
#include "exec/try.hpp"
#include "op/scan/gpu_ingestible.hpp"

#include <cassert>
#include <exception>
#include <utility>

namespace sirius::scan_manager {

split_provider::split_provider(op::scan::gpu_ingestible& ingestible, io::ioctx_resolver resolve)
  : _ingestible(&ingestible), _resolve(std::move(resolve))
{
}

bool split_provider::has_more_splits() const { return !_ingestible->has_processed_all_metadata(); }

std::function<std::unique_ptr<op::scan::scan_info>()> split_provider::next_split_provider()
{
  return _ingestible->next_split_provider(_resolve);
}

void split_provider::run(const push_callback_t& on_split)
{
  using split_type = typename value_type::value_type;
  assert(on_split);
  if (_producer.joinable()) throw std::runtime_error("split provider can only be started once");
  auto controller = std::make_shared<exec::completion_controller>();
  _completion_token =
    controller->on_completion([on_split] { on_split(exec::make_empty_try<split_type>()); });
  auto producer_slot = controller->acquire();
  try {
    _producer = std::jthread(
      [this, on_split, controller, slot = std::move(producer_slot)](std::stop_token stop) mutable {
        std::stop_callback stop_metadata(stop, [this] { _ingestible->stop_metadata_scan(); });
        (void)slot;
        try {
          while (!stop.stop_requested() && has_more_splits()) {
            auto work = next_split_provider();
            if (!work) break;
            on_split(work());
          }
        } catch (...) {
          try {
            on_split(std::current_exception());
          } catch (...) {
          }
        }
        controller->close();
      });
  } catch (...) {
    try {
      on_split(std::current_exception());
    } catch (...) {
    }
    controller->close();
  }
}

}  // namespace sirius::scan_manager
