# Native embedding C API

Tracking: [MatrixOne #28966](https://github.com/matrixorigin/matrixone/issues/28966).
Stages 3 and 4 provide the control/library foundation and bounded MO input.
Stages 5-6 add admitted Substrait/TAE bindings and native result adapters. The public
capability mask currently advertises **ENGINE_CONTROL only**. Query preparation
returns `SIRIUS_UNSUPPORTED`; this is not a working native SQL data path and must
not be substituted for Flight yet. Numeric compatibility is separately tracked
by [#28968](https://github.com/matrixorigin/matrixone/issues/28968).

## Build and link

Use Pixi and the existing native build. For example:

```sh
pixi run cmake --build build/upstream-dev-merge \
  --target sirius_embed sirius_c_smoke sirius_native_control_unittest -j 8
```

`Sirius::embed` links the bridge archive, static Sirius/DuckDB, the explicit
dummy extension loader, CUDA device-link objects and their native dependencies.
Do not link only `libsirius_embed.a` or a loadable DuckDB extension.

Building `sirius_c_smoke` also generates `extension/sirius/embedding-sdk`:

- `sirius_c.h`: C99-compatible ABI version 1, with no C++/DuckDB/cuDF includes;
- `link.json`: compiler and complete verified link arguments;
- `link.rsp`: compiler response file containing that link closure;
- `SiriusEmbedConfig.cmake`: an imported `Sirius::embed` target for CMake consumers.

The exporter obtains dependencies from the CMake/Ninja C-consumer link command,
not a manually duplicated library list. These are **build-tree artifacts** with
absolute paths tied to this Pixi/native generation. Relocatable MO runtime
packaging belongs to migration stage 7. Rebuilding the C consumer regenerates the
SDK after native, public-header or exporter changes;
do not move the response file or reuse it with an unrelated toolchain.

An external CMake consumer uses `find_package(SiriusEmbed CONFIG REQUIRED)`,
sets its executable's linker language to CXX, and links `Sirius::embed`.
Point `CMAKE_PREFIX_PATH` at the generated SDK directory.

## Ownership and control

Initialize option/output structures with their actual `struct_size` and ABI
version. Create output handles must initially be NULL; a rejected create never
overwrites a live handle. Error storage is caller-owned and overwritten per
call; there is no shared last-error string.

Engine creation requires an explicit configuration path. The native worker
override defaults to two; values 1 through 128 are accepted. One native GPU
runtime owns the process at a time. A second native runtime cannot replace its
global GPU allocators. The embedding context disables automatic DuckDB extension
loading to avoid constructing a second Sirius runtime; required extensions are
loaded explicitly.

The coordinator initializes and destroys the runtime, and performs preparation,
execution and query teardown, on one native thread. Foreign callers never own
DuckDB's thread-affine execution-window mutex. Stop-token cancellation bypasses
the command queue; a queued query can be canceled while another query runs.

Admission permits one active query plus at most 16 waiters. The total live
handle count is bounded as well, including unprepared/unreleased handles. Each
plan is bounded to 16 MiB. Query deadlines cover preparation, waiting for start
and execution; the default is 15 minutes.

`prepare`/`wait` timing out does not implicitly cancel a query. `start` is
single-use. Retrieve the query outcome with `wait`; `close` reports whether
resource release succeeded. A quiesced query whose execution timed out is still
releasable. Engine close seals admission and cancels queries but returns BUSY
until callers close their query handles. Nonzero close results preserve the
handle for inspection/retry. Destroying a handle requires exclusive caller
ownership: concurrent calls using that handle must have returned first.

Fatal synchronization and unprovable cleanup are distinct from quiescence.
They seal admission, retain unsafe owners and return GPU_UNAVAILABLE. They
cannot be repaired by freeing buffers, resetting a shared device or reopening
the runtime in-process; the host must arrange fail-stop/restart.

Fatal GPU status is retained independently of the root's one-shot completion
promise. The query owner checks it after draining; an earlier completion/error
notification cannot make a poisoned runtime reusable. GPU workers notify this
owner rather than joining executor pools themselves.

## Buffer-credit primitive

`buffer_budget` enforces both byte and lease-count limits. It never grants an
oversized request as a deadlock escape. Credit remains charged after moving or
dequeueing a lease and returns only on final release. Leases keep their accounting
state alive after the facade closes. Closing wakes blocked acquisition and
removes scheduling callbacks; callbacks run outside the accounting lock.

Native input carries these leases through filling, publication, source claims
and asynchronous GPU conversion/retry. The native result sink is still pending.

## Bounded native MO input (stage 4)

The C input operations are additive to ABI version 1; no Flight protocol changes
or new runtime configuration are required. Production preparation deliberately
remains unsupported until stage 5 implements strict plan/binding admission.
The private integration-test backend exercises the production C entry points,
pooled source and real GPU scheduler; it is not linked into `Sirius::embed`.

### Producer contract

1. Register at most 16 query-local binding IDs while the query is CREATED.
   Registration copies at most 1024 column descriptors per input. Registration
   freezes when preparation is queued; duplicate IDs are rejected.
2. After successful preparation, acquire a C-owned batch lease. Queued or
   preparing queries cannot allocate input buffers. Acquire observes both the
   caller wait timeout and query cancellation/deadline.
3. Write MO value bytes, varlena areas and raw null words with synchronous bulk
   copies. The implementation does not retain the source pointer. Neither
   Flight frames nor serialized TAE vector headers are used.
4. Publish row count and column offsets/lengths. Success consumes the batch
   handle; failure leaves it with the caller for correction or release.
   Published bytes are immutable. Zero rows do not mean EOS.
5. Finish explicitly after publishing/releasing filling batches. Finish is
   idempotent; outstanding filling leases return BUSY. Producer failure cancels
   the query and preserves the error instead of reporting EOF. Closing an
   unfinished producer cancels the query, rather than inventing successful EOS.

Query close cancels and joins execution, then returns BUSY while input handles
or caller-owned batch leases remain. Closing/releasing those handles permits
retrying query close. Cancellation never frees a buffer while its producer can
still write it. Early source abandonment returns NOT_NEEDED. Other failures
preserve their error category across blocked acquisition and query wait.

### Bounds and memory ownership

Each read has a 64 MiB byte window and at most 128 batch leases. Charges include
physical pinned-pool block rounding and retained batch descriptors, so the
maximum payload is slightly smaller than 64 MiB. Bitmap/area bytes live inside
the charged payload. Metadata also has explicit read/column/descriptor bounds.

Activation reserves every read's full window from the configured host pool
before enabling producers; insufficient aggregate host capacity fails admission.
Allocations draw on those reservations, not additional unreserved pinned memory.
Unused reservations are released during quiescent cleanup; filling/retry owners
keep their pool reservation alive until their own final release.

The consumer coalesces available batches toward 32 MiB, with at most 128 batch
slices and 64 MiB expanded data per source unit. Expanded-source credits also
bound the total outstanding units of a read to 64 MiB / 128 units. Planning
includes validity padding and string offsets. Constant and variable-width
expansion is bounded; oversized units split lazily at row boundaries, and an
unrepresentable single row fails. Producer payloads must already be bounded.

Dequeuing does not release input credit. The scan split retains immutable
native bytes until conversion and any source retry are safe. Successful decode
synchronizes the task stream before releasing host ownership; later pipeline
retries retain decoded GPU data under the existing query memory manager.
Failed quiescence retains unsafe owners and returns GPU_UNAVAILABLE.

### Live scan scheduling and types

Native ingestion uses the unified GPU scan operator, but bypasses finite file
metadata enumeration and cross-query pinned caches. Empty-open input parks
scheduling without blocking a task-creator/GPU worker. A durable, deduplicated
wakeup covers publication, EOS, error and returned source credit. Subscriptions
use weak query generations and close before queued/in-flight plan references
are drained. File-backed scans retain their existing behavior.

Layouts use little-endian MO scalar values, 24-byte varlena and raw 64-bit null
words (1 means NULL; omitted trailing words mean non-NULL). Supported OIDs are
BOOL, signed/unsigned integers through 64 bits, FLOAT32/64, DATE, DATETIME,
DECIMAL64/128, CHAR/VARCHAR/TEXT/BLOB. Other OIDs, including Decimal256, return
UNSUPPORTED; no decimal-to-float narrowing is introduced. Flat, constant and
constant-NULL vectors are supported. NULL varlena garbage is never dereferenced.

The future MO adapter must acquire native capacity before allocation/copy and
keep its reader-to-publisher edge bounded. These Sirius tests prove the native
back-pressure boundary; actual MO reader restraint is a stage-8 acceptance test.

## Validation

```sh
pixi run build/upstream-dev-merge/extension/sirius/sirius_native_control_unittest \
  '[native_control],[native_credit]'
pixi run build/upstream-dev-merge/extension/sirius/sirius_native_control_unittest \
  '[native_input]'
pixi run build/upstream-dev-merge/extension/sirius/sirius_native_gpu_unittest \
  '[native_gpu]'
pixi run build/upstream-dev-merge/extension/sirius/sirius_c_smoke \
  test/cpp/scan/memory.yaml
pixi run build/upstream-dev-merge/extension/sirius/test/cpp/sirius_unittest \
  '[gpu_pipeline_task],[completion_handler],[sirius_ffi],[tae_scan]'
```

The control tests use deterministic private drivers, not a fake public SQL
implementation. They cover thread affinity, cancellation, capacity, failure,
deadline-versus-wait-timeout distinction, fatal owner retention and credit
lifetime. Fatal retention runs in a test-owned child process because process
death is deliberately its final cleanup owner. The C smoke test separately
proves real native linkage, the public unsupported-query outcome, rejection of
a second simultaneous runtime, and GPU runtime shutdown/recreation with the
default two workers followed by one worker.

### Foundation lifecycle review

| Layer | Ownership, termination or bound |
| --- | --- |
| C handles and coordinator | Successful close releases handles exactly once; failed close preserves them. Query drivers and the backend are destroyed on the coordinator after cleanup; unsafe GPU owners are retained for process restart. |
| Control waits | Call waits have explicit timeouts; queued cancellation bypasses active work. Drivers must honor the supplied stop token and deadline. A normal driver return after the deadline is reported as TIMEOUT, not success. |
| Retained state | At most 17 live query handles, each with at most a 16 MiB copied plan. Buffer leases retain both byte and count credit until final release. |

The deadline regression is tested with a driver that returns normally when its
deadline expires, followed by successful cleanup. These checks cover the
foundation's ownership and scheduling contracts, not the future GPU data path.

The four-stage native round is not complete until a C caller can feed actual
MO inputs, run an admitted TAE query, drain bounded results, and cancel the full
data path. No SF10 or all-22 claim follows from these foundation tests.
