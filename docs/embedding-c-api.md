# Native embedding C API

Tracking: [MatrixOne #28966](https://github.com/matrixorigin/matrixone/issues/28966).
Stages 3 through 6 provide control, bounded MO/TAE sources, admitted Substrait
bindings and incremental native results. A prepared native query owns a direct
terminal sink and can execute through `start`; it does not use the materialized
collector, Arrow result wrapper or a result repository. The capability mask
advertises ENGINE_CONTROL, MO_INPUT, TAE_INPUT and NATIVE_RESULTS.
Consumers must check the MO/TAE input and NATIVE_RESULTS bits before
enabling this route. This change alone does not enable the MO migration or
replace Flight. Numeric compatibility is separately tracked
by [#28968](https://github.com/matrixorigin/matrixone/issues/28968).

The native runtime requires exactly one configured GPU; multi-GPU configurations
are rejected before runtime initialization. GPU pipeline streams remain
independent: the default is two, and one and four are also validated.

## Build and link

Use Pixi and the existing native build. For example:

```sh
pixi run cmake --build build/upstream-dev-merge \
  --target sirius_embed sirius_c_smoke sirius_native_control_unittest -j 8
```

`Sirius::embed` links the bridge archive, static Sirius/DuckDB, the explicit
dummy extension loader, CUDA device-link objects and their native dependencies.
The generated SDK records both the verified C compiler (`c_compiler`) and C++
link driver (`compiler`). CGo consumers must use both, so compiled libc calls
and final linking use the same SDK headers and sysroot.
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

The query contract's `query_id` is a length-delimited opaque identity, not C
text. `sirius_query_bind` copies exactly `query_id_bytes` (1 through 4096),
including leading, interior, or trailing NUL bytes, so the caller may release or
reuse its buffer after the call returns. Names, paths, and other actual text
fields remain length-delimited text and reject embedded NUL bytes.

Engine creation requires an explicit configuration path. The native worker
override defaults to two; values 1 through 128 are accepted. One native GPU
runtime owns the process at a time. A second native runtime cannot replace its
global GPU allocators. The embedding context disables automatic DuckDB extension
loading to avoid constructing a second Sirius runtime; required extensions are
loaded explicitly.

The YAML `sirius.embedding.metadata_capacity_bytes` setting defaults to 256 MiB
and charges copied plans, contracts, read schemas, manifests, and a conservative
allowance for parsed TAE metadata before retaining them. The adjacent
`tae_host_staging_bytes` setting defaults to 64 MiB for the bounded TAE payload pump.

Embedded TAE mode and its staging budget are owned by a Sirius binding carrier.
The scanner parser returns storage metadata only. Binding copies retain the
query context, and physical planning transfers it to the TAE ingestible.
Optional `sort_column` metadata never selects execution mode: embedded reads
use block-sized work with or without a sort key, while standalone `tae_scan`
keeps its object-sized work.

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
and asynchronous GPU conversion/retry. Native results retain the same kind of
lease through filling, queued delivery and borrowed C handles. Scheduler-facing
acquisition is nonblocking; a capacity miss does not occupy a GPU worker.

## Bound native plans and inputs (stages 4-5)

The C input and binding operations are additive to ABI version 1; no Flight
protocol change is required. Stage 5 accepts only Substrait 0.78 plans whose reads
are named `__sirius_embedded_v1/<canonical binding id>`, match copied contracts,
and use the registered read set exactly. Arbitrary catalog/file/extension reads
are rejected before DuckDB lowering. MO and in-memory TAE bindings are installed
as typed private relations; no SQL text is assembled. Preparation retains the
optimized GPU physical plan and admits source and result progress reservations
before allowing producer allocation. The coordinator owns the execution window
through task and publication quiescence.

### Producer contract

1. Register at most 16 query-local binding IDs while the query is CREATED.
   Registration copies at most 1024 column descriptors per input. Registration
   freezes when preparation is queued; duplicate IDs are rejected.
2. After successful startable preparation, acquire a C-owned batch lease.
   Producers may fill and publish before `start`, and continue while execution
   runs. Acquisition observes its caller timeout and query cancellation/deadline.
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

## Incremental native results (stage 6)

The result operations are additive to ABI version 1. They reuse the opaque
`sirius_batch_handle`; input and result operations reject the wrong handle kind,
and `sirius_batch_release` releases either kind.

| Operation | Contract |
| --- | --- |
| `sirius_query_get_schema` | Returns the copied MO output contract after preparation. Column/name pointers remain valid until successful query close. |
| `sirius_query_next_result` | After start, waits up to `wait_ms` for one borrowed batch. The output must initially be NULL. TIMEOUT does not cancel execution; EOF is a separate status. A simultaneous pending pull returns BUSY. |
| `sirius_result_describe` | Returns rows, column descriptors and logical payload size. Descriptor pointers remain valid until batch release. A zero-row batch is data, not EOF. |
| `sirius_result_read` | Copies a checked byte range synchronously into caller-owned memory. Physical pinned blocks may be segmented; no contiguous native payload pointer is exposed or retained. |
| `sirius_query_get_result_stats` | Reports retained/peak charged bytes, leases, filling/queued/borrowed batches, current parked publications and cumulative capacity misses. |

Result vectors are flat MO layouts. Scalars are little-endian, decimals retain
their integer representation and declared scale, and DATE/DATETIME use MO's
epoch and units. Raw null words use one for NULL. Values and varlena descriptors
for NULL rows are zeroed. Strings currently use the external-area form for every
non-NULL value, including short strings: the 24-byte descriptor starts with
`UINT32_MAX`, followed by 32-bit area-relative offset and length. The remaining
descriptor bytes are zero. Each column's `area_offset` locates its area in the
batch payload. Physical cuDF types must match the admitted MO contract; the
encoder does not reconstruct decimal values through floating point.

Each query has a 64 MiB result window and at most 128 leases. Charges include
allocator-rounded pinned storage, retained batch objects and column descriptors.
The bounded host conversion scratch is charged to the same window: its logical
size is 64 KiB, which may occupy a larger pool block. The 32 MiB batch target is
reduced using the actual pool rounding and descriptor charge, preferably fitting
two independently owned batches alongside scratch. Configurations whose blocks
cannot fit even one result plus scratch fail preparation. Row slicing accounts
for null words, varlena descriptors and string
areas before allocating. A single row that cannot coexist with scratch inside
the hard window returns RESOURCE_EXHAUSTED; there is no oversize grant.

Dequeuing retains the full charge. Only final batch release returns capacity;
query cancellation never invalidates borrowed bytes. Query close returns BUSY
while a borrowed result remains, including after execution quiesces. Pool
reservations remain owned until their query and outstanding owners release
them. Errors take precedence over queued data; queued batches are discarded on
failure, while already borrowed batches remain readable.

Terminal tasks obtain one of `max(2, 2 * gpu_streams)` admission tickets before
claiming input. Pre-publication compute retries retain the ticket. A finite
CUDA stream handoff transfers read-locked GPU output owners and the ticket to a
query-owned publication driver. That driver holds a bounded set of cursors,
copies GPU values directly into final host storage without a device conversion
mirror, and parks on returned host capacity independently of task-creator/GPU
workers. Its row cursor advances only after publication commits; capacity waits
do not replay operators or already published rows.

Tickets are ordered by terminal input claim, not GPU completion. Released
no-input claims do not leave sequence gaps that block publication. This retains
the order presented by the final source; SQL ORDER BY additionally depends on
that source providing global sorted order. The production integration test
checks a descending input sorted into an ascending result larger than the
window, across multiple batches. No stable order is promised without ORDER BY,
and this single tested sort shape is not a claim about every plan shape.

EOF is published only after engine tasks, publication work and query cleanup
quiesce successfully. `sirius_query_wait` waits for execution/quiescence, not for
one result; calling it instead of draining a result larger than the window can
wait until timeout. Cancellation and deadlines wake capacity waits, close
scheduling subscriptions and drain owners before cleanup. Failed CUDA
synchronization retains unsafe owners and reports GPU_UNAVAILABLE even if an
earlier terminal signal succeeded. Automatic replay after any returned result
batch is unsafe because it can duplicate rows already observed by the caller.

## Validation

```sh
pixi run build/upstream-dev-merge/extension/sirius/sirius_native_control_unittest \
  '[native_control],[native_credit]'
pixi run build/upstream-dev-merge/extension/sirius/sirius_native_control_unittest \
  '[native_input],[native_result]'
pixi run build/upstream-dev-merge/extension/sirius/sirius_native_gpu_unittest \
  '[native_gpu],[native_result_gpu]'
pixi run build/upstream-dev-merge/extension/sirius/sirius_c_smoke \
  test/cpp/scan/memory.yaml
pixi run build/upstream-dev-merge/extension/sirius/sirius_native_result_integration \
  test/cpp/scan/memory.yaml 1
pixi run build/upstream-dev-merge/extension/sirius/sirius_native_result_integration \
  test/cpp/scan/memory.yaml 2
pixi run build/upstream-dev-merge/extension/sirius/test/cpp/sirius_unittest \
  '[gpu_pipeline_task],[completion_handler],[sirius_ffi],[tae_scan]'
```

The control tests use deterministic private drivers, not a fake public SQL
implementation. They cover thread affinity, cancellation, capacity, failure,
deadline-versus-wait-timeout distinction, fatal owner retention and credit
lifetime. Fatal retention runs in a test-owned child process because process
death is deliberately its final cleanup owner. The C smoke test separately
exercises real native linkage, one-row production input/execution/result
delivery, wrong-kind handle rejection, borrowed-result close protection,
rejection of a second simultaneous runtime, and GPU runtime shutdown/recreation
with the default two workers followed by one worker.

The separate production result executable links `Sirius::embed` without
replacing its backend. Its 96 MiB cases check exact unordered row equivalence,
stalled-consumer bounds, cancellation/deadline while full, borrowed storage
after quiescence and multi-batch ORDER BY. GPU codec tests independently check
sliced null masks, MO epochs, varlena, decimal integer bits and unsigned high
bits. These commands define the acceptance checks; their presence in the tree
does not substitute for a recorded passing run.

### Foundation lifecycle review

| Layer | Ownership, termination or bound |
| --- | --- |
| C handles and coordinator | Successful close releases handles exactly once; failed close preserves them. Query drivers and the backend are destroyed on the coordinator after cleanup; unsafe GPU owners are retained for process restart. |
| Control waits | Call waits have explicit timeouts; queued cancellation bypasses active work. Drivers must honor the supplied stop token and deadline. A normal driver return after the deadline is reported as TIMEOUT, not success. |
| Retained state | At most 17 live query handles, each with at most a 16 MiB copied plan. Buffer leases retain both byte and count credit until final release. |

The deadline regression is tested with a driver that returns normally when its
deadline expires, followed by successful cleanup. These checks cover the
foundation's ownership and scheduling contracts independently of the GPU tests.

Native-route readiness additionally requires admitted TAE execution and source
boundedness evidence. MO reader integration and runtime artifact packaging are
separate migration work. No SF10 or all-22 claim follows from these tests.
