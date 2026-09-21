# Embedded TAE demand scheduling

Embedded TAE scans use the live-source interfaces. Standalone file scans retain
their existing provider path. This change does not enable the public native
execution capability by itself.

## Demand and ownership

One controller and one metadata worker serve all TAE scans in a query. A source
subscription registers notifications but does not start scanning. The task
creator requests work only after the source pipeline's dependencies finish;
later join branches therefore cannot consume all permits before their build
branches run. At most one metadata request is pending per source, with a hard
controller limit of 64 queued source notifications.

The shared work limit is `max(2, 2 * gpu_streams)`, with at most 128 configured
streams. A permit covers discovery, a queued descriptor, execution, retries and
asynchronous retirement. Dequeuing a descriptor does not return its permit.
The task's `execution_lease()` retains both the permit and its physical staging
allocation if the scan input is replaced. Retry inherits that lease. Failed
stream quiescence quarantines it with the task's other owners.

The controller parses and validates object metadata once per cache residency.
Its immutable LRU is bounded, and each work attempt opens its own datasource:
mutable datasource prefetch state is never cached or shared between tasks.
Adjacent selected blocks coalesce toward 32 MiB of decoded input. The metadata
descriptor limit can end a batch sooner. A single block is never split into
semantically independent row fragments.

## GPU and host admission

Descriptors contain extents and an immutable conservative GPU reservation floor,
not loaded payloads. The floor covers the compressed device mirror, decompressed
vectors, nvCOMP scratch, decode outputs, selection/filter copies and descriptor
scratch. Memory history cannot reduce that floor. Strict task admission obtains
a complete reservation before removing the task from the scheduler queue.
Requests larger than an eligible device's capacity fail before payload I/O.

Temporarily inadmissible tasks remain in the existing queue so other consumers
can run. Reclaim requests are asynchronous and limited to one per device.
Task/device events wake the scheduler immediately; a bounded 10 ms channel wait
also observes external memory releases because cuCascade does not expose a
reservation-release subscription. No GPU worker waits on that admission path.

Every work permit receives a staging entitlement of
`min(8 MiB, floor(host_capacity / work_limit))`, rounded down to a 2048-byte CRC
block. The configured host capacity may reduce the 64 MiB ceiling but cannot
increase it. Insufficient space for one CRC block per permit fails admission.
Entitlements are charged before discovery; physical pinned storage is allocated
only after full GPU admission. Their sum fits the dedicated TAE host budget, so
GPU workers never wait for staging capacity.

Uploads reuse that storage only after the preceding DMA completes. CRC framing
is removed in place, with no second raw or stripped payload buffer. Exact range
reads prevent a metadata read from fetching neighboring payload bytes in the
same physical CRC block. A compressed extent larger than 64 MiB can therefore
be transported in slices while its complete compressed input is assembled on
the device before decompression.

## Metadata accounting

The first TAE binding reserves **128 MiB from the existing engine metadata
budget**, whose default remains 256 MiB. Further TAE bindings in that query do
not reserve it again. Plan, manifest and schema charges remain additive in that
same budget. The query holds the lease until close, after controller and GPU
work have drained; failed registration rolls back both temporary charges.

The runtime envelope is partitioned as follows:

| Component | Bound and enforcement |
|---|---|
| Immutable object cache | 32 MiB, including parsed vector capacity, key storage and node/ownership overhead; eviction before insertion |
| Metadata parsing | 24 MiB: at most 8 MiB each for serialized bytes, decompressed bytes and predicted parsed allocation |
| All work metadata | 64 MiB shared across every permit, including converter host vectors and cuDF host descriptors |
| Controller and metadata-worker temporaries | 8 MiB allowance; 64 queued callbacks, paths limited to 4096 bytes, 64 KiB CRC metadata scratch, bounded control objects |

Before the general parser allocates, a preflight sums block and column allocation
requirements with a 2x ownership/rounding allowance and rejects more than 8 MiB.
This check counts repeated block references independently: aliased offsets cannot
amplify a small encoded metadata blob into unbounded parsed vectors. Neither the
serialized nor decompressed blob may exceed 8 MiB. Embedded scans also avoid
expanding the common data root into an array containing every full object path.

A work permit's metadata entitlement is
`min(8 MiB, 64 MiB / work_limit)`. It reserves 64 KiB for fixed objects and bounded
path copies. The remaining entitlement charges **4096 bytes per projected column
chunk**. That charge covers the scan and converter chunk arrays, grouping map and
index-vector capacities, nvCOMP host pointer/size/status arrays, fixed/string/null
decode descriptors, and cuDF host column/table/view owners. Compile-time ABI size
checks constrain the chunk, cuDF column/view and vector types used in this bound.
The combined assertion is
`4*sizeof(cudf::column) + 4*sizeof(cudf::column_view) + 1024 <= 4096`:
flat strings have one offsets child, and the four wrappers allow both input and
output owners/views. The 1024-byte component bounds the two chunk arrays,
grouping/index capacities and converter pointer/size/status/decode descriptors.
The decoder has no host array proportional to decoded rows or string bytes.

The source reserves a chunk vector no larger than 128 KiB before filling it and
uses the smaller of that limit and its per-work entitlement. With the current
64-byte chunk descriptor, two configured streams permit 2032 chunks per work
unit; 128 streams permit 48. Coalescing stops before exceeding the limit. A single
block whose projected column count exceeds it fails with `RESOURCE_EXHAUSTED`
before payload I/O instead of allocating an oversized descriptor.

Embedded varchar decoding validates references and sums lengths in 64 bits
before its int32 prefix sum. String amplification beyond the serialized-vector
reservation bound, cuDF offset overflow, or larger-than-budget CUB scratch fails
explicitly. nvCOMP status and actual decoded sizes are checked before decoding.
Coalesced null masks use exact row offsets, including neighboring blocks that
share a 32-bit validity word.

## Validation

`[tae_demand]` and `[tae_slices]` exercise global permit/credit ownership, retry and
close, the metadata lease charge, stream-scaled descriptor bounds, cache eviction,
CRC boundaries and greater-than-64-MiB transport. `[tae_source]` covers dormant
subscriptions, metadata-only discovery and object metadata reuse. `[tae_gpu]`
covers a real 65 MiB string conversion, admission-before-read, cancellation between
slices, retained pinned storage, and unaligned coalesced null masks. Existing
`[embedded_tae]` integration coverage continues to compare GPU and CPU results.
