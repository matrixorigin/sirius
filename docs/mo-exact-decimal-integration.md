# Embedded MO exact-decimal native completion

Native B follows merged Sirius #26 (`c440013c89f1a54a6e5d934aabd0f9d3ef15ed3c`)
and implements the owner-approved next increment for MO #28968 / #28966.
The semantic authority remains MO #29449 document blob
`42a89f09a1d168d02b9583cb3ea7b4de6dbb5634`; MO #29690 records the B-F
delivery sequence. The detailed implementation plan was approved before code.

## Contracts and owners

The validated import descriptor owns width, precision, scale and nullability.
Private aggregate bindings preserve that immutable contract through cloning
and reconstruction. Unsupported aggregate modifiers fail preparation.

Local and merge aggregation share one typed layout. SUM/AVG partial states
use the full selected signed 128/256-bit physical range, not a scalar decimal
precision limit. AVG retains a non-NULL count and rounds only at finalization.
Empty/all-NULL groups produce NULL. Every public result is finalized, including
a single partial batch. MIN/MAX preserve their operand type and ignore NULLs.
Ordinary and exact aggregate slots share group ordinals.

A single local partial goes directly to finalization. Multiple partials are
concatenated, regrouped and reduced before the same finalizer. Initial
coefficients can skip aggregate error polling when the accumulator is at least
32 bits wider than the full signed input domain: at most INT32_MAX input rows
fit every intermediate subset. MIN/MAX have the same infallible initial
reduction property. Same-width wide states and all merges retain checked error
polling. Final declared precision checks are never skipped. This proof does not
depend on sampled coefficient values.

Grouped tile bounds are computed directly from sorted row offsets, with at
most one identity tile added per segment. Prefix indices use 64 bits. This
avoids a CUB scan, duplicate CUDA template-kernel registrations and a host
tile-count transfer; nonzero-origin subranges and tile-boundary transitions
have independent value/count controls.

Partial states remain ordinary column-backed cuCascade batches. Existing task
reservations own coefficients, counts, grouping/permutation/reduction scratch,
and output buffers; existing partition/spill/restore owns retained batches.
There is no separate aggregate registry, cache or spill service.

Equality partitioning and hash probing use the same exact key encoding.
Mixed scales normalize trailing decimal zeros and zero itself; ordering uses
the signed coefficient at the column's fixed scale. Original columns retain
their declared schema. Canonical Decimal256 validity is restored only after
operations whose canonical inputs prove child masks redundant.

The embedded preparation path installs scoped import only for exact plans,
validates the complete reachable graph and schema, then permits readers to
start. Ordinary import and standalone execution retain their existing path.
Read filters are checked against the base schema before flat read projections
select their output schema. CASE/COALESCE producer descriptors and outer-join
nullability survive binding through private checked identity casts. Alias
inspection constructs a fresh carrier; DuckDB's shared type metadata must not
be mutated while inspecting an immutable function binding.

The exact embedded path keeps MO's optimized logical plan and skips DuckDB's
ordinary optimizer, whose coefficient-carrier folding cannot interpret MO
scales. It reconstructs typed equality join conditions for the native planner.
Schema propagation is charged against the existing query metadata admission;
original and normalized serialized plans remain bounded at 16 MiB.

Runtime numeric errors retain the failing operation's owner through the task
and session boundary, including producer cancellation during cleanup. Fatal
quiescence failures retain their existing precedence. ABI-v1 layouts stay
unchanged; statuses 12/13 are appended and capability mask 16u is enabled. The
complete current capability mask is 31u.

Unsupported exact-plan shapes fail preparation: raw numeric casts, ordinary
arithmetic/comparison over exact operands, exact IN/nested tuples, non-equality
decimal join keys, grouping sets, aggregate filters/options/order/intermediate
phases, best-effort read filters and zero-column intermediate schemas. Exact aggregates accept ALL; ordinary
grouped COUNT DISTINCT retains the existing set-state representation. Public
Decimal256 results remain limited to precision 65. These restrictions must be
respected by the MO lowering increment.

## Change and validation map

| Closure | Risk | Required proof |
| --- | --- | --- |
| Raw accumulators and aggregate finalization | R2 | Independent CPU vectors, physical versus declared bounds, cancellation, signed rounding, NULL and owner-specific errors |
| Aggregate import/binding/clone and preparation | R2 | Real importer and planner, malformed identities/modifiers, full graph rejection before reader work, unchanged ordinary IDs and C layouts |
| GPU local/merge aggregation and spill states | R3 | Actual values across widths, mixed slots, batch/merge permutations, single/empty batches, spill/restore and bounded cleanup |
| Keys, predicates and Decimal256 validity | R3 | Cross-width/scale partition/hash equality, NULL joins, signed ordering and following exact operations after gather |
| Runtime exposure and errors | R3 | Real C ABI queries, statuses 12/13 by operation, cancellation/async failure, quiescence and healthy reuse/fatal quarantine |
| SDK/build/docs | R0/R1 | Frozen native build, C consumer/schema fingerprint, pinned hooks and exact artifact provenance |

New GPU owners are constructed before submission, retain operands and scratch,
and prove task-stream quiescence before destruction. Unprovable owners transfer
to the existing fatal quarantine. Allocation, cancellation and partial failure
must leave one cleanup owner. All scratch and partial-state growth is charged
before allocation and released or spilled through existing resource ownership.
Host buffers used by asynchronous copies belong to that same owner and survive
exception unwinding until synchronization or fatal quarantine.

The ownership and wait edges are:

| Resource | Transfer and terminal owner | Failure/progress bound |
| --- | --- | --- |
| Typed binding/layout | Query import → immutable function data → physical local/merge operators | Query metadata admission and plan lifetime; no shared alias mutation |
| Coefficients/counts/tile scratch | Task resource → reducer owner → returned columns → cuCascade batch | Same task stream is synchronized on success/error; fatal quarantine retains unprovable owners |
| Borrowed operands/partial tables | Read-only task batches → reducer/concat/merge views | Caller retains inputs until task completion; concatenated input survives all queued consumers |
| Retained aggregate batches | Local output → partition queue → existing host/disk spill → restore → merge | Existing reservation/backpressure and batch retirement; no separate retained-state registry |
| Exact join keys | Borrowed input → normalized key columns → partition/probe owners | Identical encoding on both sides; task-stream ordering retains keys through consumers |
| Public output | Finalized columns → admitted codec scratch/output → native result lease → batch release | Existing bounded result credit and query close; numeric failure survives producer cleanup |

The new helpers introduce no locks, independent worker queues or retry loops.
Cancellation continues through the existing task/control owners. Stream waits
must complete or make the runtime fatal; a failed synchronization never
authorizes buffer destruction or healthy reuse. Growth is bounded by admitted
schemas, cuDF row counts and the caller's resource. The allocation-cutpoint
matrix checks both proven/checked reductions and single/multiple segments,
with zero live charged bytes after each success or injected failure.

Decimal256 publication verifies child validity against the parent using the
already admitted codec scratch slab and bounded stack bitmap chunks. A NULL
child under a valid parent is rejected; redundant child masks beneath NULL
parents preserve the parent NULL. No additional GPU allocation is introduced
by result validation.

## Delivery boundary

MO lowering remains opt-in work in C after this increment merges and is pinned.
D owns the real native-MO/embedded-MO all-22 SF1/SF10 public, resource and timing
campaign. E owns strict legacy Flight configuration/recovery cutover; F waits
for E and its verified release artifact. Neither native tests nor capability
advertisement establish migration completion.

## Local terminal evidence

Validated on 2026-10-08 in the frozen `mo` Pixi environment: release `-O3
-DNDEBUG`, CUDA 13.3.73, GCC 14.4.0, cuDF/RMM 26.08, RTX 3070 and driver
615.71.09. No Go production code, durable user format or C ABI-v1 struct layout
changes in this increment. The importer pin remains
`95d9ce8d78490db3991ab6145653716aa3ec42c9`.

| Check | Terminal result |
| --- | --- |
| CPU exact arithmetic and raw aggregate boundaries | 52,645 assertions / 8 cases passed |
| Binding, complete-plan preparation and ordinary type/function controls | 831 assertions / 80 cases passed |
| Native cancellation, ownership and numeric error controls | 606 assertions / 78 cases passed |
| Complete affected GPU group, including native input/admission/result controls | 32,349 assertions / 30 cases passed |
| Production C ABI numeric queries | 1,468 assertions / 8 cases passed under memcheck; zero sanitizer errors |
| Numeric kernels/expressions/aggregates and result codecs under memcheck | 21,294 assertions / 26 cases passed; zero sanitizer errors |
| SDK exporter | 8 tests passed |
| Independent C consumer, ordinary GPU result/release/shutdown/reuse | Passed |

The selected native targets built successfully. Pinned hooks cover every
changed file. The build-tree SDK exports the unchanged literal schema and
updated header, with source revision and dependency/artifact fingerprints;
the delivery export is regenerated from the clean committed source.

The existing nonnumeric `[native_gpu]` control selection also passed 11,034
assertions / 3 cases, but its full leak-check invocation is **not clean**:
it reports a 41,386,248-byte cuDF default pinned-pool allocation and a generic
CUB `EmptyKernel` registration diagnostic in the unchanged input-converter
path. The new aggregate CUDA object contains no CUB scan/empty-kernel symbols.
The two scan-registration diagnostics initially introduced by the aggregate
scan were removed by direct tile-prefix construction. This is not full leak
or resource acceptance; D must reconcile process baselines and complete the
required lifecycle/resource campaign.

Reproduce from the Sirius source directory. `MO_SIRIUS_TEST_CONFIG` must name a
Sirius YAML configuration with GPU 0, bounded GPU/host/disk pools and at least
four scan-manager threads; the production tests fail when it is absent.

```bash
pixi run --frozen -e mo build/mo/extension/sirius/sirius_exact_decimal_unittest
pixi run --frozen -e mo build/mo/extension/sirius/sirius_native_binding_unittest
pixi run --frozen -e mo build/mo/extension/sirius/sirius_native_control_unittest
pixi run --frozen -e mo build/mo/extension/sirius/sirius_native_gpu_unittest \
  '[decimal_import_gpu],[decimal_aggregate_gpu],[exact_decimal_gpu],[native_gpu],[native_result_gpu],[native_admission]'
pixi run --frozen -e mo compute-sanitizer --tool memcheck --error-exitcode 99 \
  build/mo/extension/sirius/sirius_native_gpu_unittest \
  '[decimal_import_gpu],[decimal_aggregate_gpu],[exact_decimal_gpu],[native_result_gpu]'
pixi run --frozen -e mo compute-sanitizer --tool memcheck --error-exitcode 99 \
  build/mo/extension/sirius/sirius_native_numeric_unittest '[native_numeric]'
```

## Retained kernel measurements

The [raw repetitions](mo-exact-decimal-integration-kernels.csv) retain all three
complete benchmark invocations. Each uses a three-second mixed-operation
device warmup, two excluded warmups per case, then seven measured repetitions
at 262,144 rows. Inputs are uploaded before timing. Both routes use the same
caller-owned 64 MiB initial / 128 MiB maximum pool. Allocation, masks, error
storage and required stream completion are timed; output destruction is
excluded symmetrically. No GPU clock settings were changed. GPU measurements
ran sequentially after builds/tests completed.

| Matched case | Ordinary control median ms | Exact median ms | Exact/control |
| --- | --- | --- | --- |
| Evaluator add64, identical values | 0.199669 | 0.163029 | 0.8165 |
| SUM64 → 128, safe full-domain accumulation | 0.276819 | 0.219540 | 0.7931 |

The SUM control widens before cuDF reduction. The exact route includes local
state, counts and mandatory final precision/NULL publication. These are local
kernel/evaluator controls, not full SQL or resource acceptance.

Historical primitive comparison remains separate. The unchanged checked-add64
kernel's three run medians were 0.033070, 0.027530 and 0.033940 ms; the pooled
median is 18.15% above Native A's noncontemporaneous sample. All runs are
retained, and this difference is not declared an accepted performance tradeoff
or a passed rollout gate. D still requires fresh matched baseline/query
measurements and the approved 10% policy before rollout, together with the
required native-MO/embedded-MO SF1/SF10 campaign.
