# Embedded MO exact-decimal import and scalar execution

This is the first of the two native integration increments approved for
[MO #28968](https://github.com/matrixorigin/matrixone/issues/28968) and
[MO #28966](https://github.com/matrixorigin/matrixone/issues/28966). It follows
merged Sirius #25 and importer #4. The semantic contract remains the approved
MO #29449 document blob `42a89f09a1d168d02b9583cb3ea7b4de6dbb5634`.

This document records the Native A boundary and its historical measurements.
[Native B](mo-exact-decimal-integration.md) supersedes the exposure boundary
below with complete embedded preparation, aggregates, keys and capability 16u.

## Contract and ownership

Physical coefficient width, declared precision, scale, and nullability survive
Substrait import, DuckDB binding, Sirius translation, cloning and GPU evaluation.
The first type owner is the validated import descriptor. Ordinary DuckDB
decimal inference cannot substitute a narrower carrier. Decimal64/128 use
tagged BIGINT/HUGEINT DuckDB carriers; Decimal256 uses the signed-high/unsigned
low limb STRUCT introduced by #25. Sirius retains a distinct MO decimal type.

The private alias records the descriptor and is validated against the complete
physical carrier on conversion. It is internal binding metadata, not a new
wire encoding or registered SQL type. The public literal wire contract is the
approved `ExactDecimalLiteral` protobuf with one fixed-width coefficient field,
defined under `proto/matrixone/sirius/numeric/v1/`.
The build-tree SDK exports and fingerprints this schema beside its C header.
The pinned protobuf reader accepts standard unknown fields and length varints;
the final coefficient must still have exactly the declared physical width.

One scoped importer validates exact-decimal identities and output type anchors
from its own bounded serialized plan. Private scalar markers bind immutable
input/output types and retain MO's declared result. They cannot execute on CPU.
The typed signature survives the DuckDB/Sirius round trip. Existing function
IDs retain their values; exact-decimal IDs are appended.

GPU scalar execution uses the caller's task stream and resource. Checked
results retain the failing operation's internal error kind. CASE and COALESCE
propagate active rows, so an inactive arithmetic or narrowing failure does not
become a query failure. At most one conditional branch's temporary columns
remain owned at a time. Failure proves stream quiescence before cleanup;
unprovable owners transfer to the existing fatal process quarantine.
cuDF STRUCT selection may superimpose the parent bitmap on child columns.
After validating canonical operands and proving stream completion, conditional
selection removes only those redundant child masks without copying coefficient
data. A following exact operation verifies that parent-only validity survives.

No process-wide type/function registry or GPU allocator is replaced. Query
metadata remains bounded by the existing plan/admission contract. Literal
broadcast and error reduction allocate only through the supplied resource;
there is no per-row host callback or numeric string conversion.

## Exposure boundary

The helper and direct-consumer tests can import and evaluate the scalar family.
Production embedded admission still rejects the new extension, and the exact
capability remains clear. Ordinary importer calls continue without the scoped
handler. Aggregates, keys, complete admission, public statuses 12/13 and
capability advertisement belong to the next native increment.

The remaining approved sequence is:

1. Sirius import/type/scalar plumbing, with capability disabled (this increment).
2. Sirius aggregates/keys, complete native admission and capability.
3. MO lowering, reconstruction, typed errors and query-local evidence.
4. Real native-MO/embedded-MO SF1/SF10 parity, resource and performance campaign.
5. MO embedded default and strict Flight configuration/recovery cutover.
6. Sidecar Flight retirement after the verified MO release artifact exists.

The campaign requires streams 2 plus controls 1/4, keeps the 10% numeric
regression policy, and retains the 1.0 Flight+MO gate only on available equivalent
coverage. Native MO is the complete correctness oracle. Direct TAE and Docker
image/base changes remain outside this migration milestone.

## Validation map

| Closure | Risk | Required evidence |
| --- | --- | --- |
| Type/literal and binding descriptors | R2 | Independent byte/domain controls, malformed identity/width rejection, binding and clone/round-trip fidelity |
| GPU scalar and conditional owners | R3 | Actual GPU values, masks, scalar/cast error kinds, empty inputs, inactive failures, same-stream cleanup |
| Existing consumers and exposure | R2 | Ordinary type/function controls, native ownership suites, unchanged C ABI/capability |
| Build, schema and documentation | R0/R1 | Native build, registered tests and pinned static hooks |

Full public SQL acceptance is the MO lowering/campaign boundary. This increment
does not claim all-22 execution, resource rollout acceptance or migration completion.

The standalone decimal benchmark retains the primitive measurements and adds
ordinary/exact expression add controls on the same values and resource. Report
their raw repetitions separately: these controls measure evaluator cost, not
full-domain error equivalence or complete query performance.

## Local evidence

Validated on 2026-10-08 using the frozen `mo` Pixi environment: CUDA 13.3.73,
GCC 14.4.0, cuDF/RMM 26.08, an RTX 3070 and driver 615.71.09.

| Check | Result |
| --- | --- |
| Native binding, logical types and ordinary function controls | 727 assertions / 75 cases |
| CPU exact-decimal foundation | 52,330 assertions / 5 cases |
| Native ownership and protocol controls | 594 assertions / 77 cases |
| Imported expressions, decimal kernels and native GPU/result controls | 31,459 assertions / 18 cases |
| Compute Sanitizer memcheck, imported expressions | 1,664 assertions / 7 cases; zero sanitizer errors |
| SDK export tests | 8 tests |
| Independent C ABI consumer and SDK schema fingerprint | Passed |

All listed tests passed. The native targets built and pinned pre-commit hooks
passed. Memcheck covers invalid memory access; it does not establish the later
resource, concurrency or leak acceptance campaign.

Reproduce the GPU and sanitizer groups from the Sirius source directory:

```bash
pixi run --frozen -e mo build/mo/extension/sirius/sirius_native_gpu_unittest \
  '[decimal_import_gpu],[exact_decimal_gpu],[native_gpu],[native_result_gpu]'
pixi run --frozen -e mo compute-sanitizer --tool memcheck --error-exitcode 99 \
  build/mo/extension/sirius/sirius_native_gpu_unittest '[decimal_import_gpu]'
pixi run --frozen -e mo build/mo/extension/sirius/sirius_exact_decimal_benchmark
```

The benchmark uses two warmups; the
[raw scalar repetitions](mo-exact-decimal-import-kernels.csv) record seven
measured repetitions at 262,144 rows. Against the
recorded #25 primitive run, checked-operation median wall time changed by
+1.49% (add64), +0.77% (widened multiply), -0.05% (add256), and +3.83%
(divide256). This is a historical local comparison, not a concurrent SQL
acceptance run. The new ordinary/exact expression add controls measured
0.199190/0.158990 ms median on identical values and the same bounded pool.
