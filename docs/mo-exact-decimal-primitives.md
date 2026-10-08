# MO exact-decimal primitives

This is the first native implementation increment for
[MO #28968](https://github.com/matrixorigin/matrixone/issues/28968), following
the approved numeric design revision
`42a89f09a1d168d02b9583cb3ea7b4de6dbb5634` in MO #29449.

It preserves the existing Decimal64/128 carriers and input/result ownership.
It does not register the MO exact-decimal Substrait family, advertise its
capability, change ordinary DuckDB decimal execution, or admit new MO queries.
The importer is pinned to merged duckdb-substrait #3; its scoped handler is
installed by the following native integration increment.

## Representation and arithmetic

`sirius::mo_decimal::decimal_type` stores physical coefficient width, precision
and scale independently. The native wire remains little-endian signed
two's-complement coefficients: 8, 16 and 32 bytes, with separate validity.
Working Decimal256 precision reaches 76; public result precision stops at 65.
In particular, Decimal256(15,2) remains physically 32 bytes.

Decimal64/128 use existing cuDF fixed-point columns. Decimal256 uses a STRUCT
of signed high64 followed by three unsigned limbs in descending significance;
validity belongs to the parent, not to individual limbs. Input decoding and
result interleaving reuse the existing admitted storage and scratch slab.

The checked host/device core supports arithmetic, comparisons, proven-safe
negation and checked normal rescaling. It uses fixed 512-bit scratch, never
per-row allocation, floating point or numeric strings. Multiplication rescales
and rounds before checking the final physical and declared domains. Division
consumes the supplied result scale; it does not derive MO types or session
settings. Scale reduction and division round half away from zero; DIV and
modulo truncate. NULL, an inactive execution mask and SELECT zero divisors
produce NULL without arithmetic failure.

Scalar failure is `decimal_error::out_of_range`; checked cast failure is
`decimal_error::invalid_input`. These are internal errors here, not new public
C statuses. The native integration increment maps the operation-owned status
without parsing messages or inferring from the enclosing aggregate.

## Ownership and use

`evaluate_decimal_columns` takes the caller's task stream and reservation-aware
RMM resource. Operand and mask owners must outlive that stream's work. Result
values and the bounded per-row error vector are owned together. Failure first
proves stream quiescence; an unprovable owner is retained and reported through
the existing fatal GPU path. No device-wide synchronization or independent
allocator/pool is installed. The pinned cuDF API requires an explicit result
null count, computed on the supplied stream.

## Local validation

Use the frozen Sirius MO Pixi environment in the existing checkout:

```sh
pixi run --frozen -e mo mo-configure-embedding-sdk
pixi run --frozen -e mo cmake --build build/mo --target \
  sirius_exact_decimal_unittest sirius_native_control_unittest \
  sirius_native_gpu_unittest -j8
pixi run --frozen -e mo build/mo/extension/sirius/sirius_exact_decimal_unittest
pixi run --frozen -e mo build/mo/extension/sirius/sirius_native_control_unittest
pixi run --frozen -e mo build/mo/extension/sirius/sirius_native_gpu_unittest \
  '[exact_decimal_gpu],[native_gpu],[native_result_gpu]'
```

The CPU suite includes independent integer/rational boundary vectors, not
expectations computed by the native implementation. GPU tests check those same
independent full-width results, masks, 64/128-bit controls, sliced wide output,
and constant/NULL/retry input decoding. The existing native suite retains its
1/2/4-worker checks. Small native tests are not SF10/all-22 correctness or
performance evidence. Full capability, MO lowering and public/resource/timing
gates remain required before default cutover.

## Kernel measurements

The standalone benchmark uses 262,144 rows, two warm-ups and seven measured
repetitions per operation. One caller-owned RMM pool starts at 64 MiB and is
capped at 128 MiB. Both the cuDF controls and checked kernels use that pool
and one task stream. Timing includes output/error/mask allocation and stream
quiescence, and excludes input upload, warm-ups and output destruction.

```sh
pixi run --frozen -e mo cmake --build build/mo \
  --target sirius_exact_decimal_benchmark -j8
pixi run --frozen -e mo build/mo/extension/sirius/sirius_exact_decimal_benchmark \
  > decimal-kernels.csv
```

The cuDF multiply control widens both operands to Decimal128 first. Merely
requesting a Decimal128 output still computes at Decimal64 width in the
pinned cuDF and can wrap; the signed physical-domain test uses an independent
host `__int128` oracle. The add control measures cuDF's unchecked arithmetic
on the same inputs. Decimal256 has no equivalent cuDF decimal carrier.

The CSV retains every sample. These kernel measurements establish the cost
of these primitives; all-22 numeric query timing and the rollout regression
gate belong to the native plan integration and public campaign.

### Local snapshot

[Raw repetitions](mo-exact-decimal-kernels.csv), measured on 2026-10-08 from
this increment based on Sirius `e2e2f08f9fd1eaa1253297493df0ab11a6078664`:
RTX 3070, NVIDIA driver 615.71.09, CUDA 13.3.73, GCC 14.4.0, frozen Pixi `mo`
environment, DuckDB `069cc9f9b5be802405797faecc284961b07c70ef`, importer
`dd4cab14b82754ca919633913436cd530f496e00`. Pixi lock SHA-256:
`ed808ec91232d916b49e11abb14aba0dcdd82fa13d908342bc970ce584ce4fef`.

| Operation | Median wall time (ms) |
| --- | ---: |
| cuDF add64, unchecked | 0.195870 |
| Checked add64 | 0.027580 |
| cuDF widened multiply64-to128 | 0.499538 |
| Checked multiply64-to128 | 0.500228 |
| Checked add256 | 0.262309 |
| Checked divide256 | 4.662440 |

The checked multiply differs by approximately 0.14% in this sample. Wide
operations report absolute throughput because cuDF has no equivalent carrier.
