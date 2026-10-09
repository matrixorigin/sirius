# Exact-decimal NULL predicates and condition-free joins

The MatrixOne public numeric fixture found that IS NOT NULL enters cuDF's
AST unary path with an exact decimal carrier. Decimal256's private four-limb
STRUCT is not a supported AST unary operand. Ordinary IS NULL can reach the
same path inside a larger boolean expression.

The invariant is that SQL nullness depends only on the canonical top-level
validity mask, for every physical decimal width. Coefficient values and private
limb masks do not define another SQL value. The negation is an unsupported AST
operation, a NULL boolean result, or a predicate that inspects carrier limbs.

The expression evaluator owns this boundary. Materialize exact decimal NULL
predicates with cuDF is_null/is_valid on the caller's stream and resource. When
an enclosing expression requests AST mode, retain the resulting boolean column
through the existing temporary-column owner and reference it from the parent
AST. The standalone AST translator declines the unsupported private carrier.
No numeric URI, function identity, capability or C ABI layout changes.

The change adds no wait, registry, worker, copy of coefficients or independent
allocation owner. The child and result retain their existing task lifetime;
result allocation uses the task resource. Errors and cancellation unwind through
the existing evaluator/task cleanup. The boolean result is one byte per row,
bounded by the admitted input batch. IS NOT NULL uses is_valid directly.

The focused GPU regression covers Decimal64/128/256, all-valid and mixed-NULL
columns, a nonzero-offset all-NULL slice, empty input and typed NULL literals.
It checks nonnullable BOOL8 values independently under materialize, interpreted
AST and JIT AST strategies, both directly and inside NOT. Standalone translation
must decline the private carriers. Existing ordinary unary tests remain controls.

MO C keeps the original public CASE/COALESCE plus NULL-predicate query and adds
an all-width NULL-predicate control. C must pin the merged native fix before
delivery. Native and small public tests do not establish D's full-data or
performance acceptance.

Exact preparation also skips DuckDB's ordinary optimizer, leaving JOIN ON true
as an unsupported ANY_JOIN. Preserve that join's kind, output projection maps
and mark binding while lowering the literal TRUE condition to equal constant
TINYINT keys in the existing GPU join. The join planner already materializes
computed keys and excludes those private columns from public output. FALSE,
NULL and arbitrary predicates do not enter this transformation. No scalar-row
assumption or new cross-product operator is needed.

The production C ABI regression covers INNER/LEFT/RIGHT/FULL joins with both
sides present, either side empty and both empty. It independently checks
multiplicity, duplicate right values, right NULLs and Decimal256 high limbs.
FALSE and NULL controls retain rejection before input or GPU work starts.
Existing join tasks, partition/concat owners and resource admission retain all
buffers and errors. The additional key costs one byte per input row per side
plus the existing materialization/hash scratch, charged to those owners. Full
query performance remains D's gate.

Local development validation on 2026-10-09 passes all 22 MO native preparations
without starting readers, and the complete public MySQL numeric fixture,
including exact values/metadata/errors, prepared division, NULL predicates and
scalar aggregates on empty/all-NULL input. This used an explicitly marked
development SDK; MO's delivered gitlink remains merged Sirius #28 until this
fix merges. The standalone exact/ordinary GPU selection and native binding
suite pass. Clean SDK fingerprints and final native test counts are recorded
in the PR delivery evidence. None of these checks establishes SF1/SF10 parity,
full leak acceptance or a passed performance rollout gate.
