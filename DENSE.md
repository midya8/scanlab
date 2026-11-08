# Coupled dense-state scans

v0.3 removes the diagonal-transition assumption for a separate, small exact path. A state has 1..4 real rational coordinates, a sequence 0..32 steps, and each supplied transition is a full square matrix. This is a reference implementation of established matrix-affine scans and derivatives, not a novel scan, general AD engine, dense Hessian implementation or hardware benchmark. The original diagonal, gated-float and diagonal HVP paths remain separate and unchanged.

## Order is part of the contract

Column states obey `h[t] = A[t] h[t-1] + b[t]`. Applying pair 1 and then pair 2 yields

```
(A1, b1) followed by (A2, b2) = (A2 A1, A2 b1 + b2)
```

Matrix products are associative but generally not commutative. A scalar or diagonal-only test cannot expose every mistaken matrix order. The fixture begins with a shear and a quarter-turn, nonzero writes and initial state `[1,-2]`. The correct state after these two steps is `[2,-2]`; reversing only their matrix product gives `[3,0]` while keeping the write composition correct. The report's deliberate wrong-matrix-order scan preserves all its wrong states rather than only a pass/fail label.

A pair-reduction/prefix-reconstruction scan keeps every full matrix pair, operand ID, half-open input interval and composition depth. Odd tails are retained; length zero and one have no compositions. A zero transition matrix is a valid reset, not an error. Nothing uses a matrix inverse, so singular transitions need no special numerical workaround. The fixture's reset writes `[2/3,-1/2]`, erasing the primal dependence on earlier state; a perturbation of that reset matrix can still have a nonzero derivative.

## Full-matrix derivatives need a transpose

The caller supplies a fixed cotangent `q[t]` for each output state and a perturbation for each matrix entry, write entry and initial-state entry. For the scalar functional `sum_t q[t]^T h[t]`:

```
dh[t] = A[t] dh[t-1] + dA[t] h[t-1] + db[t]
lambda[t] = q[t] + A[t+1]^T lambda[t+1]
grad_A[t] = lambda[t] h[t-1]^T
grad_b[t] = lambda[t]
grad_initial = A[0]^T lambda[0]
```

Terminal lambda is the last cotangent. The reverse scan uses the **next transposed matrix**, not the current matrix and not the next untransposed matrix. The report retains all reverse maps, original-time indices, adjoints and full primal/tangent/reverse DAGs. A Frobenius inner product pairs every matrix-gradient entry with its perturbation; write and initial-state products are included too. No Jacobian is materialized.

On the fixture, `q dot (J v)` and `(J^T q) dot v` both equal `-440563/15120` exactly. Omitting the transpose only inside the reverse propagation gives `-487549/15120`, gap `7831/2520`. Initial-state projection still uses the correct transpose in that control, so it isolates the missing transpose in the recurrence rather than conflating two bugs. A scalar-width or zero-cotangent case can make this control invisible; tests explicitly cover both. No universal bug-detection claim is made.

The final correct state is `[9/2,187/36]`. The deliberately reversed-matrix-product scan instead ends at `[-14/9,2/3]` on this fixture. These are constructed controls, not evidence about trained-model quality.

## Count dense arithmetic, not diagonal arithmetic

All entries are processed, including zero entries. Classical matrix products are used, with no sparsity, diagonal, identity, zero, Strassen or library optimizations. Each dot product uses its first term as its sum initializer, then adds the remaining terms. Allocation and assignments are outside scalar arithmetic counts.

For state width `d`, a pair composition uses one matrix-matrix product, one matrix-vector product, and one vector addition:

- Scalar multiplications: `d^3 + d^2`.
- Scalar additions: `d^3`.
- Each prefix application uses `d^2` multiplications and `d^2` additions.
- Each sequential state update uses those same `d^2` multiplications and additions.

With `c` recorded compositions and `n` tokens, scan totals are `c*(d^3+d^2)+n*d^2` multiplications and `c*d^3+n*d^2` additions. An instrumented rational type counts the actual multiplication/addition calls in the implementation on 51 length/width cases, independently of these formulas.

The six-token, two-state fixture has seven pair compositions, seven matrix-matrix products and thirteen matrix-vector products. Forward scan uses **108 multiplications and 80 additions**, versus **24 and 24** sequentially. Unit pair-compose/apply span is **4 versus 6**, treating a whole dense pair composition/application as one unit. That span convention deliberately hides matrix width cost; it is not constant hardware time, scalar-circuit span or latency. The cubic composition term explains why the earlier diagonal cost model must not be reused here.

Counts are per supplied-pair scan, not end-to-end derivatives. Coefficient construction for tangents, matrix transposes, outer-product gradients, scalar cotangent reductions, input validation, copies and fraction bit complexity are not included. State and witness memory grow with sequence length. This is serial Python: no hardware parallelism, wall-clock speedup or training throughput is claimed.

## Independent verification

63 tests include all 46 prior tests plus 17 dense-path tests:

- 216 associativity cases with rotations, shears, zero, identity and other nonsymmetric matrices, plus a noncommuting-order failure control.
- 96 length/width cases: lengths 0..31, widths 1..3. Rebuild each prefix independently as a product of `(d+1)`-dimensional homogeneous transforms; compare every output to the dense scan and plain recurrence. This oracle has separate matrix-product code.
- Full DAG operand, interval, depth and matrix-pair checks for all lengths 0..32. Each interval pair is checked against its independent homogeneous product.
- 51 instrumented arithmetic cases: lengths 0..16, widths 1, 2, 4. Counts come from overloaded exact-rational multiplication/addition calls, not a second copy of the scan cost formula.
- 36 models, lengths 0..5, widths 1..3, two fixed seeds. **672 matrix/write/initial-state parameter coordinates** have their VJP checked against independently expanded full epsilon-polynomial coefficients. No scan, derivative recurrence or finite-difference tolerance is used by that oracle.
- 24 full-direction polynomial models check every state JVP and the scalar directional derivative, in addition to transposition.
- 27 diagonal restrictions agree with the preserved v0.1 diagonal scan. This is state equality, not equal operation counts: the dense implementation still processes all zero entries.
- Empty, one-step and maximum 32-by-4 shapes, singular resets, computed values larger than the input bit bound, malformed matrices/directions/JSON, no input mutation or global RNG changes, two CLI modes and exact saved-report parity.
- The new SVG binds **88 report-value cells** and embeds the full report. An undisplayed reverse-index mutation changes its metadata. Prior saved figures and demos are unchanged.

Exact input syntax follows the original rational path: integers, rational/decimal strings or Python `Fraction`, no bool or float; at most 80 characters per numeric string and 128 bits per input numerator/denominator. Full matrices are square and match the state width. JSON is at most 1 MB, has exactly `pairs`, `initial`, `cotangents`, `direction`, and rejects duplicate/unknown keys or nonfinite literals atomically with exit 2 and empty stdout. Computed fractions are not clipped to the external input bound, and bit growth can dominate runtime.

## Attribution

Implementation, instrumentation, tests, fixtures and figures are original project code. The algebra and derivative rules are established. No upstream code, compiler, GPU result or model weights were copied or reproduced.

1. Guy Blelloch, [Prefix Sums and Their Applications](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/CMU-CS-90-190.html), CMU, November 1990. Associative scan foundation; our reduction/reconstruction is an educational implementation, not a verbatim CUDA kernel.
2. JAX, [associative_scan API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html). Official associative/reverse scan semantics, including matrix-product example. This project has no JAX dependency or JAX execution guarantees.
3. Justin Chiu, [Differentiating through an associative parallel scan](https://justinchiu.netlify.app/blog/pscan_diff/), January 8, 2024. Practitioner derivation of reverse affine adjoints and transpose order.
4. Bruun, Larsen, Hinnerskov and Oancea, [Reverse-Mode AD of Multi-Reduce and Scan in Futhark](https://futhark-lang.org/publications/ifl23-ad.pdf), IFL 2023, especially section 5. Primary research on reverse scan differentiation and matrix/Jacobian cost. We do not reproduce their compiler or GPU benchmarks.
5. [Dive into Deep Learning, Automatic Differentiation](https://d2l.smola.org/chapter_mdl-calculus/mdl-matrix-calculus-autodiff.html), section 25.4, equation 25.4.6. Educational derivation of the matrix-weight gradient as an adjoint/previous-state outer product. The package does not implement a neural-network framework.
6. Petersen and Pedersen, [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf), version November 15, 2012. Reference for matrix transpose, product and derivative identities. No matrix inversion is required here.

Sources retrieved November 8, 2026. This finite exact laboratory does not establish real-task accuracy, stability of arbitrary floating matrix recurrences, or practical speedups from general dense scans.
