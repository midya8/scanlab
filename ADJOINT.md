# Exact reverse scans and curvature witnesses

v0.2 adds derivatives of the supplied **diagonal affine coefficients**, independent of the smaller float sigmoid-gated model in v0.1. This is an implementation and certificate exercise in established scan differentiation, not a new differentiation algorithm. Python executes every operation serially.

## Variables and indexing

Times are zero-based: `h[t] = a[t] * h[t-1] + b[t]`, and `h[-1] = initial`. Products are coordinatewise. Parameters are every entry of `initial`, every `a[t]`, and every `b[t]`. The caller supplies equally shaped rational perturbations `dinitial`, `da`, `db`, fixed output cotangents `q[t]`, and fixed squared-loss targets. The same strict exact-input bounds apply: 0..128 tokens, width 1..8, no float/bool, strings at most 80 characters and input rational numerator/denominator at most 128 bits. Computed intermediates are not truncated or revalidated as external inputs. Fraction bit growth can be large.

For arbitrary fixed `q`, the scalar functional is `sum_t q[t] dot h[t]`. Its total adjoint satisfies

```
lambda[t] = q[t] + a[t+1] * lambda[t+1]
lambda[n-1] = q[n-1]
```

Notice the **next** decay. Reversing the pairs without shifting decays is wrong. Build maps `(a[t+1], q[t])` in reverse time, with a zero terminal decay, and apply the existing inclusive affine scan to initial zero. Prefix index `i` represents original time `n-1-i`. The output includes this index mapping and the full reverse DAG. A zero decay needs no special division or inverse, so reset steps work.

The parameter VJP is

```
grad_a[t] = lambda[t] * h[t-1]
grad_b[t] = lambda[t]
grad_initial = a[0] * lambda[0]
```

An empty sequence has no output or loss and all initial-state derivatives are zero.

## Tangents and the transposition certificate

With the primal states known,

```
dh[t] = a[t] * dh[t-1] + da[t] * h[t-1] + db[t]
```

is another affine scan. It gives `J v` without materializing `J`. The audit returns both sides of `q dot (J v) = (J^T q) dot v`. Both are exact `Fraction` values. The fixture gives `-817/120` on each side, with exact gap zero.

A deliberate wrong-index control uses `a[t]` instead of `a[t+1]` in the reverse maps. It gives `139/360` on the VJP side and a transposition gap `-259/36` on this fixture. This is a control, not a bad implementation used by the real path. The control can be invisible on special inputs, for example zero cotangents or direction. The report names detection for this input only; tests also cover invisible controls.

## Squared-loss Hessian-vector products

For `L = sum_t ||h[t] - target[t]||^2 / 2`, use residual `r[t] = h[t] - target[t]` as the output cotangent. This produces loss adjoints `mu[t]` and the exact loss gradient. Differentiating those adjoints gives

```
dmu[t] = dh[t] + da[t+1] * mu[t+1] + a[t+1] * dmu[t+1]
dmu[n-1] = dh[n-1]
```

This is another reverse affine scan with forcing `dh[t] + da[t+1] * mu[t+1]`. The Hessian-vector entries are

```
(H v)_a[t] = dmu[t] * h[t-1] + mu[t] * dh[t-1]
(H v)_b[t] = dmu[t]
(H v)_initial = da[0] * mu[0] + a[0] * dmu[0]
```

No dense Hessian is constructed. This is hand-written forward-over-reverse for this recurrence, not an AD engine, symbolic compiler, neural-network trainer or general nonlinear scan rule. Targets and cotangents are held fixed under differentiation. The recurrence is nonlinear in its jointly varied decay/state parameters; no global convexity or positive-curvature guarantee is claimed.

Fixture loss is `2473/288`, first directional derivative `83689/17280`, and second directional derivative `20591177/414720`. All parameter gradient/HVP coordinates, residuals, primal/tangent/reverse DAGs and reverse forcing terms are retained in the JSON and compressed SVG metadata.

## Independent checks, not an identity alone

A transposition identity can agree when both sides share a bug. Tests therefore also use an independent bivariate polynomial expansion. Substitute every parameter with `p + epsilon*v + eta*u`, multiply polynomials through a plain sequential recurrence, then expand the linear and squared losses. There are no scan calls, derivative recurrences or floating finite differences in that oracle.

- 36 models: lengths 0..5, widths 1..3, two fixed seeds. Each of **432 parameter coordinates** checks the linear-functional VJP, squared-loss gradient and HVP against the corresponding polynomial `eta` and `epsilon*eta` coefficients.
- 24 models: lengths 0..7, widths 1..3. Every state JVP, loss, first directional derivative, and second derivative matches full polynomial coefficients, including the factor of two on the squared epsilon term.
- 24 separate two-direction Hessian symmetry checks: `u^T H v = v^T H u`.
- 51 models: lengths 0..16, widths 1, 2, 4. Each reverse adjoint is checked against an independently summed product of future decays, including negative and zero decays.
- Empty, length-one and maximum 128-by-8 shapes; computed values above the input bit limit; malformed shapes/types/JSON; no input mutation or global RNG changes; saved report and CLI parity.
- The SVG binds **68 numeric cells** to the report and embeds the entire witness. Changing an undisplayed trace changes its metadata. All v0.1 tests and saved demo/figure remain unchanged.

## Work and span boundaries

The report records five scan DAGs: primal, tangent, arbitrary-cotangent reverse, loss reverse, and HVP reverse. Their existing counters describe only compositions and prefix applications on supplied pairs, treating each whole-vector composition/application as one dependency-span unit. They do not count the elementwise construction of tangent/HVP coefficients, gradient products, scalar loss/dot reductions, allocations, validation or fraction bit complexity. These phases have dependencies and their spans cannot be read as one end-to-end latency by taking a minimum or a maximum. This is not a throughput or training-time benchmark. Stored states and witnesses use memory growing with sequence length.

## Attribution and reading

Implementation, tests, fixtures and figure are original project code. Derivative rules and scan ideas are established; no upstream implementation was copied.

1. Guy Blelloch, [Prefix Sums and Their Applications](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/CMU-CS-90-190.html), CMU, November 1990. Primary report for associative prefix scans; the project's pair-reduction/reconstruction is not a literal CUDA implementation.
2. Justin Chiu, [Differentiating through an associative parallel scan](https://justinchiu.netlify.app/blog/pscan_diff/), January 8, 2024. Practitioner derivation of the shifted reverse affine recurrence.
3. Bruun, Larsen, Hinnerskov and Oancea, [Reverse-Mode AD of Multi-Reduce and Scan in Futhark](https://futhark-lang.org/publications/ifl23-ad.pdf), IFL 2023. Primary paper, especially section 5: reverse scan rules and specialization. Their compiler and GPU measurements are not implemented or reproduced here.
4. JAX, [associative_scan API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html). Official semantics of associative and reverse scans. This package does not depend on JAX or reproduce its execution model.
5. JAX, [The autodiff cookbook: JVPs and VJPs](https://docs.jax.dev/en/latest/301/cookbook.html). Official explanation of JVP/VJP and forward-over-reverse HVPs. We write these rules by hand rather than call JAX.
6. Steven G. Johnson, [Notes on Adjoint Methods for 18.335](https://math.mit.edu/~stevenj/18.336/adjoint.pdf), updated April 30, 2021. University teaching notes grounding reverse-mode/adjoint sensitivities. This project concerns finite recurrences, not implicit PDE or eigenproblem solvers.

Sources retrieved November 8, 2026. Paper/compiler/hardware results are background only, not project performance claims.
