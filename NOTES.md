# Model and evidence notes

## Affine recurrence and exact scan

At each position, supplied diagonal vectors define
`h_t = a_t * h_(t-1) + b_t`, where `*` is elementwise. A pair `(a,b)` means the
affine map `h -> a*h+b`. Applying left `(a1,b1)` then right `(a2,b2)` gives
`(a2*a1, a2*b1+b2)`. The order matters. Function composition is associative
in exact arithmetic, so a prefix product reconstructs the same state as the
sequential recurrence. Initial state is applied to **each** inclusive prefix;
it need not be zero. Decays may be negative, zero or exceed one: the exact
engine checks algebra, not stability or a trained parameterization.

The implementation reduces adjacent nonoverlapping pairs, recursively scans
the reduced list and reconstructs missing prefixes. Odd tails are retained
without zero padding; a last unpaired prefix already present in the recursive
scan is reused rather than recomputed. The certificate retains every input
interval, node id, operand ids, exact output pair, dependency depth and final
prefix node. It is a reduction/reconstruction scan, not a verbatim copy of a
particular in-place CUDA up-sweep/down-sweep implementation.

For `n=2^k` the actual pair composition count is `2*n-k-2` (zero at n=1).
For general positive n it is less than `2*n`; reduction/reconstruction depth
is at most `2*ceil(log2(n))`, then one unit to apply the prefix to the initial
state. The measured DAG in the checked-in n=16 fixture has 26 compositions
and composition-plus-apply span 7; these are actual counts, not values obtained
by replacing the DAG with that upper bound.

For width d, one diagonal pair composition performs exactly `2*d` scalar
multiplications and `d` additions. Applying n prefix pairs to the initial state
performs `n*d` more multiplications/additions. Sequential supplied-pair updates
perform `n*d` multiplications and `n*d` additions. These counters exclude input
validation, coefficient generation, copying, indexing, comparisons, output
readout, allocation, metadata and arbitrary-precision numerator/denominator
work. One vector compose/apply is treated as one unit of dependency span, not
a scalar instruction or measured parallel machine cycle. Both algorithms run
serial Python. Linear arithmetic-operation work does not imply linear bit
complexity when exact fraction sizes grow, nor less real memory or walltime.

## Fixed diagonal LTI convolution

Given fixed `a,B,C,D` and scalar sequence x:

```
h_t = a*h_(t-1) + B*x_t
y_t = sum_j C_j*h_tj + D*x_t
K_k = sum_j C_j*a_j**k*B_j
boundary_t = sum_j C_j*a_j**(t+1)*h_initial_j
y_t = boundary_t + sum_(s=0..t) K_(t-s)*x_s + D*x_t
```

The extra boundary term is necessary for a nonzero initial state. A zero-input,
nonzero-state negative control catches its omission. This direct convolution
has n(n+1)/2 input-kernel terms, not an FFT complexity claim. Its kernel is
supplied small-model arithmetic, not the structured Cauchy/HiPPO machinery of
S4. Tests cover 120 fixed models including empty inputs, varied state widths,
negative writes, skip terms and nonzero initial states.

## Gated float model and derivatives

A separate floating model has d state components and scalar x:

```
a_tj = sigmoid(theta_j + u_j*x_t)
h_tj = a_tj*h_(t-1)j + B_j*x_t
y_t = sum_j C_j*h_tj + D*x_t
L = sum_t (y_t-target_t)**2 / (2*n)
```

This is affine in the **previous state**, not in the input sequence. Gate
coefficients depend on current x, not h; therefore all coefficients could in
principle be prepared before an associative scan. The float model here is
evaluated sequentially and streamed, with an explicit reverse recurrence. We
do not claim its float outputs are bitwise equal to a parallel reassociation.
The exact supplied-pair path and the sigmoid float path are separate APIs.
There is no exact sigmoid in the rational engine.

Let `dy_t=(y_t-target_t)/n`, and let `carry_j` be the state adjoint from future
positions. In reverse time:

```
z_j = carry_j + dy_t*C_j
gate_j = z_j*h_(t-1)j*a_tj*(1-a_tj)
grad_theta_j += gate_j
grad_u_j     += gate_j*x_t
grad_B_j     += z_j*x_t
grad_C_j     += dy_t*h_tj
grad_D       += dy_t*x_t
grad_x_t     = dy_t*D + sum_j(z_j*B_j + gate_j*u_j)
carry_j      = z_j*a_tj
```

Final carry is the initial-state gradient. The fixture reports all 16 gradients
and central differences with epsilon 1e-6, maximum error
`6.684514053389989e-12`. Tests separately probe 123 parameter/input/initial
derivatives across widths 1,2,3 and lengths 1,2,5, using independent direct
forward losses. Finite differences and floating calculations validate these
cases, not a formal proof of every floating computation or saturated sigmoid.
The stable sigmoid branch avoids overflowing exp for large negative logits,
but floating saturation can still produce gates exactly 0 or 1.

Streaming carries only the last d-dimensional state between chunks. Outputs
match sequential operation order, including all 64 compositions of seven
positions into nonempty chunks. A large legal-input test ensures a computed
carry is not incorrectly rejected by reapplying external-input bounds to it.
The retained traces/output lists still occupy sequence-length memory; constant
state size alone is not a constant-memory implementation claim for this audit.

The nonlinear control uses B=C=1, D=0, theta=0, u=2 and zero initial state.
At x=[1,1], `f(2*x)-2*f(x)` ends at about 0.2024334241, so no fixed **linear**
convolution equals that model for all inputs. This does not rule out degenerate
input-dependent models that happen to be linear, input-conditioned kernels,
or larger nonlinear convolutional architectures. With u=0 the gate is fixed;
a separate test recovers an exactly representable LTI float reference.

## Floating cancellation is a failing control, not a tolerance pass

The separate floating affine API uses the same reduction/reconstruction DAG.
With decay 1, initial 0 and writes [1e16,1,-1e16,1], recurrence ends at 1 and
scan ends at 0, while the rational calculation on exact integer writes ends
at 2. Associativity of real affine maps does not imply associativity of the
floating implementation. The report's numeric equality flag does not inspect
signed-zero or NaN payload bit patterns; nonfinite inputs and computed states
or intermediate pairs are rejected. Float affine inputs have magnitude <=1e100,
state width 1..8 and sequence length <=128. No reproducible GPU reduction,
backward stability or worst-case error-bound certificate is supplied.

## Verification scope

30 tests: 27 core tests and three SVG tests. Besides the cases above, 729 scalar
integer affine triplets check associativity and a noncommuting pair checks
composition order. All 264 combinations of lengths 0..65 and widths 1..4 agree
with a separate closed-form product expansion that uses neither recurrence nor
scan. DAG tests recover every operand and prefix interval and recompute counts.
Work/span bounds are checked for every length 1..128, with exact power-of-two
counts. Strict shape/value/JSON errors return no success payload. Tests check
no input mutation/global RNG effects. The offline demo asserts exact successes
and actual float/nonlinear failure controls. There is no benchmark gate made
weak so the new mechanism looks good.

The SVG binds all 114 displayed values/labels to the report; float displays are
rounded and exact fractions are not. It embeds the entire compressed report,
including intermediate composition pairs and full state/gradient traces. A
hidden intermediate-write mutation changes metadata even without changing a
visible cell. CI checks Python 3.10-3.13, saved full JSON and byte-exact SVG,
and both CLI modes from an installed seven-module wheel outside checkout.

## Sources and distinctions

All code and fixtures are written here; none of these established mechanisms
is claimed as a new algorithm. Sources were read on November 8, 2026.

1. [Guy E. Blelloch, Prefix Sums and Their Applications, CMU-CS-90-190, November 1990](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/CMU-CS-90-190.html): primary report context for associative all-prefix operations and recurrence applications.
2. [NVIDIA GPU Gems 3, Chapter 39, Parallel Prefix Sum with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda): inclusive/exclusive scan distinction, O(n log n) naive work versus O(n) tree scan, and hardware constraints beyond work counts. This project is inclusive, serial Python and not that CUDA kernel.
3. [Gu et al., Efficiently Modeling Long Sequences with Structured State Spaces, 2022 version](https://arxiv.org/html/2111.00396v3): sections 2.3/2.4 recurrent and fixed LTI convolution views. S4's structured parameterization, Cauchy kernels, discretization and trained benchmarks are not implemented here.
4. [Gu and Dao, Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752): input-dependent parameters remove the fixed-convolution route; hardware-aware scan is a separate contribution. Our sigmoid gate is not its discretized selective SSM, full architecture, fused kernel or empirical performance result.
5. [Dive into Deep Learning, Selective State Space Models](https://d2l.smola.org/chapter_recurrent-modern/mamba.html): educational equations for input-dependent decays, writes/reads and reverse state adjoints. Our equations are deliberately smaller and independently differentiated; no code is copied.
6. [Hazy Research, Structured State Spaces: A Brief Survey of Related Models, January 14, 2022](https://hazyresearch.stanford.edu/blog/2022-01-14-s4-2): connects restricted linear recurrences, convolution and surrounding nonlinear layers. We do not generalize the LTI equivalence to arbitrary nonlinear models.
7. [Official state-spaces/mamba repository](https://github.com/state-spaces/mamba/): source-of-truth boundary for actual Mamba-family implementations and requirements. ScanLab is not compatible with its checkpoints, API or GPU kernels, and makes no claim about the latest model generation.

No pretrained model downloads, paid APIs, private data, optimizer quality,
learned selective-copy result, language-model accuracy, long-context retrieval,
S4/Mamba reproduction, continuous-time identification, GPU/CPU speed benchmark,
FFT implementation, low-precision kernel or production reliability claim.
