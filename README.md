# ScanLab

A causal state-space laboratory in dependency-free Python: exact affine
recurrence, a work-efficient associative prefix scan, fixed LTI convolution,
and hand-derived reverse gradients for a small input-gated recurrence.

![Exact algebra, actual work, floating failure and gradients](assets/scan-audit.svg)

## The claims are deliberately separate

- **Same exact states, different dependency graph.** For the 16-token,
  two-state rational fixture, sequential recurrence and scan agree exactly.
  The scan performs 26 pair compositions. Unit composition/apply span drops
  from 16 to 7, while scalar multiplications increase from 32 to 136 and
  additions from 32 to 84. Both run serially in Python. Span is not latency,
  scalar counts are not arbitrary-precision bit complexity, and there is no
  GPU or throughput claim.
- **Convolution needs fixed dynamics.** The fixed diagonal LTI path exactly
  matches direct causal convolution, including nonzero initial state and
  skip connection. An input-dependent gate breaks superposition by
  `0.20243342` on the explicit control, so a single fixed linear convolution
  cannot represent that model on all inputs. The supplied convolution is
  direct quadratic work, not an FFT or an S4 kernel implementation.
- **Real algebra is not floating arithmetic.** With writes
  `[1e16,1,-1e16,1]`, decay 1 and initial 0, sequential floats finish at 1,
  the same scan DAG finishes at 0, and exact rational algebra finishes at 2.
  No bitwise or exact-sigmoid equivalence claim is hidden behind a tolerance.
- **Differentiated, not benchmarked as a trained model.** The input-gated model
  exposes every state, gate, output, parameter gradient, input gradient and
  initial-state gradient. All 123 derivative probes across nine shapes pass
  independent central differences. The displayed 16-probe fixture has maximum
  error about `6.685e-12`. All 64 chunk partitions of a seven-token input
  preserve the causal streaming outputs. A tested descent step reduces the
  fixture loss; no learned-task, language-model or generalization claim.

## Run offline

```bash
python -m scanlab demo
python -m scanlab audit examples/input.json
python -m unittest discover -s tests -v
bash examples/demo.sh
python examples/render_demo.py --check
```

30 tests include 729 scalar affine associativity cases, 264 length/width cases
against a closed-form expansion, 120 LTI models, full DAG scope/dependency/count
checks, every length through 128 for work/span bounds, 123 independent gradient
probes, 64 streaming partitions, strict input errors and both real failure
controls. The 114-cell figure binds the complete report, including hidden
intermediate factors, not only headline numbers. Five CI jobs check Python
3.10-3.13 plus both commands from an installed seven-module wheel outside checkout.

```python
from scanlab.exact import audit
from scanlab.lti import compare
from scanlab.gated import forward, loss_and_grad, streaming

certificate = audit(
    [{"decay": ["1/2"], "write": [1]},
     {"decay": ["3/4"], "write": [-1]}],
    initial=["1/3"],
)
```

The strict JSON audit input has exactly `pairs` and `initial`; each pair has
exactly `decay` and `write`, vectors matching the 1..8 state width. Sequence
length is 0..128. Exact numbers are integers or rational/decimal strings
(Python API also accepts `Fraction`), never floats/bools. String length <=80,
input numerator/denominator <=128 bits, payload <=1 MB. Unknown/duplicate keys,
nonfinite JSON and invalid shapes fail with exit 2 and empty stdout. Internal
rational results may grow beyond the input bound. The separate float model
accepts finite float/int inputs and parameters of magnitude <=100, not bools;
its input-independent state width is 1..8 and sequence length <=128.

[Mathematics, derivative equations, count contract, source notes and limits](NOTES.md).

This code is written here, with the scan/state-space literature attributed.
It is **not Mamba, S4, a GPU kernel, a trained sequence benchmark, an exact
floating reduction, or a new scan algorithm**. It uses a simple input-sigmoid
gate, not Mamba's input-dependent continuous-time discretization or full block.
MIT licensed.
