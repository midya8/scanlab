#!/usr/bin/env bash
set -eu
python -m scanlab demo >/tmp/scanlab-demo.json
cmp /tmp/scanlab-demo.json examples/demo.json
python -m scanlab audit examples/input.json >/tmp/scanlab-audit.json
python examples/render_demo.py --check
python - <<'PY'
import json
r=json.load(open('/tmp/scanlab-demo.json'))
assert r['exact_affine']['exact_states_identical']
assert r['fixed_lti']['exact_outputs_identical']
assert r['floating_cancellation']['max_absolute_difference']==1
assert r['gated']['gradient_check']['max_absolute_error']<1e-8
assert abs(r['not_a_fixed_convolution']['superposition_residual'][-1])>.1
print('Exact controls pass; floating cancellation and nonlinear fixed-kernel controls fail as intended.')
PY
