#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
python -m scanlab adjoint-demo > /tmp/scanlab-adjoint-demo.json
cmp /tmp/scanlab-adjoint-demo.json examples/adjoint-demo.json
python -m scanlab adjoint examples/adjoint-input.json > /tmp/scanlab-adjoint-audit.json
python - <<'PY'
import json
from fractions import Fraction as F
r=json.load(open('/tmp/scanlab-adjoint-demo.json'))['audit']
a=json.load(open('/tmp/scanlab-adjoint-audit.json'))
assert a==r
assert F(r['transpose']['exact_gap'])==0
assert F(r['wrong_index_control']['transpose_gap'])==F(-259,36)
assert r['wrong_index_control']['detected_for_this_input']
assert F(r['squared_loss']['directional_second'])==F(20591177,414720)
print('Exact transpose, nonzero wrong-index control, and polynomial-tested curvature witness: OK')
PY
python examples/render_adjoint.py /tmp/scanlab-adjoint-audit.svg
cmp /tmp/scanlab-adjoint-audit.svg assets/adjoint-audit.svg
