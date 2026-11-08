#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
python -m scanlab dense-demo > /tmp/scanlab-dense-demo.json
cmp /tmp/scanlab-dense-demo.json examples/dense-demo.json
python -m scanlab dense examples/dense-input.json >/tmp/scanlab-dense-audit.json
python - <<'PY'
import json
from fractions import Fraction as F
r=json.load(open('/tmp/scanlab-dense-demo.json'))['audit'];a=json.load(open('/tmp/scanlab-dense-audit.json'));assert a==r
assert F(r['transpose']['exact_gap'])==0
assert F(r['controls']['missing_transpose']['exact_gap'])==F(7831,2520)
assert r['controls']['reversed_matrix_product']['differs_for_this_input']
assert r['states'][-1]==['9/2','187/36']
c=r['witness']['forward']['counts'];assert c['scalar_multiplications']==108 and c['scalar_additions']==80
print('Coupled exact states, transposition, wrong-order/missing-transpose controls and dense work: OK')
PY
python examples/render_dense.py /tmp/scanlab-dense-audit.svg
cmp /tmp/scanlab-dense-audit.svg assets/dense-audit.svg
