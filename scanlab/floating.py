"""Floating reassociation control using the SAME scan DAG as the exact engine."""
import math
from .exact import _scan_pairs


def audit(pairs,initial):
    def vec(xs,width=None):
        if not isinstance(xs,(list,tuple))or not 1<=len(xs)<=8 or(width is not None and len(xs)!=width):raise ValueError('invalid width')
        if any(isinstance(x,bool)or not isinstance(x,(int,float))or not math.isfinite(x)or abs(x)>1e100 for x in xs):raise ValueError('finite numeric entries <=1e100 required')
        return tuple(float(x) for x in xs)
    h=vec(initial);width=len(h)
    if not isinstance(pairs,(list,tuple))or len(pairs)>128:raise ValueError('invalid length')
    ps=[]
    for pair in pairs:
        if not isinstance(pair,dict)or set(pair)!={'decay','write'}:raise ValueError('invalid affine pair')
        ps.append((vec(pair['decay'],width),vec(pair['write'],width)))
    r=_scan_pairs(ps,h);seq=[]
    for a,b in ps:h=tuple(x*y+z for x,y,z in zip(a,h,b));seq.append(h)
    if not all(math.isfinite(x)for state in seq+r['states']for x in state):raise ValueError('nonfinite computed state')
    for row in r['composition_trace']:
        if not all(math.isfinite(x)for v in row['pair']for x in v):raise ValueError('nonfinite intermediate affine pair')
    gap=max((abs(x-y)for s,t in zip(seq,r['states'])for x,y in zip(s,t)),default=0.0)
    return {'sequential_states':seq,'scan':r,'max_absolute_difference':gap,'numeric_states_equal':seq==r['states'],
        'contract':'float operation reassociation can change results. Numeric equality here does not inspect signed-zero bit patterns.'}
