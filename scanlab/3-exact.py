"""Strict rational affine scans; dependency counters are not runtime timings."""
from fractions import Fraction as F
import re


def rational(x):
    if isinstance(x,bool) or not isinstance(x,(str,int,F)):raise ValueError('exact int/string/Fraction required')
    if isinstance(x,str) and (len(x)>80 or not re.fullmatch(r'-?\d+(?:/\d+|\.\d+)?',x)):raise ValueError('invalid rational string')
    try:v=F(x)
    except (ValueError,ZeroDivisionError):raise ValueError('invalid rational') from None
    if max(abs(v.numerator).bit_length(),v.denominator.bit_length())>128:raise ValueError('input components exceed 128 bits')
    return v


def vector(xs,size=None):
    if not isinstance(xs,(list,tuple)) or not 1<=len(xs)<=8 or (size is not None and len(xs)!=size):raise ValueError('vector width must be 1..8 and match state width')
    return tuple(rational(x) for x in xs)


def prepare(pairs,initial):
    h=vector(initial);d=len(h)
    if not isinstance(pairs,(list,tuple)) or len(pairs)>128:raise ValueError('sequence must contain 0..128 affine pairs')
    out=[]
    for p in pairs:
        if not isinstance(p,dict) or set(p)!={'decay','write'}:raise ValueError('pair requires exactly decay and write')
        out.append((vector(p['decay'],d),vector(p['write'],d)))
    return out,h


def affine(left,right):
    """Apply left first and right second. Not commutative."""
    a,b=left;c,d=right
    if not (len(a)==len(b)==len(c)==len(d)):raise ValueError('width mismatch')
    return tuple(y*x for x,y in zip(a,c)),tuple(y*x+z for x,y,z in zip(b,c,d))


def sequential(pairs,initial):
    ps,h=prepare(pairs,initial);states=[]
    for a,b in ps:h=tuple(x*y+z for x,y,z in zip(a,h,b));states.append(h)
    return states


def scan(pairs,initial):
    """Pair reduction and prefix reconstruction, including non-power-of-two tails.

    Executes serial Python but records the actual associative composition DAG.
    Node depth counts pair compositions, each treated as a unit-span primitive.
    """
    ps,h=prepare(pairs,initial)
    return _scan_pairs(ps,h)


def _scan_pairs(ps,h):
    d=len(h);nodes=[];trace=[]
    for i,(a,b) in enumerate(ps):nodes.append({'id':'i'+str(i),'start':i,'end':i+1,'depth':0,'pair':(a,b)})
    def combine(left,right):
        assert left['end']==right['start']
        node={'id':'c'+str(len(trace)),'start':left['start'],'end':right['end'],
            'depth':1+max(left['depth'],right['depth']),'pair':affine(left['pair'],right['pair'])}
        trace.append({**node,'left':left['id'],'right':right['id']});return node
    def prefixes(xs):
        if len(xs)<=1:return xs
        reduced=[combine(xs[i],xs[i+1]) if i+1<len(xs) else xs[i] for i in range(0,len(xs),2)]
        rec=prefixes(reduced);out=[]
        for i,x in enumerate(xs):
            if i==0:out.append(x)
            elif i%2:out.append(rec[i//2])
            elif i==len(xs)-1:out.append(rec[i//2])
            else:out.append(combine(rec[i//2-1],x))
        return out
    pref=prefixes(nodes);states=[tuple(a*x+b for a,x,b in zip(n['pair'][0],h,n['pair'][1])) for n in pref]
    span=max((n['depth']+1 for n in pref),default=0);n=len(ps);comps=len(trace)
    return {'states':states,'prefix_pairs':[{'through':i,'node':p['id'],'decay':p['pair'][0],'write':p['pair'][1],'composition_depth':p['depth']}for i,p in enumerate(pref)],
        'composition_trace':trace,'counts':{'tokens':n,'state_width':d,'pair_compositions':comps,
            'scan_scalar_multiplications':2*d*comps+n*d,'scan_scalar_additions':d*comps+n*d,
            'sequential_scalar_multiplications':n*d,'sequential_scalar_additions':n*d,
            'sequential_update_span':n,'scan_composition_plus_apply_span':span},
        'contract':'scalar arithmetic counts on supplied pairs; serial Python, unit composition-DAG span, not timings or bit complexity'}


def audit(pairs,initial):
    r=scan(pairs,initial);s=sequential(pairs,initial);assert r['states']==s
    return {'input':{'pairs':pairs,'initial':initial},'sequential_states':s,'scan':r,'exact_states_identical':True}


def jsonify(x):
    if isinstance(x,F):return str(x)
    if isinstance(x,dict):return {k:jsonify(v)for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [jsonify(v)for v in x]
    return x
