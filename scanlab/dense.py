"""Small exact coupled matrix-affine recurrences and hand-written derivatives.

All matrices are dense for operation counts, even if entries happen to be zero.
No inverses, AD framework, hardware benchmark, or dense Hessian implementation.
"""
from fractions import Fraction as F
from .exact import vector


def matrix(xs,d):
    if not isinstance(xs,(list,tuple))or len(xs)!=d:raise ValueError('matrix must be square and match width')
    return tuple(vector(row,d)for row in xs)


def prepare(pairs,initial):
    h=vector(initial);d=len(h)
    if d>4:raise ValueError('dense state width must be 1..4')
    if not isinstance(pairs,(list,tuple))or len(pairs)>32:raise ValueError('dense length must be 0..32')
    out=[]
    for p in pairs:
        if not isinstance(p,dict)or set(p)!={'matrix','write'}:raise ValueError('dense pair needs exactly matrix and write')
        out.append((matrix(p['matrix'],d),vector(p['write'],d)))
    return out,h


def eye(d):return tuple(tuple(F(i==j)for j in range(d))for i in range(d))
def zero(d):return tuple((F(0),)*d for _ in range(d))
def transpose(a):return tuple(zip(*a))
def _dotv(x,y):
    terms=[a*b for a,b in zip(x,y)];return sum(terms[1:],terms[0])
def mv(a,x):return tuple(_dotv(row,x)for row in a)
def mm(a,b):return tuple(tuple(_dotv(row,col)for col in zip(*b))for row in a)
def plus(a,b):return tuple(x+y for x,y in zip(a,b))


def compose(left,right):
    a,b=left;c,d=right
    return mm(c,a),plus(mv(c,b),d)


def _scan(ps,h,wrong_order=False):
    nodes=[{'id':'i'+str(i),'start':i,'end':i+1,'depth':0,'pair':p}for i,p in enumerate(ps)];trace=[]
    def combine(l,r):
        assert l['end']==r['start']
        p=(mm(l['pair'][0],r['pair'][0]),plus(mv(r['pair'][0],l['pair'][1]),r['pair'][1]))if wrong_order else compose(l['pair'],r['pair'])
        out={'id':'c'+str(len(trace)),'start':l['start'],'end':r['end'],'depth':max(l['depth'],r['depth'])+1,'pair':p}
        trace.append({**out,'left':l['id'],'right':r['id']});return out
    def prefixes(xs):
        if len(xs)<=1:return xs
        reduced=[combine(xs[i],xs[i+1])if i+1<len(xs)else xs[i]for i in range(0,len(xs),2)]
        rec=prefixes(reduced);out=[]
        for i,x in enumerate(xs):
            if i==0:out.append(x)
            elif i%2 or i==len(xs)-1:out.append(rec[i//2])
            else:out.append(combine(rec[i//2-1],x))
        return out
    pref=prefixes(nodes);states=[plus(mv(p['pair'][0],h),p['pair'][1])for p in pref];d=len(h);n=len(ps);c=len(trace)
    return {'states':states,'composition_trace':trace,'prefixes':pref,
        'counts':{'tokens':n,'state_width':d,'pair_compositions':c,'matrix_matrix_products':c,'matrix_vector_products':c+n,
            'scalar_multiplications':c*(d**3+d*d)+n*d*d,'scalar_additions':c*d**3+n*d*d,
            'sequential_scalar_multiplications':n*d*d,'sequential_scalar_additions':n*d*d,
            'unit_pair_compose_apply_span':max((x['depth']+1 for x in pref),default=0),'sequential_update_span':n},
        'contract':'dense classical dot products, no zero skipping; scalar sum starts with its first term, not counted zero initialization; serial Python unit-pair span, not bit complexity or end-to-end derivatives'}


def sequential(pairs,initial):
    ps,h=prepare(pairs,initial);states=[]
    for a,b in ps:h=plus(mv(a,h),b);states.append(h)
    return states


def scan(pairs,initial):
    ps,h=prepare(pairs,initial);return _scan(ps,h)


def _dot(gradient,direction):
    return sum((x*y for x,y in zip(gradient['initial'],direction['initial'])),F(0))+sum((x*y for p,q in zip(gradient['pairs'],direction['pairs'])for row,vrow in zip(p['matrix'],q['matrix'])for x,y in zip(row,vrow)),F(0))+sum((x*y for p,q in zip(gradient['pairs'],direction['pairs'])for x,y in zip(p['write'],q['write'])),F(0))


def audit(pairs,initial,cotangents,direction):
    ps,h=prepare(pairs,initial);n=len(ps);d=len(h)
    if not isinstance(cotangents,(list,tuple))or len(cotangents)!=n:raise ValueError('cotangent length mismatch')
    q=[vector(x,d)for x in cotangents]
    if not isinstance(direction,dict)or set(direction)!={'pairs','initial'}:raise ValueError('direction needs exactly pairs and initial')
    ds,dh=prepare(direction['pairs'],direction['initial'])
    if len(ds)!=n or len(dh)!=d:raise ValueError('direction shape mismatch')
    dr={'pairs':[{'matrix':a,'write':b}for a,b in ds],'initial':dh}
    forward=_scan(ps,h);states=forward['states'];prev=[h]+states[:-1]
    tangent=_scan([(a,plus(mv(da,x),db))for (a,b),(da,db),x in zip(ps,ds,prev)],dh)
    reverse_maps=[(transpose(ps[t+1][0])if t+1<n else zero(d),q[t])for t in reversed(range(n))]
    reverse=_scan(reverse_maps,(F(0),)*d);lam=list(reversed(reverse['states']))
    def gradient(ls):return {'pairs':[{'matrix':tuple(tuple(x*y for y in p)for x in l),'write':l}for l,p in zip(ls,prev)],'initial':mv(transpose(ps[0][0]),ls[0])if n else(F(0),)*d}
    grad=gradient(lam);lhs=sum((x*y for qr,dt in zip(q,tangent['states'])for x,y in zip(qr,dt)),F(0));rhs=_dot(grad,dr)
    wrong_maps=[(ps[t+1][0]if t+1<n else zero(d),q[t])for t in reversed(range(n))]
    wrong_lam=list(reversed(_scan(wrong_maps,(F(0),)*d)['states']));wrong_dot=_dot(gradient(wrong_lam),dr)
    wrong_states=_scan(ps,h,wrong_order=True)['states']
    return {'states':states,'state_jvp':tangent['states'],'vjp':grad,'adjoints':lam,'transpose':{'cotangent_dot_jvp':lhs,'vjp_dot_direction':rhs,'exact_gap':lhs-rhs},
        'controls':{'missing_transpose':{'vjp_dot_direction':wrong_dot,'exact_gap':lhs-wrong_dot,'detected_for_this_input':wrong_dot!=lhs},
            'reversed_matrix_product':{'states':wrong_states,'differs_for_this_input':wrong_states!=states}},
        'witness':{'forward':forward,'tangent':tangent,'reverse':reverse,'reverse_maps':reverse_maps,'reverse_original_times':list(reversed(range(n)))},
        'contract':'exact derivatives of supplied dense transitions and writes; fixed cotangents; no general AD engine, dense HVP or hardware speed claim'}


def demo():
    mats=[[[1,1],[0,1]],[[0,-1],[1,0]],[[0,0],[0,0]],[[1,'1/2'],[-1,2]],[['1/3',1],[2,'-1/2']],[[0,1],[-1,'1/4']]]
    writes=[[0,1],[1,-1],['2/3','-1/2'],[1,0],[-1,'1/3'],['1/2',2]]
    directions=[[[0,'1/3'],[-1,0]],[[1,0],['1/2',-1]],[[0,1],[-1,0]],[['1/4',0],[0,'-1/3']],[[0,-1],['2/3',1]],[[1,'-1/2'],[0,1]]]
    inp={'pairs':[{'matrix':a,'write':b}for a,b in zip(mats,writes)],'initial':[1,-2],
        'cotangents':[[1,0],[0,-1],['1/2',1],[-1,2],[1,'1/3'],[2,-1]],
        'direction':{'pairs':[{'matrix':a,'write':[F(t,7),F(1-t,5)]}for t,a in enumerate(directions)],'initial':['1/3','-1/2']}}
    return {'input':inp,'audit':audit(**inp)}
