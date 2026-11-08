"""Exact derivatives of supplied diagonal affine pairs, not a general AD engine.

Established reverse-scan and forward-over-reverse rules; see ADJOINT.md.
All computed coefficients bypass external-input bit bounds, as exact.scan does.
"""
from fractions import Fraction as F
from .exact import prepare,vector,_scan_pairs


def _rows(xs,n,d,name):
    if not isinstance(xs,(list,tuple)) or len(xs)!=n:raise ValueError(name+' length mismatch')
    return [vector(x,d) for x in xs]


def _reverse(ps,q):
    """Scan reversed maps: lambda[t] = q[t] + a[t+1]*lambda[t+1]."""
    n=len(ps);d=len(q[0]) if n else 1
    maps=[(ps[t+1][0] if t+1<n else (F(0),)*d,q[t])for t in reversed(range(n))]
    r=_scan_pairs(maps,(F(0),)*d)
    return list(reversed(r['states'])),r


def _grads(ps,h,states,lam):
    previous=[h]+states[:-1]
    return {'pairs':[{'decay':tuple(x*y for x,y in zip(l,p)),'write':l}for l,p in zip(lam,previous)],
        'initial':tuple(a*l for a,l in zip(ps[0][0],lam[0])) if ps else (F(0),)*len(h)}


def dot_direction(gradient,direction):
    """Pair-layout Euclidean pairing; inputs here are already validated."""
    return sum((x*y for x,y in zip(gradient['initial'],direction['initial'])),F(0))+sum((x*y for g,v in zip(gradient['pairs'],direction['pairs'])for key in ('decay','write')for x,y in zip(g[key],v[key])),F(0))


def derivative_audit(pairs,initial,cotangents,direction,targets):
    ps,h=prepare(pairs,initial);n=len(ps);d=len(h)
    q=_rows(cotangents,n,d,'cotangents');target=_rows(targets,n,d,'targets')
    if not isinstance(direction,dict) or set(direction)!={'pairs','initial'}:raise ValueError('direction requires exactly pairs and initial')
    ds,dh=prepare(direction['pairs'],direction['initial'])
    if len(ds)!=n or len(dh)!=d:raise ValueError('direction shape mismatch')
    dr={'pairs':[{'decay':a,'write':b}for a,b in ds],'initial':dh}
    forward=_scan_pairs(ps,h);states=forward['states'];previous=[h]+states[:-1]
    tangent_maps=[(a,tuple(da*x+db for da,x,db in zip(va,prev,vb)))for (a,b),(va,vb),prev in zip(ps,ds,previous)]
    tangent=_scan_pairs(tangent_maps,dh);dots=tangent['states']
    lam,reverse=_reverse(ps,q) if n else ([],_scan_pairs([], (F(0),)*d))
    gradient=_grads(ps,h,states,lam)
    lhs=sum((a*b for qt,dt in zip(q,dots)for a,b in zip(qt,dt)),F(0));rhs=dot_direction(gradient,dr)
    residual=[tuple(x-y for x,y in zip(s,t))for s,t in zip(states,target)]
    loss=sum((x*x/F(2)for r in residual for x in r),F(0))
    loss_lam,loss_reverse=_reverse(ps,residual) if n else ([],_scan_pairs([], (F(0),)*d))
    # Differentiate the reverse recurrence. The shifted da multiplies next lambda.
    forcing=[tuple(dots[t][j]+(ds[t+1][0][j]*loss_lam[t+1][j] if t+1<n else 0)for j in range(d))for t in range(n)]
    dlam,hvp_reverse=_reverse(ps,forcing) if n else ([],_scan_pairs([], (F(0),)*d))
    prevdot=[dh]+dots[:-1]
    hvp={'pairs':[{'decay':tuple(dl*x+l*dx for dl,x,l,dx in zip(lt,p,lamt,dp)),'write':lt}for lt,p,lamt,dp in zip(dlam,previous,loss_lam,prevdot)],
        'initial':tuple(da*l+a*dl for da,l,a,dl in zip(ds[0][0],loss_lam[0],ps[0][0],dlam[0])) if n else (F(0),)*d}
    # Deliberate bug: use a[t], not a[t+1], in reverse propagation.
    wrong_maps=[(ps[t][0],q[t])for t in reversed(range(n))]
    wrong_lam=list(reversed(_scan_pairs(wrong_maps,(F(0),)*d)['states']))
    wrong_gradient=_grads(ps,h,states,wrong_lam)
    wrong=dot_direction(wrong_gradient,dr)
    return {'states':states,'direction':dr,'cotangents':q,'state_jvp':dots,'vjp':gradient,
        'transpose':{'cotangent_dot_jvp':lhs,'vjp_dot_direction':rhs,'exact_gap':lhs-rhs},
        'squared_loss':{'value':loss,'targets':target,'residual':residual,'gradient':_grads(ps,h,states,loss_lam),
            'hessian_vector':hvp,'directional_first':sum((x*y for rt,dt in zip(residual,dots)for x,y in zip(rt,dt)),F(0)),
            'directional_second':dot_direction(hvp,dr)},
        'witness':{'forward':forward,'tangent':tangent,'reverse':reverse,'loss_reverse':loss_reverse,'hvp_reverse':hvp_reverse,
            'reverse_index_map':[{'scan_index':i,'original_time':n-1-i,'propagates_from_time':n-i if i else None}for i in range(n)],
            'hvp_reverse_forcing':forcing,'loss_adjoints':loss_lam,'loss_adjoint_jvp':dlam},
        'wrong_index_control':{'vjp_dot_direction':wrong,'transpose_gap':lhs-wrong,'detected_for_this_input':wrong!=lhs},
        'contract':'exact rational supplied-pair derivatives, fixed cotangents and targets; serial Python, not general AD, no dense Jacobian/Hessian, no runtime speed claim'}


def demo():
    pairs=[{'decay':[x],'write':[y]}for x,y in zip(('1/2','-1/3','0','2','3/4','-1/2'),('1','-2','1/3','0','2/3','1'))]
    direction={'pairs':[{'decay':[x],'write':[y]}for x,y in zip(('1/3','-1/2','2/3','1/4','-1/5','1'),('0','1/3','-1','1/2','2/3','-1/4'))],'initial':['-2/3']}
    inp={'pairs':pairs,'initial':['2'],'cotangents':[['1'],['-2'],['1/2'],['0'],['3/2'],['-1']],
        'direction':direction,'targets':[['0'],['1/3'],['-1'],['2'],['1/2'],['0']]}
    return {'input':inp,'audit':derivative_audit(**inp)}
