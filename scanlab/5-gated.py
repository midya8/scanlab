"""Small input-dependent diagonal recurrence with explicit reverse derivatives.

Not Mamba: no continuous-time discretization, projections, conv front end or SSM block.
"""
import math


def number(x):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or abs(x)>100:raise ValueError('finite float/int of magnitude <=100 required')
    return float(x)


def validate(params,inputs,initial):
    if not isinstance(params,dict) or set(params)!={'theta','u','B','C','D'}:raise ValueError('wrong parameter keys')
    if not isinstance(initial,(list,tuple)) or not 1<=len(initial)<=8:raise ValueError('state width must be 1..8')
    d=len(initial);p={}
    for k in ('theta','u','B','C'):
        if not isinstance(params[k],(list,tuple)) or len(params[k])!=d:raise ValueError('parameter width mismatch')
        p[k]=[number(x) for x in params[k]]
    p['D']=number(params['D'])
    if not isinstance(inputs,(list,tuple)) or len(inputs)>128:raise ValueError('input length must be 0..128')
    return p,[number(x)for x in inputs],[number(x)for x in initial]


def sigmoid(x):
    if x>=0:return 1/(1+math.exp(-x))
    z=math.exp(x);return z/(1+z)


def forward(params,inputs,initial):
    p,xs,h=validate(params,inputs,initial)
    return _forward(p,xs,h)


def _forward(p,xs,h):
    trace=[];ys=[]
    for x in xs:
        a=[sigmoid(t+u*x)for t,u in zip(p['theta'],p['u'])];prev=h[:]
        h=[z*s+b*x for z,s,b in zip(a,h,p['B'])];y=sum(c*s for c,s in zip(p['C'],h))+p['D']*x
        if not all(math.isfinite(z) for z in h+[y]):raise ValueError('nonfinite computed output')
        trace.append({'input':x,'decay':a,'previous_state':prev,'state':h[:],'output':y});ys.append(y)
    return {'outputs':ys,'final_state':h,'trace':trace}


def loss_and_grad(params,inputs,initial,targets):
    p,xs,h=validate(params,inputs,initial)
    if not xs or not isinstance(targets,(list,tuple)) or len(targets)!=len(xs):raise ValueError('nonempty matching targets required')
    targets=[number(x)for x in targets];r=forward(p,xs,h);n=len(xs);width=len(h)
    grad={k:[0.0]*width for k in ('theta','u','B','C')};grad['D']=0.0;dx=[0.0]*n;carry=[0.0]*width
    for i in reversed(range(n)):
        row=r['trace'][i];x=xs[i];dy=(row['output']-targets[i])/n;grad['D']+=dy*x;dx[i]+=dy*p['D']
        for j in range(width):
            z=carry[j]+dy*p['C'][j];a=row['decay'][j];dg=z*row['previous_state'][j]*a*(1-a)
            grad['theta'][j]+=dg;grad['u'][j]+=dg*x;grad['B'][j]+=z*x;grad['C'][j]+=dy*row['state'][j]
            dx[i]+=z*p['B'][j]+dg*p['u'][j];carry[j]=z*a
    loss=sum((y-t)**2 for y,t in zip(r['outputs'],targets))/(2*n)
    return {'loss':loss,'parameter_gradients':grad,'input_gradients':dx,'initial_gradients':carry,'forward':r}


def streaming(params,inputs,initial,chunk_sizes):
    p,xs,h=validate(params,inputs,initial)
    if not isinstance(chunk_sizes,(list,tuple)) or any(isinstance(x,bool)or not isinstance(x,int)or x<1 for x in chunk_sizes)or sum(chunk_sizes)!=len(xs):raise ValueError('positive chunks must partition inputs')
    out=[];at=0;boundaries=[]
    for size in chunk_sizes:
        r=_forward(p,xs[at:at+size],h);h=r['final_state'];out+=r['outputs'];at+=size;boundaries.append({'through':at,'state':h[:]})
    return {'outputs':out,'final_state':h,'boundaries':boundaries}


def finite_difference_audit(params,inputs,initial,targets,epsilon=1e-6):
    """Diagnostic central differences, not a proof of derivatives."""
    import copy
    if isinstance(epsilon,bool)or not isinstance(epsilon,(int,float))or not 0<epsilon<=.001:raise ValueError('epsilon must be in(0,.001]')
    p,xs,h=validate(params,inputs,initial);result=loss_and_grad(p,xs,h,targets);n=len(xs)
    def loss(pp,xx,hh):return sum((y-number(t))**2 for y,t in zip(forward(pp,xx,hh)['outputs'],targets))/(2*n)
    rows=[]
    for k in ('theta','u','B','C','D','input','initial'):
        size=1 if k=='D' else len(xs)if k=='input'else len(h)
        for j in range(size):
            up,dn=copy.deepcopy(p),copy.deepcopy(p);xu,xd=xs[:],xs[:];hu,hd=h[:],h[:]
            if k=='D':up[k]+=epsilon;dn[k]-=epsilon;analytic=result['parameter_gradients'][k]
            elif k=='input':xu[j]+=epsilon;xd[j]-=epsilon;analytic=result['input_gradients'][j]
            elif k=='initial':hu[j]+=epsilon;hd[j]-=epsilon;analytic=result['initial_gradients'][j]
            else:up[k][j]+=epsilon;dn[k][j]-=epsilon;analytic=result['parameter_gradients'][k][j]
            numeric=(loss(up,xu,hu)-loss(dn,xd,hd))/(2*epsilon)
            rows.append({'field':k,'index':j,'analytic':analytic,'central_difference':numeric,'absolute_error':abs(analytic-numeric)})
    return {'epsilon':epsilon,'rows':rows,'count':len(rows),'max_absolute_error':max(r['absolute_error']for r in rows)}
