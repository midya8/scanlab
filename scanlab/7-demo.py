"""Exact law controls and independently checked floating derivatives."""
from fractions import Fraction as F
from .exact import audit
from .lti import compare
from .floating import audit as float_audit
from .gated import forward,loss_and_grad,streaming,finite_difference_audit


def demo():
    pairs=[{'decay':[F(1,2),F(3,4)],'write':[F((i%3)-1,3),F((i%5)-2,5)]}for i in range(16)]
    exact=audit(pairs,[F(1,3),F(-1,5)])
    lti=compare(['1/2','3/4'],[1,'-1/3'],[2,'1/2'],'1/5',[1,0,-1,2,0,1],[1,-1])
    p={'theta':[-.4,.2],'u':[1.1,-.7],'B':[.8,-.5],'C':[1.2,.6],'D':.1};xs=[.3,-.8,1.2,.1,-.4];h=[.2,-.1];ys=[.1,-.2,.3,.2,-.1]
    g=loss_and_grad(p,xs,h,ys);stream=streaming(p,xs,h,[2,1,2]);assert stream['outputs']==g['forward']['outputs']
    # A single changing gate suffices to violate linear superposition. B,C fixed.
    control={'theta':[0.0],'u':[2.0],'B':[1.0],'C':[1.0],'D':0.0}
    one=forward(control,[1,1],[0])['outputs'];two=forward(control,[2,2],[0])['outputs'];gap=[b-2*a for a,b in zip(one,two)]
    assert abs(gap[-1])>.1
    cancel=[{'decay':[1.0],'write':[x]}for x in (1e16,1.,-1e16,1.)]
    rounding=float_audit(cancel,[0.0]);assert rounding['max_absolute_difference']==1.0
    return {'floating_cancellation':rounding,'exact_affine':exact,'fixed_lti':lti,'gated':{'parameters':p,'inputs':xs,'initial':h,'targets':ys,'gradient':g,'gradient_check':finite_difference_audit(p,xs,h,ys),'streaming':stream},
        'not_a_fixed_convolution':{'parameters':control,'f_x':one,'f_2x':two,'superposition_residual':gap},
        'limits':'Serial CPU Python. Exact rational algebra separate from float sigmoid derivatives. No Mamba/S4 implementation, training benchmark, GPU speed or bitwise reassociation claim.'}
