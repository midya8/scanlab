"""Direct finite causal convolution of a supplied diagonal LTI recurrence."""
from .exact import rational,vector,sequential


def compare(decay,write,read,skip,inputs,initial):
    a=vector(decay);d=len(a);b=vector(write,d);c=vector(read,d);h=vector(initial,d);skip=rational(skip)
    if not isinstance(inputs,(list,tuple)) or len(inputs)>128:raise ValueError('input length must be 0..128')
    xs=[rational(x) for x in inputs];pairs=[{'decay':a,'write':[z*x for z in b]} for x in xs]
    states=[]
    for x in xs:
        h=tuple(aj*hj+bj*x for aj,hj,bj in zip(a,h,b));states.append(h)
    h=vector(initial,d)
    rec=[sum(z*s for z,s in zip(c,state))+skip*x for state,x in zip(states,xs)]
    kernel=[sum(cj*(aj**k)*bj for aj,bj,cj in zip(a,b,c)) for k in range(len(xs))]
    boundary=[sum(cj*(aj**(t+1))*hj for aj,cj,hj in zip(a,c,h)) for t in range(len(xs))]
    conv=[boundary[t]+sum(kernel[t-j]*xs[j]for j in range(t+1))+skip*xs[t] for t in range(len(xs))]
    assert rec==conv
    return {'inputs':xs,'decay':a,'write':b,'read':c,'skip':skip,'initial':h,'states':states,'impulse_kernel':kernel,
        'initial_state_contribution':boundary,'recurrent_outputs':rec,'convolution_outputs':conv,'exact_outputs_identical':True,
        'direct_convolution_input_terms':len(xs)*(len(xs)+1)//2,
        'contract':'fixed time/input-invariant parameters; direct quadratic convolution, no FFT or learned S4 kernel'}
