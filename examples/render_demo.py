"""Deterministic SVG with source-bound cells and complete witness metadata."""
import base64,html,json,sys,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scanlab.demo import demo
from scanlab.exact import jsonify


def values(r):
    out={};c=r['exact_affine']['scan']['counts']
    for key in ('tokens','state_width','pair_compositions','scan_scalar_multiplications','sequential_scalar_multiplications','scan_scalar_additions','sequential_scalar_additions','scan_composition_plus_apply_span','sequential_update_span'):out['count-'+key]=c[key]
    l=r['fixed_lti']
    for i in range(len(l['inputs'])):
        for key in ('inputs','impulse_kernel','initial_state_contribution','recurrent_outputs','convolution_outputs'):out[f'lti-{i}-{key}']=l[key][i]
    f=r['floating_cancellation']
    for i in range(4):out[f'float-{i}-seq']=format(f['sequential_states'][i][0],'.8g');out[f'float-{i}-scan']=format(f['scan']['states'][i][0],'.8g')
    g=r['gated']['gradient_check'];out['gradient-count']=g['count'];out['gradient-error']=format(g['max_absolute_error'],'.3e')
    for i,row in enumerate(g['rows']):
        out[f'grad-{i}-name']=row['field']+'['+str(row['index'])+']'
        for key in ('analytic','central_difference','absolute_error'):out[f'grad-{i}-{key}']=format(row[key],'.6e')
    out['nonlinear-gap']=format(r['not_a_fixed_convolution']['superposition_residual'][-1],'.8g')
    return out


def render(report):
    r=jsonify(report);v=values(r);out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="1640" viewBox="0 0 1120 1640">','<rect width="1120" height="1640" fill="#edf3f5"/>','<rect x="16" y="16" width="1088" height="1608" rx="18" fill="white"/>']
    payload=base64.b64encode(zlib.compress(json.dumps(r,sort_keys=True,separators=(',',':')).encode())).decode();out.append('<metadata>'+payload+'</metadata>')
    def text(x,y,s,size=17,bold=False,color='#273746'):out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}"'+(' font-weight="bold"'if bold else '')+'>'+html.escape(str(s))+'</text>')
    def cell(x,y,k):out.append(f'<text id="cell-{k}" data-value="{html.escape(str(v[k]),quote=True)}" x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="17" fill="#273746">'+html.escape(str(v[k]))+'</text>')
    text(40,58,'ScanLab / parallelizable algebra is not a speed benchmark',26,True)
    text(40,89,'Exact rational recurrence + work-efficient prefix DAG. A separate float model has explicit reverse gradients.',16,color='#627789')
    text(40,136,'1 Same exact states, shorter dependency span, more scalar work',21,True)
    text(40,168,'Apply left then right: (a1,b1) composed with (a2,b2) = (a2*a1, a2*b1+b2).')
    text(40,198,'Tokens:');cell(120,198,'count-tokens');text(190,198,'State width:');cell(310,198,'count-state_width');text(390,198,'Pair compositions:');cell(570,198,'count-pair_compositions')
    for x,s in ((60,'Counter'),(480,'Sequential recurrence'),(820,'Scan + prefix apply')):text(x,240,s,16,True)
    for i,(label,seq,scan)in enumerate((('Scalar multiplications','sequential_scalar_multiplications','scan_scalar_multiplications'),('Scalar additions','sequential_scalar_additions','scan_scalar_additions'),('Unit dependency span','sequential_update_span','scan_composition_plus_apply_span'))):
        y=276+i*35;text(60,y,label);cell(540,y,'count-'+seq);cell(880,y,'count-'+scan)
    text(40,389,'Serial Python executes both paths. Span treats a whole vector composition/apply as one unit.',16,color='#627789')
    text(40,432,'2 Fixed LTI recurrence equals causal convolution, including initial state',21,True)
    xs=(55,195,395,615,830)
    for x,s in zip(xs,('Input','Kernel at lag t','Initial-state term','Recurrent y','Convolution y')):text(x,472,s,16,True)
    for i in range(6):
        y=505+i*31
        if i%2==0:out.append(f'<rect x="40" y="{y-22}" width="1030" height="30" fill="#edf3f5"/>')
        for x,key in zip(xs,('inputs','impulse_kernel','initial_state_contribution','recurrent_outputs','convolution_outputs')):cell(x,y,f'lti-{i}-{key}')
    text(40,710,'Direct quadratic finite convolution, no FFT, HiPPO, S4 kernel or continuous-time discretization.',16,color='#627789')
    text(40,752,'3 Floating arithmetic is not associative',21,True)
    text(40,783,'decay=1; writes=[1e16,1,-1e16,1]; initial=0. Exact rational final state is 2.')
    for x,s in ((65,'Token'),(270,'Sequential float'),(610,'Scan float')):text(x,821,s,16,True)
    for i in range(4):text(80,852+i*29,i+1);cell(300,852+i*29,f'float-{i}-seq');cell(640,852+i*29,f'float-{i}-scan')
    text(40,980,'4 Input-dependent gates: differentiated, streaming, not a fixed convolution',21,True)
    text(40,1012,'h_t = sigmoid(theta + u*x_t) * h_(t-1) + B*x_t; y_t = C dot h_t + D*x_t',17)
    text(40,1045,'Checked derivatives:');cell(245,1045,'gradient-count');text(315,1045,'Max central-difference error:');cell(600,1045,'gradient-error')
    for x,s in ((60,'Derivative'),(335,'Analytic'),(575,'Central difference'),(845,'Absolute error')):text(x,1085,s,16,True)
    for i in range(16):
        y=1114+i*24
        if i%2==0:out.append(f'<rect x="40" y="{y-18}" width="1030" height="23" fill="#edf3f5"/>')
        for x,key in zip((60,335,575,845),('name','analytic','central_difference','absolute_error')):cell(x,y,f'grad-{i}-{key}')
    text(40,1520,'Fixed-kernel failure: final f(2x)-2f(x) =');cell(430,1520,'nonlinear-gap');text(630,1520,'(float values rounded above)',15,color='#627789')
    text(40,1560,'Streaming chunks preserve causal outputs. Gates are functions of the current input, not prior state.',16,color='#627789')
    text(40,1590,'No trained-model quality, GPU throughput, Mamba implementation, exact sigmoid or bitwise equivalence claim.',16,color='#627789')
    out.append('</svg>');return '\n'.join(out)+'\n'

if __name__=='__main__':
    path=Path(__file__).resolve().parents[1]/'assets/scan-audit.svg';text=render(demo())
    if '--check'in sys.argv:
        if path.read_text()!=text:raise SystemExit('figure differs')
    else:path.write_text(text)
