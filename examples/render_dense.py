"""Deterministic coupled-state SVG, full report in compressed metadata."""
import base64,html,json,sys,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scanlab.dense import demo
from scanlab.exact import jsonify


def cells(report):
    r=report['audit'];out=[]
    for t in range(6):
        for group in ('states','state_jvp','adjoints'):
            for j in range(2):out.append((f'state/{t}/{group}/{j}',str(r[group][t][j])))
    for t in range(6):
        p=r['vjp']['pairs'][t]
        for i in range(2):
            for j in range(2):out.append((f'vjp/{t}/matrix/{i}/{j}',str(p['matrix'][i][j])))
        for j in range(2):out.append((f'vjp/{t}/write/{j}',str(p['write'][j])))
    for j in range(2):out.append((f'vjp/initial/{j}',str(r['vjp']['initial'][j])))
    for key in ('cotangent_dot_jvp','vjp_dot_direction','exact_gap'):out.append(('transpose/'+key,str(r['transpose'][key])))
    for key in ('vjp_dot_direction','exact_gap'):out.append(('control/missing_transpose/'+key,str(r['controls']['missing_transpose'][key])))
    for j in range(2):out.append((f'control/wrong_final/{j}',str(r['controls']['reversed_matrix_product']['states'][-1][j])))
    for key in ('pair_compositions','scalar_multiplications','scalar_additions','sequential_scalar_multiplications','sequential_scalar_additions','unit_pair_compose_apply_span','sequential_update_span'):out.append(('work/'+key,str(r['witness']['forward']['counts'][key])))
    return out


def render(report):
    vals=dict(cells(report));lines=['<svg xmlns="http://www.w3.org/2000/svg" width="1220" height="1500" viewBox="0 0 1220 1500">',
        '<rect width="1220" height="1500" fill="#edf3f5"/><rect x="16" y="16" width="1188" height="1468" rx="18" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#25384b}.title{font-size:25px;font-weight:bold}.section{font-size:20px;font-weight:bold}.note{font-size:16px;fill:#59728a}.cell{font-size:17px;font-family:monospace}.head{font-size:16px;font-weight:bold}</style>']
    def text(x,y,s,cls='note'):lines.append(f'<text x="{x}" y="{y}" class="{cls}">{html.escape(s)}</text>')
    def cell(x,y,key):lines.append(f'<text x="{x}" y="{y}" class="cell" data-key="{key}">{html.escape(vals[key])}</text>')
    text(40,56,'ScanLab / coupled states change order, gradients and work','title')
    text(40,87,'Exact 2-by-2 supplied transitions, including shear, rotation and singular reset. Serial Python.')
    text(40,133,'1  Coupled recurrence: h[t] = A[t] h[t-1] + b[t]','section')
    text(40,164,'Apply left then right: (A2 A1, A2 b1 + b2). Reverse adjoint uses A[t+1]^T.')
    xs=[50,115,275,445,615,785,960]
    for x,s in zip(xs,['t','h[0]','h[1]','dh[0]','dh[1]','lambda[0]','lambda[1]']):text(x,210,s,'head')
    for t in range(6):
        y=246+t*34;lines.append(f'<rect x="40" y="{y-23}" width="1140" height="32" fill="'+('#edf3f5'if t%2==0 else'#f8fafb')+'"/>');text(xs[0],y,str(t),'cell')
        for x,(group,j)in zip(xs[1:],[(g,j)for g in ('states','state_jvp','adjoints')for j in range(2)]):cell(x,y,f'state/{t}/{group}/{j}')
    text(40,470,'2  Every matrix-parameter VJP is lambda[t] h[t-1]^T','section')
    xs=[50,115,285,455,625,795,975]
    for x,s in zip(xs,['t','grad A00','grad A01','grad A10','grad A11','grad b0','grad b1']):text(x,512,s,'head')
    for t in range(6):
        y=550+t*34;text(xs[0],y,str(t),'cell');keys=[f'vjp/{t}/matrix/{i}/{j}'for i in range(2)for j in range(2)]+[f'vjp/{t}/write/{j}'for j in range(2)]
        for x,key in zip(xs[1:],keys):cell(x,y,key)
    text(50,772,'Initial-state VJP:');cell(455,772,'vjp/initial/0');cell(625,772,'vjp/initial/1')
    text(40,824,'3  Two failures a diagonal-only model cannot expose','section')
    for y,label,key in [(865,'q dot (J v)','transpose/cotangent_dot_jvp'),(901,'(J^T q) dot v','transpose/vjp_dot_direction'),(937,'Exact transposition gap','transpose/exact_gap'),(979,'Missing transpose in reverse: dot','control/missing_transpose/vjp_dot_direction'),(1015,'Missing-transpose gap','control/missing_transpose/exact_gap')]:text(55,y,label);cell(790,y,key)
    text(55,1057,'Reverse matrix product: wrong final state');cell(790,1057,'control/wrong_final/0');cell(980,1057,'control/wrong_final/1')
    text(40,1110,'4  Shorter unit span, more dense arithmetic','section')
    items=[('Pair compositions','pair_compositions'),('Dense scan scalar multiplications','scalar_multiplications'),('Sequential scalar multiplications','sequential_scalar_multiplications'),('Dense scan scalar additions','scalar_additions'),('Sequential scalar additions','sequential_scalar_additions'),('Scan compose + apply unit span','unit_pair_compose_apply_span'),('Sequential update unit span','sequential_update_span')]
    for i,(label,key)in enumerate(items):y=1150+i*35;text(55,y,label);cell(790,y,'work/'+key)
    text(40,1430,'Dense dot products count all entries, including zeros; no diagonal/zero-skipping optimization.')
    text(40,1460,'Span treats a whole matrix pair operation as one unit. Derivative setup and bit complexity are excluded.')
    payload=zlib.compress(json.dumps(jsonify(report),sort_keys=True,separators=(',',':')).encode());lines.append('<metadata id="full-witness" data-encoding="zlib-base64">'+base64.b64encode(payload).decode()+'</metadata></svg>');return '\n'.join(lines)+'\n'

if __name__=='__main__':Path(sys.argv[1]if len(sys.argv)>1 else'assets/dense-audit.svg').write_text(render(demo()))
