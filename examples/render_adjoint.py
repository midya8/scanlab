"""Deterministic exact-derivative SVG. Every numeric cell is provenance-tagged."""
import base64,gzip,html,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scanlab.adjoint import demo
from scanlab.exact import jsonify


def cells(report):
    r=report['audit'];out=[]
    for t in range(len(r['states'])):
        for key,value in [('state',r['states'][t][0]),('jvp',r['state_jvp'][t][0]),('cotangent',r['cotangents'][t][0]),('adjoint',r['vjp']['pairs'][t]['write'][0])]:out.append((f'linear/{t}/{key}',str(value)))
    for t in range(len(r['states'])):
        for group in ('gradient','hessian_vector'):
            for key in ('decay','write'):out.append((f'loss/{t}/{group}/{key}',str(r['squared_loss'][group]['pairs'][t][key][0])))
    for key in ('cotangent_dot_jvp','vjp_dot_direction','exact_gap'):out.append(('transpose/'+key,str(r['transpose'][key])))
    for key in ('vjp_dot_direction','transpose_gap'):out.append(('wrong/'+key,str(r['wrong_index_control'][key])))
    for key in ('value','directional_first','directional_second'):out.append(('loss/'+key,str(r['squared_loss'][key])))
    for group in ('gradient','hessian_vector'):out.append(('loss/'+group+'/initial',str(r['squared_loss'][group]['initial'][0])))
    for key in ('forward','tangent','reverse','loss_reverse','hvp_reverse'):
        for counter in ('pair_compositions','scan_composition_plus_apply_span'):out.append(('work/'+key+'/'+counter,str(r['witness'][key]['counts'][counter])))
    return out


def render(report):
    records=cells(report);values=dict(records);lines=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="1510" viewBox="0 0 1120 1510">',
        '<rect width="1120" height="1510" fill="#edf3f5"/><rect x="16" y="16" width="1088" height="1478" rx="18" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#25384b}.title{font-size:25px;font-weight:bold}.section{font-size:20px;font-weight:bold}.note{font-size:16px;fill:#59728a}.cell{font-size:17px;font-family:monospace}.head{font-size:16px;font-weight:bold}</style>']
    def text(x,y,s,cls='note'):lines.append(f'<text x="{x}" y="{y}" class="{cls}">{html.escape(s)}</text>')
    def cell(x,y,key):lines.append(f'<text x="{x}" y="{y}" class="cell" data-key="{key}">{html.escape(values[key])}</text>')
    text(40,56,'ScanLab / reverse scans and exact derivative certificates','title')
    text(40,88,'Supplied rational affine pairs. No dense Jacobian/Hessian. Serial Python, not GPU timing.')
    text(40,136,'1  Reverse time is not enough: shift the decay by one step','section')
    text(40,167,'h[t] = a[t] h[t-1] + b[t]; lambda[t] = q[t] + a[t+1] lambda[t+1]')
    text(40,195,'Tangent: dh[t] = a[t] dh[t-1] + da[t] h[t-1] + db[t]')
    xs=[55,135,335,550,770];heads=['t','h[t]','dh[t]','q[t]','lambda[t]']
    for x,s in zip(xs,heads):text(x,235,s,'head')
    for t in range(6):
        y=270+34*t;lines.append(f'<rect x="40" y="{y-23}" width="1040" height="32" fill="'+('#edf3f5'if t%2==0 else'#f8fafb')+'"/>');text(xs[0],y,str(t),'cell')
        for x,key in zip(xs[1:],('state','jvp','cotangent','adjoint')):cell(x,y,f'linear/{t}/{key}')
    text(40,500,'2  Transposition test: q dot (J v) equals (J^T q) dot v','section')
    for y,label,key in [(537,'q dot (J v)','transpose/cotangent_dot_jvp'),(571,'(J^T q) dot v','transpose/vjp_dot_direction'),(605,'Exact gap','transpose/exact_gap'),(647,'Wrong a[t] instead of a[t+1]: dot','wrong/vjp_dot_direction'),(681,'Wrong-index transposition gap','wrong/transpose_gap')]:text(55,y,label);cell(690,y,key)
    text(40,731,'3  Squared loss: exact gradient and Hessian-vector product','section')
    text(40,760,'L = sum (h[t] - target[t])^2 / 2. Targets and cotangents stay fixed.')
    xs=[55,140,355,575,815]
    for x,s in zip(xs,['t','dL/da[t]','dL/db[t]','(H v)_a[t]','(H v)_b[t]']):text(x,802,s,'head')
    for t in range(6):
        y=837+34*t;text(xs[0],y,str(t),'cell')
        for x,(group,key)in zip(xs[1:],[('gradient','decay'),('gradient','write'),('hessian_vector','decay'),('hessian_vector','write')]):cell(x,y,f'loss/{t}/{group}/{key}')
    for y,label,key in [(1065,'Initial-state gradient','loss/gradient/initial'),(1099,'Initial-state H v','loss/hessian_vector/initial'),(1133,'Loss L','loss/value'),(1167,'First directional derivative','loss/directional_first'),(1201,'Second directional derivative v^T H v','loss/directional_second')]:text(55,y,label);cell(690,y,key)
    text(40,1250,'4  Five recorded scan DAGs, not total derivative runtime','section')
    for x,s in zip([55,475,760],['Scan phase','Pair compositions','Compose + apply span']):text(x,1285,s,'head')
    for i,key in enumerate(('forward','tangent','reverse','loss_reverse','hvp_reverse')):
        y=1316+i*29;text(55,y,key);cell(530,y,'work/'+key+'/pair_compositions');cell(825,y,'work/'+key+'/scan_composition_plus_apply_span')
    text(40,1480,'Coefficients, elementwise derivatives, loss reductions and validation are outside these DAG counters.')
    payload=json.dumps(jsonify(report),sort_keys=True,separators=(',',':')).encode()
    compressed=gzip.compress(payload,mtime=0)
    # RFC 1952 OS header byte: Python 3.11/3.12 may expose zlib's host OS.
    # Canonicalize to 255 (unknown) without changing the deflate payload or CRC.
    compressed=compressed[:9]+b'\xff'+compressed[10:]
    enc=base64.b64encode(compressed).decode()
    lines.append('<metadata id="full-witness" data-encoding="gzip-base64">'+enc+'</metadata></svg>');return '\n'.join(lines)+'\n'

if __name__=='__main__':Path(sys.argv[1]if len(sys.argv)>1 else'assets/adjoint-audit.svg').write_text(render(demo()))
