"""Independent homogeneous matrix products, full polynomials and op instrumentation."""
import copy,json,random,subprocess,sys,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from scanlab import dense
from scanlab.exact import jsonify,scan as diagonal_scan


def matmul(a,b):
    # Independent rectangular loop, intentionally not dense.mm.
    out=[[F(0)for _ in b[0]]for _ in a]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):out[i][j]+=a[i][k]*b[k][j]
    return out


def homogeneous(p):
    d=len(p['write']);return [list(row)+[b]for row,b in zip(p['matrix'],p['write'])]+[[F(0)]*d+[F(1)]]


def expansion(pairs,initial):
    state=[[F(x)]for x in initial]+[[F(1)]];d=len(initial);result=[]
    for t in range(len(pairs)):
        # Rebuild each prefix from independent homogeneous transforms.
        prod=[[F(i==j)for j in range(d+1)]for i in range(d+1)]
        for p in pairs[:t+1]:prod=matmul(homogeneous(p),prod)
        result.append(tuple(row[0]for row in matmul(prod,state)[:-1]))
    return result


def flat(v):return list(v['initial'])+[x for p in v['pairs']for row in p['matrix']for x in row]+[x for p in v['pairs']for x in p['write']]


def layout(xs,n,d):
    m=d+n*d*d;return {'initial':xs[:d],'pairs':[{'matrix':[xs[d+t*d*d+i*d:d+t*d*d+(i+1)*d]for i in range(d)],'write':xs[m+t*d:m+(t+1)*d]}for t in range(n)]}


def case(n,d,seed):
    rng=random.Random(seed)
    def params():return layout([F(rng.randrange(-2,3),rng.randrange(1,4))for _ in range(d+n*(d*d+d))],n,d)
    return {**params(),'direction':params(),'cotangents':[[F(rng.randrange(-2,3),3)for _ in range(d)]for _ in range(n)]}


def padd(a,b):return [sum(xs,F(0))for xs in zip(a+[F(0)]*max(0,len(b)-len(a)),b+[F(0)]*max(0,len(a)-len(b)))]
def pmul(a,b):
    out=[F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i+j]+=x*y
    return out


def polynomial(inp,perturb):
    """Expand full epsilon polynomials by direct sequential matrix-vector products."""
    n=len(inp['pairs']);d=len(inp['initial']);base=flat(inp);direction=flat(perturb)
    lifts=layout([[F(x),F(y)]for x,y in zip(base,direction)],n,d);h=lifts['initial'];states=[];loss=[F(0)]
    for t,p in enumerate(lifts['pairs']):
        h=[padd(sum_polys([pmul(a,x)for a,x in zip(row,h)]),b)for row,b in zip(p['matrix'],p['write'])];states.append(h)
        loss=padd(loss,sum_polys([[v*F(q)for v in x]for x,q in zip(h,inp['cotangents'][t])]))
    return states,loss


def sum_polys(xs):
    out=[F(0)]
    for x in xs:out=padd(out,x)
    return out


class Count(F):
    mult=0;adds=0
    def __mul__(self,other):Count.mult+=1;return Count(super().__mul__(other))
    def __add__(self,other):Count.adds+=1;return Count(super().__add__(other))
    __radd__=__add__
    __rmul__=__mul__


class Dense(unittest.TestCase):
    def test_fixture_and_controls(self):
        r=dense.demo()['audit'];self.assertEqual(r['states'][-1],(F(9,2),F(187,36)));self.assertEqual(r['transpose']['exact_gap'],0)
        self.assertEqual(r['transpose']['cotangent_dot_jvp'],F(-440563,15120));self.assertEqual(r['controls']['missing_transpose']['exact_gap'],F(7831,2520));self.assertTrue(r['controls']['reversed_matrix_product']['differs_for_this_input'])
        c=r['witness']['forward']['counts'];self.assertEqual((c['scalar_multiplications'],c['scalar_additions'],c['unit_pair_compose_apply_span']),(108,80,4))

    def test_216_associativity_cases(self):
        matrices=[((F(1),F(0)),(F(0),F(1))),((F(0),F(-1)),(F(1),F(0))),((F(1),F(1)),(F(0),F(1))),dense.zero(2),((F(1),F(2)),(F(-1),F(0))),((F(-2),F(0)),(F(1),F(3)))];pairs=[(a,(F(i,3),F(1-i,2)))for i,a in enumerate(matrices)]
        for a in pairs:
            for b in pairs:
                for c in pairs:self.assertEqual(dense.compose(dense.compose(a,b),c),dense.compose(a,dense.compose(b,c)))

    def test_96_shapes_against_homogeneous_products(self):
        for n in range(32):
            for d in (1,2,3):
                inp=case(n,d,33*n+d);r=dense.scan(inp['pairs'],inp['initial']);self.assertEqual(r['states'],expansion(inp['pairs'],inp['initial']));self.assertEqual(r['states'],dense.sequential(inp['pairs'],inp['initial']))

    def test_full_dag_operand_scope_pair_and_depth(self):
        for n in range(33):
            inp=case(n,2,3+n);r=dense.scan(inp['pairs'],inp['initial']);ps,h=dense.prepare(inp['pairs'],inp['initial']);nodes={'i'+str(i):{'start':i,'end':i+1,'depth':0,'pair':p}for i,p in enumerate(ps)}
            for node in r['composition_trace']:
                l=nodes[node['left']];q=nodes[node['right']];self.assertEqual(l['end'],q['start']);self.assertEqual((node['start'],node['end']),(l['start'],q['end']));self.assertEqual(node['depth'],max(l['depth'],q['depth'])+1)
                # Independent augmented product for this exact interval.
                transform=homogeneous({'matrix':dense.eye(2),'write':[F(0)]*2})
                for a,b in ps[node['start']:node['end']]:transform=matmul(homogeneous({'matrix':a,'write':b}),transform)
                self.assertEqual(node['pair'][0],tuple(tuple(row[:-1])for row in transform[:-1]));self.assertEqual(node['pair'][1],tuple(row[-1]for row in transform[:-1]));nodes[node['id']]=node
            for i,p in enumerate(r['prefixes']):self.assertEqual(p['start'],0);self.assertEqual(p['end'],i+1);self.assertEqual(p['pair'],nodes[p['id']]['pair'])

    def test_scalar_counters_against_instrumented_arithmetic(self):
        for n in range(17):
            for d in (1,2,4):
                inp=case(n,d,91+n+d);ps,h=dense.prepare(inp['pairs'],inp['initial']);ps=[(tuple(tuple(Count(x)for x in row)for row in a),tuple(Count(x)for x in b))for a,b in ps];h=tuple(Count(x)for x in h);Count.mult=Count.adds=0;r=dense._scan(ps,h);c=r['counts'];self.assertEqual((Count.mult,Count.adds),(c['scalar_multiplications'],c['scalar_additions']))
                self.assertEqual(c['pair_compositions'],len(r['composition_trace']));self.assertLess(c['pair_compositions'],max(1,2*n))

    def test_672_parameter_coordinates_against_full_polynomials(self):
        checked=0
        for n in range(6):
            for d in (1,2,3):
                for seed in range(2):
                    inp=case(n,d,100*n+10*d+seed);r=dense.audit(**inp);p=d+n*(d*d+d)
                    for k in range(p):
                        basis=layout([F(i==k)for i in range(p)],n,d);_,loss=polynomial(inp,basis);self.assertEqual(flat(r['vjp'])[k],loss[1]if len(loss)>1 else 0);checked+=1
        self.assertEqual(checked,672)

    def test_24_directional_polynomial_models(self):
        for n in range(8):
            for d in (1,2,3):
                inp=case(n,d,19*n+d);r=dense.audit(**inp);states,loss=polynomial(inp,inp['direction']);self.assertEqual(r['state_jvp'],[tuple(x[1]if len(x)>1 else 0 for x in row)for row in states]);self.assertEqual(r['transpose']['cotangent_dot_jvp'],loss[1]if len(loss)>1 else 0);self.assertEqual(r['transpose']['exact_gap'],0)

    def test_diagonal_restriction_matches_original(self):
        for n in range(9):
            for d in (1,2,4):
                inp=case(n,d,79+n+d);pairs=[];diag=[]
                for p in inp['pairs']:
                    a=[p['matrix'][i][i]for i in range(d)];pairs.append({'matrix':[[a[i]if i==j else F(0)for j in range(d)]for i in range(d)],'write':p['write']});diag.append({'decay':a,'write':p['write']})
                self.assertEqual(dense.scan(pairs,inp['initial'])['states'],diagonal_scan(diag,inp['initial'])['states'])
                ps,h=dense.prepare(pairs,inp['initial']);self.assertEqual(dense._scan(ps,h,wrong_order=True)['states'],dense._scan(ps,h)['states'])

    def test_reverse_transpose_and_shift_witness(self):
        inp=dense.demo()['input'];r=dense.audit(**inp);n=len(inp['pairs'])
        for i,(a,b)in enumerate(r['witness']['reverse_maps']):
            t=n-i-1;self.assertEqual(r['witness']['reverse_original_times'][i],t);self.assertEqual(a,dense.transpose(tuple(tuple(F(x)for x in row)for row in inp['pairs'][t+1]['matrix']))if t+1<n else dense.zero(2));self.assertEqual(b,tuple(F(x)for x in inp['cotangents'][t]));self.assertEqual(r['witness']['reverse']['states'][i],r['adjoints'][t])

    def test_empty_singular_maximum_and_large_computed_values(self):
        for n,d in ((0,4),(1,4),(32,4)):
            inp=case(n,d,88);r=dense.audit(**inp);self.assertEqual(r['transpose']['exact_gap'],0)
        inp=case(4,2,4);inp['pairs'][1]['matrix']=[[0,0],[0,0]];a=dense.audit(**inp);inp['initial']=[99,-81];b=dense.audit(**inp);self.assertEqual(a['states'][1:],b['states'][1:])
        inp=case(3,1,5);inp['pairs']=[{'matrix':[[1<<100]],'write':[0]}]*3;inp['initial']=[1<<100];r=dense.audit(**inp);self.assertGreater(r['states'][-1][0].numerator.bit_length(),128)

    def test_bad_shapes_and_bounds(self):
        inp=dense.demo()['input'];changes=[('pairs',inp['pairs']*6),('initial',[0]*5),('initial',[1<<129,0]),('cotangents',[]),('cotangents',[[True,0]]*6),('direction',{}),('direction',{'initial':[0],'pairs':[]}),('pairs',[{'matrix':[[1,2]],'write':[1,2]}]),('pairs',[{'matrix':[[1,2],[3,4]],'write':[True,1]}])]
        for k,v in changes:
            with self.subTest(k=k):
                with self.assertRaises(ValueError):dense.audit(**{**inp,k:v})

    def test_input_rng_and_invisible_controls(self):
        inp=case(6,2,7);before=copy.deepcopy(inp);rng=random.getstate();dense.audit(**inp);self.assertEqual(inp,before);self.assertEqual(random.getstate(),rng)
        inp['cotangents']=[[0,0]]*6;self.assertFalse(dense.audit(**inp)['controls']['missing_transpose']['detected_for_this_input'])
        inp=case(6,1,4);self.assertFalse(dense.audit(**inp)['controls']['missing_transpose']['detected_for_this_input'])

    def test_saved_report_and_two_cli_modes(self):
        root=Path(__file__).resolve().parents[1];report=json.dumps(jsonify(dense.demo()),sort_keys=True,indent=2)+'\n';self.assertEqual((root/'examples/dense-demo.json').read_text(),report)
        for mode,path,expected in [('dense-demo',None,report),('dense','examples/dense-input.json',json.dumps(jsonify(dense.demo()['audit']),sort_keys=True,indent=2)+'\n')]:
            p=subprocess.run([sys.executable,'-m','scanlab',mode]+([path]if path else[]),cwd=root,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(p.stdout,expected)

    def test_cli_invalid_input_is_atomic(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'in.json'
            for text in ('{}','{"pairs":[],"pairs":[]}',json.dumps({**jsonify(dense.demo()['input']),'targets':[]}),json.dumps(jsonify(dense.demo()['input'])).replace('"1/3"','NaN',1),'x'*1_000_001):
                p.write_text(text);r=subprocess.run([sys.executable,'-m','scanlab','dense',str(p)],cwd=root,capture_output=True,text=True);self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'');self.assertIn('error:',r.stderr)

if __name__=='__main__':unittest.main()
