"""Independent bivariate polynomial checks, not finite-difference tolerances."""
import copy,json,os,random,subprocess,sys,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from scanlab.adjoint import demo,derivative_audit,dot_direction
from scanlab.exact import jsonify


def add(p,q):
    r=dict(p)
    for k,v in q.items():r[k]=r.get(k,F(0))+v
    return {k:v for k,v in r.items()if v}


def mul(p,q):
    r={}
    for (i,j),x in p.items():
        for (k,l),y in q.items():
            key=(i+k,j+l);r[key]=r.get(key,F(0))+x*y
    return {k:v for k,v in r.items()if v}


def flatten(v):return list(v['initial'])+[x for p in v['pairs']for key in ('decay','write')for x in p[key]]


def layout(flat,n,d):
    return {'initial':flat[:d],'pairs':[{'decay':flat[d+2*t*d:d+(2*t+1)*d],'write':flat[d+(2*t+1)*d:d+(2*t+2)*d]}for t in range(n)]}


def polynomial(inp,v,u):
    """Fully expand h(theta+e*v+z*u), then its linear and square losses."""
    n=len(inp['pairs']);d=len(inp['initial']);p=flatten(inp);v=flatten(v);u=flatten(u)
    lifts=layout([{(0,0):F(x),(1,0):F(y),(0,1):F(z)}for x,y,z in zip(p,v,u)],n,d)
    h=lifts['initial'];states=[];linear={};square={}
    for t,step in enumerate(lifts['pairs']):
        h=[add(mul(a,x),b)for a,x,b in zip(step['decay'],h,step['write'])];states.append(h)
        for j,x in enumerate(h):
            linear=add(linear,{k:v*F(inp['cotangents'][t][j])for k,v in x.items()})
            res=add(x,{(0,0):-F(inp['targets'][t][j])});square=add(square,{k:v/2 for k,v in mul(res,res).items()})
    return states,linear,square


def case(n,d,seed):
    rng=random.Random(seed)
    def vec():return [F(rng.randrange(-3,4),rng.randrange(1,5))for _ in range(d)]
    def params():return {'pairs':[{'decay':vec(),'write':vec()}for _ in range(n)],'initial':vec()}
    return {**params(),'cotangents':[vec()for _ in range(n)],'direction':params(),'targets':[vec()for _ in range(n)]}


class Adjoint(unittest.TestCase):
    def test_fixture(self):
        r=demo()['audit'];self.assertEqual(r['transpose']['cotangent_dot_jvp'],F(-817,120));self.assertEqual(r['transpose']['exact_gap'],0)
        self.assertEqual(r['wrong_index_control']['transpose_gap'],F(-259,36));self.assertTrue(r['wrong_index_control']['detected_for_this_input'])
        self.assertEqual(r['squared_loss']['value'],F(2473,288));self.assertEqual(r['squared_loss']['directional_second'],F(20591177,414720))

    def test_all_coordinate_derivatives_against_independent_polynomials(self):
        checked=0
        for n in range(6):
            for d in (1,2,3):
                for seed in range(2):
                    inp=case(n,d,100*n+10*d+seed);r=derivative_audit(**inp);v=inp['direction'];p=2*n*d+d
                    for k in range(p):
                        basis=layout([F(i==k)for i in range(p)],n,d)
                        _,lin,sq=polynomial(inp,v,basis)
                        self.assertEqual(flatten(r['vjp'])[k],lin.get((0,1),0))
                        self.assertEqual(flatten(r['squared_loss']['gradient'])[k],sq.get((0,1),0))
                        self.assertEqual(flatten(r['squared_loss']['hessian_vector'])[k],sq.get((1,1),0));checked+=1
        self.assertEqual(checked,432)

    def test_jvp_loss_derivatives_against_full_polynomial(self):
        for n in range(8):
            for d in (1,2,3):
                inp=case(n,d,n*13+d);r=derivative_audit(**inp);zero=layout([F(0)]*(2*n*d+d),n,d)
                states,lin,sq=polynomial(inp,inp['direction'],zero)
                self.assertEqual(r['state_jvp'],[tuple(x.get((1,0),0)for x in row)for row in states])
                self.assertEqual(r['transpose']['cotangent_dot_jvp'],lin.get((1,0),0))
                self.assertEqual(r['squared_loss']['value'],sq.get((0,0),0))
                self.assertEqual(r['squared_loss']['directional_first'],sq.get((1,0),0))
                self.assertEqual(r['squared_loss']['directional_second'],2*sq.get((2,0),0))

    def test_hessian_bilinear_symmetry(self):
        for n in range(8):
            for d in (1,2,3):
                inp=case(n,d,7*n+d);other=case(n,d,1000+n*d)['direction'];r=derivative_audit(**inp)
                rr=derivative_audit(**{**inp,'direction':other})
                self.assertEqual(dot_direction(r['squared_loss']['hessian_vector'],other),dot_direction(rr['squared_loss']['hessian_vector'],inp['direction']))

    def test_adjoint_independent_path_products(self):
        for n in range(17):
            for d in (1,2,4):
                inp=case(n,d,31+n+d);r=derivative_audit(**inp)
                for t in range(n):
                    for j in range(d):
                        prod=F(1);expected=F(0)
                        for k in range(t,n):
                            if k>t:prod*=inp['pairs'][k]['decay'][j]
                            expected+=F(inp['cotangents'][k][j])*prod
                        self.assertEqual(r['vjp']['pairs'][t]['write'][j],expected)

    def test_reverse_witness_index_and_maps(self):
        r=demo()['audit'];inp=demo()['input'];w=r['witness'];n=len(inp['pairs'])
        for i,m in enumerate(w['reverse_index_map']):
            t=n-i-1;self.assertEqual(m,{'scan_index':i,'original_time':t,'propagates_from_time':t+1 if i else None})
        for i,p in enumerate(w['reverse']['prefix_pairs']):
            t=n-1-i
            self.assertEqual(w['reverse']['states'][i],r['vjp']['pairs'][t]['write'])
            self.assertEqual(p['through'],i)
        self.assertEqual(w['forward']['states'],r['states']);self.assertEqual(w['tangent']['states'],r['state_jvp'])

    def test_empty_and_maximum_shape(self):
        for n,d in ((0,8),(1,8),(128,8)):
            inp={'pairs':[{'decay':[1]*d,'write':[0]*d}for _ in range(n)],'initial':[0]*d,'cotangents':[[1]*d for _ in range(n)],
                'direction':{'pairs':[{'decay':[0]*d,'write':[0]*d}for _ in range(n)],'initial':[0]*d},'targets':[[0]*d for _ in range(n)]}
            r=derivative_audit(**inp);self.assertEqual(r['transpose']['exact_gap'],0);self.assertEqual(r['squared_loss']['directional_second'],0)
            self.assertEqual(r['vjp']['initial'],(F(n),)*d)

    def test_computed_values_not_revalidated_as_inputs(self):
        n=4;x=1<<100
        inp={'pairs':[{'decay':[x],'write':[x]}for _ in range(n)],'initial':[x],'cotangents':[[x]for _ in range(n)],
            'direction':{'pairs':[{'decay':[x],'write':[x]}for _ in range(n)],'initial':[x]},'targets':[[0]for _ in range(n)]}
        r=derivative_audit(**inp);self.assertEqual(r['transpose']['exact_gap'],0);self.assertGreater(r['states'][-1][0].numerator.bit_length(),128)

    def test_input_not_mutated_or_global_rng_touched(self):
        inp=case(5,3,4);old=copy.deepcopy(inp);rng=random.getstate();derivative_audit(**inp);self.assertEqual(inp,old);self.assertEqual(random.getstate(),rng)

    def test_bad_shapes_and_types(self):
        original=demo()['input']
        changes=[('cotangents',[]),('cotangents',[[1]]*7),('cotangents',[[True]]*6),('cotangents',[[0.5]]*6),('targets',None),('targets',[[1,2]]*6),('direction',{}),('direction',{'pairs':[],'initial':[1]}),('direction',{'pairs':original['pairs'],'initial':[1,2]}),('initial',[1<<129])]
        for key,value in changes:
            with self.subTest(key=key,value=value):
                with self.assertRaises(ValueError):derivative_audit(**{**original,key:value})

    def test_zero_cotangent_direction_and_invisible_control(self):
        inp=case(4,2,5);inp['cotangents']=[[0,0]]*4;inp['direction']=layout([F(0)]*18,4,2);r=derivative_audit(**inp)
        self.assertEqual(flatten(r['vjp']),[0]*18);self.assertEqual(flatten(r['squared_loss']['hessian_vector']),[0]*18)
        self.assertFalse(r['wrong_index_control']['detected_for_this_input'])

    def test_saved_demo_and_cli(self):
        root=Path(__file__).resolve().parents[1];expected=json.dumps(jsonify(demo()),sort_keys=True,indent=2)+'\n'
        self.assertEqual((root/'examples/adjoint-demo.json').read_text(),expected)
        for mode,path,value in (('adjoint-demo',None,expected),('adjoint','examples/adjoint-input.json',json.dumps(jsonify(demo()['audit']),sort_keys=True,indent=2)+'\n')):
            p=subprocess.run([sys.executable,'-m','scanlab',mode]+([path]if path else[]),cwd=root,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(p.stdout,value)

    def test_cli_rejects_bad_input(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'in.json'
            for text in ('{}','{"pairs":[],"pairs":[]}',json.dumps({**demo()['input'],'unknown':3}),json.dumps(demo()['input']).replace('"2"','NaN',1),'x'*1_000_001):
                path.write_text(text);p=subprocess.run([sys.executable,'-m','scanlab','adjoint',str(path)],cwd=root,capture_output=True,text=True);self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'');self.assertIn('error:',p.stderr)

if __name__=='__main__':unittest.main()
