from fractions import Fraction as F
from itertools import product
from pathlib import Path
import copy,json,math,random,subprocess,sys,tempfile,unittest
from scanlab.exact import affine,audit,scan,sequential,jsonify,rational
from scanlab.lti import compare
from scanlab.gated import forward,loss_and_grad,streaming
from scanlab.floating import audit as float_audit
from scanlab.demo import demo


def pairs(n,width):return [{'decay':[F((i+j)%5-2,3)for j in range(width)],'write':[F((2*i+j)%7-3,5)for j in range(width)]}for i in range(n)]


def expanded(ps,h):
    # Independent closed-form expansion, no recurrence and no scan composition.
    return [tuple(h[j]*math.prod(F(ps[k]['decay'][j])for k in range(t+1))+sum(F(ps[k]['write'][j])*math.prod(F(ps[u]['decay'][j])for u in range(k+1,t+1))for k in range(t+1))for j in range(len(h)))for t in range(len(ps))]


class ExactTests(unittest.TestCase):
    def test_729_associative_scalar_triplets(self):
        count=0
        for values in product((-1,0,1),repeat=6):
            a,b,c=[((F(values[i]),),(F(values[i+1]),))for i in (0,2,4)]
            self.assertEqual(affine(affine(a,b),c),affine(a,affine(b,c)));count+=1
        self.assertEqual(count,729)

    def test_noncommutativity_is_not_silently_reversed(self):
        a=((F(2),),(F(1),));b=((F(3),),(F(4),));self.assertEqual(affine(a,b),((F(6),),(F(7),)));self.assertNotEqual(affine(a,b),affine(b,a))

    def test_every_length_zero_to_65_four_widths_against_expansion(self):
        count=0
        for n,width in product(range(66),range(1,5)):
            ps=pairs(n,width);h=[F(i-2,7)for i in range(width)];want=expanded(ps,h)
            self.assertEqual(scan(ps,h)['states'],want);self.assertEqual(sequential(ps,h),want);count+=1
        self.assertEqual(count,264)

    def test_all_dag_nodes_scope_depth_arithmetic_and_prefixes(self):
        for n in range(1,34):
            ps=pairs(n,2);r=scan(ps,[1,-1]);known={'i'+str(i):{'start':i,'end':i+1,'depth':0,'pair':(tuple(x['decay']),tuple(x['write']))}for i,x in enumerate(ps)}
            for row in r['composition_trace']:
                a,b=known[row['left']],known[row['right']];self.assertEqual(a['end'],b['start']);self.assertEqual(row['start'],a['start']);self.assertEqual(row['end'],b['end']);self.assertEqual(row['depth'],1+max(a['depth'],b['depth']));self.assertEqual(row['pair'],affine(a['pair'],b['pair']));known[row['id']]=row
            for i,prefix in enumerate(r['prefix_pairs']):
                node=known[prefix['node']];self.assertEqual((node['start'],node['end']),(0,i+1));self.assertEqual(prefix['composition_depth'],node['depth'])
            c=r['counts'];m=len(r['composition_trace']);self.assertEqual(c['pair_compositions'],m);self.assertEqual(c['scan_scalar_multiplications'],4*m+2*n);self.assertEqual(c['scan_scalar_additions'],2*m+2*n)
            self.assertEqual(c['scan_composition_plus_apply_span'],max(p['composition_depth']+1 for p in r['prefix_pairs']))

    def test_linear_work_and_logarithmic_dependency_bounds(self):
        for n in range(1,129):
            c=scan(pairs(n,1),[0])['counts'];self.assertLess(c['pair_compositions'],2*n);self.assertLessEqual(c['scan_composition_plus_apply_span'],2*math.ceil(math.log2(n))+1)
        for k in range(8):
            n=2**k;c=scan(pairs(n,1),[0])['counts'];self.assertEqual(c['pair_compositions'],2*n-k-2)
        c=scan(pairs(16,2),[0,0])['counts'];self.assertGreater(c['scan_scalar_multiplications'],c['sequential_scalar_multiplications']);self.assertLess(c['scan_composition_plus_apply_span'],c['sequential_update_span'])

    def test_every_exact_chunk_boundary(self):
        ps=pairs(17,3);h=[F(1,2),-1,2];want=sequential(ps,h)
        for split in range(18):
            a=scan(ps[:split],h)['states'];carry=a[-1]if a else h;b=scan(ps[split:],carry)['states'];self.assertEqual(a+b,want)

    def test_identity_zero_decay_and_nonzero_initial(self):
        ps=[{'decay':[1],'write':[0]},{'decay':[0],'write':[2]},{'decay':[1],'write':[-1]}];self.assertEqual(scan(ps,[9])['states'],[(F(9),),(F(2),),(F(1),)])
        self.assertEqual(scan([],[-1])['states'],[]);self.assertEqual(scan([],[-1])['counts']['scan_composition_plus_apply_span'],0)

    def test_exact_input_validation(self):
        for x in (True,.5,float('nan'),'nan','1/0','1e3','1/'+'9'*81,2**128):
            with self.assertRaises(ValueError):rational(x)
        for ps,h in (([],[]),(pairs(129,1),[0]),(pairs(1,2),[0]),([{'decay':[1],'write':[0],'extra':1}],[0]),([{'decay':[.5],'write':[0]}],[0])):
            with self.assertRaises(ValueError):scan(ps,h)

    def test_inputs_and_global_rng_unchanged(self):
        ps=pairs(8,2);h=[1,2];before=copy.deepcopy((ps,h));state=random.getstate();audit(ps,h);self.assertEqual((ps,h),before);self.assertEqual(random.getstate(),state)


class LTITests(unittest.TestCase):
    def test_120_fixed_models_recurrence_convolution_and_initial_term(self):
        count=0
        for width,n in product(range(1,5),range(30)):
            a=[F(j+1,j+3)for j in range(width)];b=[j-1 for j in range(width)];c=[F(j+2,5)for j in range(width)];h=[F(1,j+1)for j in range(width)];xs=[i%3-1 for i in range(n)]
            r=compare(a,b,c,'1/3',xs,h);self.assertEqual(r['recurrent_outputs'],r['convolution_outputs']);self.assertEqual(r['direct_convolution_input_terms'],n*(n+1)//2);count+=1
        self.assertEqual(count,120)

    def test_missing_nonzero_boundary_term_is_a_real_fault(self):
        r=compare(['1/2'],[1],[1],0,[0,0,0],[2]);self.assertEqual(r['convolution_outputs'],[F(1),F(1,2),F(1,4)]);self.assertNotEqual(r['convolution_outputs'],[0,0,0])

    def test_impulse_skip_and_superposition(self):
        def call(xs):return compare(['1/2','1/3'],[1,-1],[2,1],'1/7',xs,[0,0])['convolution_outputs']
        x=[1,0,-1,2];y=[0,1,1,-1];self.assertEqual(call([a+b for a,b in zip(x,y)]),[a+b for a,b in zip(call(x),call(y))])
        r=compare(['1/2'],[2],[3],1,[1,0,0,0],[0]);self.assertEqual(r['convolution_outputs'],[F(7),F(3),F(3,2),F(3,4)])

    def test_lti_bounds(self):
        for args in (([1],[1,2],[1],0,[1],[0]),([1],[1],[1],0,[float('inf')],[0]),([1],[1],[1],0,[1]*129,[0])):
            with self.assertRaises(ValueError):compare(*args)


class GatedTests(unittest.TestCase):
    def test_all_123_parameter_input_and_initial_derivatives(self):
        count=0;eps=1e-6
        for width,n in product((1,2,3),(1,2,5)):
            p={'theta':[.2*(j-1)for j in range(width)],'u':[.3*(j+1)for j in range(width)],'B':[.2-.1*j for j in range(width)],'C':[.4+.1*j for j in range(width)],'D':.15};xs=[.2*(i-2)for i in range(n)];h=[.1*(j+1)for j in range(width)];ys=[.1*(i%2)for i in range(n)];r=loss_and_grad(p,xs,h,ys)
            def loss(pp,xx,hh):return sum((a-b)**2 for a,b in zip(forward(pp,xx,hh)['outputs'],ys))/(2*n)
            probes=[]
            for key in ('theta','u','B','C'):
                for j in range(width):
                    up=copy.deepcopy(p);dn=copy.deepcopy(p);up[key][j]+=eps;dn[key][j]-=eps;probes.append(((loss(up,xs,h)-loss(dn,xs,h))/(2*eps),r['parameter_gradients'][key][j]))
            up=copy.deepcopy(p);dn=copy.deepcopy(p);up['D']+=eps;dn['D']-=eps;probes.append(((loss(up,xs,h)-loss(dn,xs,h))/(2*eps),r['parameter_gradients']['D']))
            for j in range(n):
                up=xs[:];dn=xs[:];up[j]+=eps;dn[j]-=eps;probes.append(((loss(p,up,h)-loss(p,dn,h))/(2*eps),r['input_gradients'][j]))
            for j in range(width):
                up=h[:];dn=h[:];up[j]+=eps;dn[j]-=eps;probes.append(((loss(p,xs,up)-loss(p,xs,dn))/(2*eps),r['initial_gradients'][j]))
            for numerical,analytic in probes:self.assertLess(abs(numerical-analytic),1e-7+1e-5*abs(numerical));count+=1
        self.assertEqual(count,123)

    def test_demo_finite_difference_audit(self):
        r=demo()['gated']['gradient_check'];self.assertEqual(r['count'],16);self.assertLess(r['max_absolute_error'],1e-8)

    def test_all_64_streaming_partitions(self):
        p={'theta':[.1,.2],'u':[.5,-.4],'B':[1,-1],'C':[.4,.6],'D':.1};xs=[.1*i for i in range(7)];h=[.2,-.3];want=forward(p,xs,h);n=0
        for bits in product((0,1),repeat=6):
            sizes=[];size=1
            for bit in bits:
                if bit:sizes.append(size);size=1
                else:size+=1
            sizes.append(size);r=streaming(p,xs,h,sizes);self.assertEqual(r['outputs'],want['outputs']);self.assertEqual(r['final_state'],want['final_state']);n+=1
        self.assertEqual(n,64)

    def test_large_legal_inputs_streaming_carry_is_not_revalidated_as_input(self):
        p={'theta':[100],'u':[0],'B':[100],'C':[100],'D':100};xs=[100]*5;self.assertEqual(streaming(p,xs,[100],[1,1,1,1,1])['outputs'],forward(p,xs,[100])['outputs'])

    def test_causality_and_future_inputs_do_not_change_prefix(self):
        p={'theta':[0],'u':[1],'B':[1],'C':[1],'D':0};self.assertEqual(forward(p,[1,2,3],[0])['outputs'][:2],forward(p,[1,2,-3],[0])['outputs'][:2])

    def test_input_gate_breaks_fixed_linear_convolution(self):
        r=demo()['not_a_fixed_convolution'];self.assertEqual(r['superposition_residual'][0],0);self.assertGreater(abs(r['superposition_residual'][-1]),.1)

    def test_zero_gate_weight_reduces_to_lti_float_reference(self):
        p={'theta':[0],'u':[0],'B':[2],'C':[3],'D':1};xs=[1,0,-1,2];r=forward(p,xs,[1])['outputs'];want=compare(['1/2'],[2],[3],1,xs,[1])['recurrent_outputs'];self.assertEqual(r,list(map(float,want)))

    def test_gradient_descent_step_lowers_fixture_loss(self):
        r=demo()['gated'];p=r['parameters'];g=r['gradient']['parameter_gradients'];new={k:[v-.01*dv for v,dv in zip(p[k],g[k])]for k in ('theta','u','B','C')};new['D']=p['D']-.01*g['D'];self.assertLess(loss_and_grad(new,r['inputs'],r['initial'],r['targets'])['loss'],r['gradient']['loss'])

    def test_gated_shape_values_and_chunk_errors(self):
        p={'theta':[0],'u':[1],'B':[1],'C':[1],'D':0}
        for bad in ({},dict(p,extra=1),dict(p,B=[1,2]),dict(p,D=True),dict(p,D=float('nan')),dict(p,D=101)):
            with self.assertRaises(ValueError):forward(bad,[1],[0])
        for chunks in ([0,1],[2],[True]):
            with self.assertRaises(ValueError):streaming(p,[1],[0],chunks)
        with self.assertRaises(ValueError):loss_and_grad(p,[],[0],[])
        with self.assertRaises(ValueError):loss_and_grad(p,[1],[0],[0,1])

    def test_input_parameter_rng_unchanged(self):
        r=demo()['gated'];p,x,h,y=(r[k]for k in ('parameters','inputs','initial','targets'));before=copy.deepcopy((p,x,h,y));rng=random.getstate();loss_and_grad(p,x,h,y);self.assertEqual((p,x,h,y),before);self.assertEqual(random.getstate(),rng)


class FloatingAndCLITests(unittest.TestCase):
    def test_floating_cancellation_changes_scan_but_exact_scan_does_not(self):
        ps=[{'decay':[1.0],'write':[x]}for x in (1e16,1.,-1e16,1.)];r=float_audit(ps,[0]);self.assertEqual(r['sequential_states'][-1],(1.,));self.assertEqual(r['scan']['states'][-1],(0.,));self.assertEqual(r['max_absolute_difference'],1)
        exact=[{'decay':[1],'write':[x]}for x in (10**16,1,-10**16,1)];self.assertEqual(audit(exact,[0])['scan']['states'][-1],(F(2),))

    def test_floating_refuses_nonfinite_inputs_and_computed_overflow(self):
        for ps,h in (([{'decay':[1],'write':[float('nan')]}],[0]),([{'decay':[1e100],'write':[1e100]}]*10,[1e100])):
            with self.assertRaises(ValueError):float_audit(ps,h)

    def test_demo_matches_saved_output(self):
        want=Path(__file__).resolve().parents[1]/'examples/demo.json';self.assertEqual(json.loads(want.read_text()),jsonify(demo()))

    def test_cli_strict_schema_and_failures(self):
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'input.json'
            def cli(*args):return subprocess.run([sys.executable,'-m','scanlab',*args],capture_output=True,text=True)
            data={'initial':[0],'pairs':[{'decay':['1/2'],'write':[1]}]};path.write_text(json.dumps(data));r=cli('audit',str(path));self.assertEqual(r.returncode,0);self.assertEqual(json.loads(r.stdout),jsonify(audit(data['pairs'],data['initial'])))
            for text in ('{}','{"initial":[],"initial":[]}','NaN',' '*1_000_001,json.dumps(dict(data,extra=1))):
                path.write_text(text);r=cli('audit',str(path));self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'')
            for args in (('audit',),('demo',str(path))):
                r=cli(*args);self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'')

if __name__=='__main__':unittest.main()
