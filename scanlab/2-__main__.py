import argparse,json,sys
from pathlib import Path
from .exact import audit,jsonify
from .demo import demo

def main(argv=None):
    p=argparse.ArgumentParser(description='Exact affine scan and small gated state-space audit');p.add_argument('mode',choices=('demo','audit'));p.add_argument('input',nargs='?');a=p.parse_args(argv)
    try:
        if a.mode=='demo':
            if a.input:raise ValueError('demo takes no input')
            r=demo()
        else:
            if not a.input:raise ValueError('audit needs input JSON')
            raw=Path(a.input).read_bytes()
            if len(raw)>1_000_000:raise ValueError('input exceeds1MB')
            def pairs(xs):
                d={}
                for k,v in xs:
                    if k in d:raise ValueError('duplicate key')
                    d[k]=v
                return d
            data=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
            if not isinstance(data,dict)or set(data)!={'pairs','initial'}:raise ValueError('input needs exactly pairs and initial')
            r=audit(data['pairs'],data['initial'])
        print(json.dumps(jsonify(r),sort_keys=True,indent=2));return 0
    except (ValueError,TypeError,OSError,UnicodeError)as e:print('error: '+str(e),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
