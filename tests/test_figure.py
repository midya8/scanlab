import base64,copy,importlib.util,json,unittest,zlib
from pathlib import Path
from xml.etree import ElementTree as ET
from scanlab.demo import demo
from scanlab.exact import jsonify
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('render_demo',ROOT/'examples/render_demo.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
class FigureTests(unittest.TestCase):
    def test_exact_svg_and_full_witness_metadata(self):
        report=demo();s=r.render(report);self.assertEqual(s,(ROOT/'assets/scan-audit.svg').read_text());meta=ET.fromstring(s).find('{http://www.w3.org/2000/svg}metadata');self.assertEqual(json.loads(zlib.decompress(base64.b64decode(meta.text))),jsonify(report))
    def test_every_source_bound_cell(self):
        v=r.values(jsonify(demo()));seen={}
        for node in ET.fromstring(r.render(demo())).iter():
            if node.attrib.get('id','').startswith('cell-'):
                k=node.attrib['id'][5:];self.assertNotIn(k,seen);seen[k]=node.text;self.assertEqual(node.text,str(v[k]));self.assertEqual(node.attrib['data-value'],str(v[k]))
        self.assertEqual(set(v),set(seen));self.assertEqual(len(v),114)
    def test_hidden_affine_write_tampering_changes_metadata(self):
        report=demo();bad=copy.deepcopy(report);bad['exact_affine']['scan']['composition_trace'][0]['pair']=(bad['exact_affine']['scan']['composition_trace'][0]['pair'][0],(999,999));self.assertEqual(r.values(jsonify(report)),r.values(jsonify(bad)));self.assertNotEqual(r.render(report),r.render(bad))
