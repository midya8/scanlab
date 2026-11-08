import base64,copy,gzip,importlib.util,json,unittest
from pathlib import Path
from xml.etree import ElementTree as ET
from scanlab.adjoint import demo
from scanlab.exact import jsonify
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('render_adjoint',ROOT/'examples/render_adjoint.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class AdjointFigure(unittest.TestCase):
    def test_all_numeric_cells_are_bound(self):
        r=demo();text=(ROOT/'assets/adjoint-audit.svg').read_text();self.assertEqual(text,mod.render(r));tree=ET.fromstring(text)
        cells=[(n.attrib['data-key'],n.text)for n in tree.iter()if'data-key'in n.attrib];self.assertEqual(sorted(cells),sorted(mod.cells(r)));self.assertEqual(len(cells),68);self.assertEqual(len(dict(cells)),68)
    def test_complete_hidden_witness(self):
        tree=ET.fromstring(mod.render(demo()));node=next(n for n in tree.iter()if n.attrib.get('id')=='full-witness')
        self.assertEqual(json.loads(gzip.decompress(base64.b64decode(node.text))),jsonify(demo()))
    def test_hidden_trace_mutation_changes_figure(self):
        r=demo();s=mod.render(r);r['audit']['witness']['hvp_reverse']['composition_trace'][0]['left']='fault';self.assertNotEqual(s,mod.render(r));self.assertEqual(mod.cells(r),mod.cells(demo()))
