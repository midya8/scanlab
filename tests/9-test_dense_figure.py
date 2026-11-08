import base64,importlib.util,json,unittest,zlib
from pathlib import Path
from xml.etree import ElementTree as ET
from scanlab.dense import demo
from scanlab.exact import jsonify
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('render_dense',ROOT/'examples/render_dense.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class DenseFigure(unittest.TestCase):
    def test_all_value_cells_and_byte_regeneration(self):
        r=demo();text=(ROOT/'assets/dense-audit.svg').read_text();self.assertEqual(text,mod.render(r));tree=ET.fromstring(text);cells=[(n.attrib['data-key'],n.text)for n in tree.iter()if'data-key'in n.attrib];self.assertEqual(sorted(cells),sorted(mod.cells(r)));self.assertEqual(len(cells),88);self.assertEqual(len(dict(cells)),88)
    def test_complete_witness_metadata(self):
        tree=ET.fromstring(mod.render(demo()));node=next(n for n in tree.iter()if n.attrib.get('id')=='full-witness');self.assertEqual(json.loads(zlib.decompress(base64.b64decode(node.text))),jsonify(demo()))
    def test_undisplayed_reverse_map_changes_metadata(self):
        r=demo();before=mod.render(r);r['audit']['witness']['reverse_original_times'][0]=99;self.assertEqual(mod.cells(r),mod.cells(demo()));self.assertNotEqual(before,mod.render(r))
