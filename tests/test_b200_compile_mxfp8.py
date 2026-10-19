import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from b200_bundle import validate_tuning
from b200_compile_mxfp8 import DEFAULT_RECIPE, load_recipe, verify_sources


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.recipe = load_recipe(DEFAULT_RECIPE)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def load_changed(self, change):
        recipe = copy.deepcopy(self.recipe)
        change(recipe)
        path = self.root / "recipe.json"
        path.write_text(json.dumps(recipe))
        return load_recipe(path)

    def test_retained_choices_match_the_qualified_selection(self):
        path = ROOT / "b200/kernels/choices.json"
        build = json.loads((ROOT / "b200/build.json").read_text())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         build["selected_cache_sha256"])
        self.assertEqual(validate_tuning(path, build["measurement_policy"]), 198)

    def test_duplicate_output_specialization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load_changed(lambda recipe: recipe["variants"].append(recipe["variants"][0]))

    def test_missing_source_pin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inventory is incomplete"):
            self.load_changed(lambda recipe: recipe["sources"].pop(next(iter(recipe["sources"]))))

    def test_invalid_split_k_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "split-K"):
            self.load_changed(lambda recipe: recipe["variants"][0].update(
                split_k_slices=2, swap_ab=False))

    def test_boolean_cannot_be_used_as_a_capacity(self):
        with self.assertRaisesRegex(ValueError, "positive integers"):
            self.load_changed(lambda recipe: recipe["variants"][0].update(max_active_clusters=True))

    def test_changed_source_is_rejected_before_compilation(self):
        source = self.root / "kernel.py"
        source.write_text("def kernel(): return 1\n")
        recipe = {"sources": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}}
        verify_sources(recipe, self.root)
        source.write_text("def kernel(): return 2\n")
        with self.assertRaisesRegex(ValueError, "source differs"):
            verify_sources(recipe, self.root)


if __name__ == "__main__":
    unittest.main()
