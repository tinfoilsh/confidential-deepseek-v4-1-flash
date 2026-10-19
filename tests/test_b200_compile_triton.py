import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from b200_compile_triton import DEFAULT_RECIPE, build, load_recipe


class TritonRecipeTests(unittest.TestCase):
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

    def test_unpinned_entrypoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "source pin"):
            self.load_changed(lambda recipe: recipe["specializations"][0].update(
                entrypoint="unreviewed.module:kernel"))

    def test_empty_specializations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "inventory is incomplete"):
            self.load_changed(lambda recipe: recipe["specializations"].clear())

    def test_argument_cannot_be_both_runtime_and_constant(self):
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            self.load_changed(lambda recipe: recipe["specializations"][0]["constexprs"].update(
                logits_ptr=0))

    def test_changed_assembler_is_rejected_before_creating_output(self):
        assembler = self.root / "ptxas"
        assembler.write_text("an unrelated assembler")
        output = self.root / "output"
        with self.assertRaisesRegex(ValueError, "assembler differs"):
            build(DEFAULT_RECIPE, output, assembler)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
