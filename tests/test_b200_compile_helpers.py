import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from b200_compile_helpers import DEFAULT_RECIPE, load_recipe, verify_files


class HelperRecipeTests(unittest.TestCase):
    def setUp(self):
        self.recipe = load_recipe(DEFAULT_RECIPE)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_missing_cuda_pin_is_rejected(self):
        recipe = copy.deepcopy(self.recipe)
        recipe["cuda_sources"].pop("bin/nvcc")
        path = self.root / "recipe.json"
        path.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, "inventory is incomplete"):
            load_recipe(path)

    def test_changed_source_is_rejected(self):
        path = self.root / "driver.c"
        path.write_text("int compiled_value = 1;\n")
        sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
        verify_files(self.root, sources)
        path.write_text("int compiled_value = 2;\n")
        with self.assertRaisesRegex(ValueError, "source differs"):
            verify_files(self.root, sources)

    def test_source_cannot_escape_package_root(self):
        with self.assertRaisesRegex(ValueError, "contained relative paths"):
            verify_files(self.root, {"../driver.c": "unused"})


if __name__ == "__main__":
    unittest.main()
