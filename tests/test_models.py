import unittest

from exceptions import ModelNotFoundError
from main import build_system
from models import GPTModel, ModelType


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.manager = build_system()

    def test_smallest_sufficient_active_model(self):
        selected = self.manager.select_model("openai", 4096)
        self.assertEqual(selected.model_name, "GPT-Model")
        self.assertEqual(self.manager.select_model("OpenAI", 4097).model_name, "GPT-Large")
        selected.is_active = False
        self.assertEqual(self.manager.select_model("OpenAI", 4096).model_name, "GPT-Large")

    def test_invalid_token_count(self):
        for count in (0, -1, 2.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.manager.select_model("OpenAI", count)

    def test_embedding_capacity(self):
        self.assertEqual(
            self.manager.select_model("OpenAI", 8191, ModelType.EMBEDDING).model_name,
            "Embed-Model",
        )
        with self.assertRaises(ModelNotFoundError):
            self.manager.select_model("OpenAI", 8192, ModelType.EMBEDDING)


class MaskedKeyTests(unittest.TestCase):
    def test_masking(self):
        for key, expected in [("x", "*"), ("xy", "**"), ("example", "ex*****")]:
            with self.subTest(key_length=len(key)):
                model = GPTModel("Demo", "Local", 1024, 0.7, key)
                self.assertEqual(model.api_key, expected)
