import sys
import unittest
from tim_reasoning import RecipeTagger


class TestRecipeTagger(unittest.TestCase):
    MODEL_TAGGER_PATH = '/Users/rlopez/PTG/experiments/models/recipe_tagger'

    @classmethod
    def setUpClass(cls):
        cls.recipe_tagger = RecipeTagger(cls.MODEL_TAGGER_PATH)

    def test_predict_entities(self):
        recipe_text = '25 grams whole coffee beans.'
        expected_tokens = ['25', 'grams', 'whole', 'coffee beans', '.']
        expected_tags = ['QUANTITY', 'UNIT', 'O', 'INGREDIENT', 'O']
        actual_tokens, actual_tags = self.recipe_tagger.predict_entities(recipe_text)
        self.assertEqual(actual_tags, expected_tags)
        self.assertEqual(actual_tokens, expected_tokens)

        recipe_text = 'Clean the knife by wiping with a paper towel.'
        expected_tokens = ['Clean', 'the', 'knife', 'by', 'wiping', 'with a', 'paper towel', '.']
        expected_tags = ['ACTION', 'O', 'TOOL', 'O', 'ACTION', 'O', 'TOOL', 'O']
        actual_tokens, actual_tags = self.recipe_tagger.predict_entities(recipe_text)
        self.assertEqual(actual_tags, expected_tags)
        self.assertEqual(actual_tokens, expected_tokens)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        TestRecipeTagger.MODEL_TAGGER_PATH = sys.argv.pop()
        
    unittest.main()
