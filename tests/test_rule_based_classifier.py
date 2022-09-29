import sys
import unittest
from tim_reasoning import RuleBasedClassifier
from tim_reasoning import RecipeTagger


class TestRuleBasedClassifier(unittest.TestCase):
    TAGGER_MODEL_PATH = None

    @classmethod
    def setUpClass(cls):
        recipe_tagger = RecipeTagger(cls.TAGGER_MODEL_PATH)
        cls.classifier = RuleBasedClassifier(recipe_tagger)

    def test_is_mistake(self):
        recipe_step = 'Put tortilla on cutting board.'
        detected_action = 'Put tortilla'
        expected = False
        actual = self.classifier.is_mistake(recipe_step, detected_action)
        self.assertEqual(actual, expected)

        recipe_step = 'Clean the knife by wiping with a paper towel.'
        detected_action = 'Put tortilla'
        expected = True
        actual = self.classifier.is_mistake(recipe_step, detected_action)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        TestRuleBasedClassifier.TAGGER_MODEL_PATH = sys.argv.pop()
    else:
        TestRuleBasedClassifier.TAGGER_MODEL_PATH = '/Users/rlopez/PTG/experiments/models/recipe_tagger'

    unittest.main()
