import sys
import unittest
from tim_reasoning import RuleBasedClassifier


class TestRuleBasedClassifier(unittest.TestCase):
    MODEL_TAGGER_PATH = None

    @classmethod
    def setUpClass(cls):
        cls.classifier = RuleBasedClassifier(cls.MODEL_TAGGER_PATH)

    def test_is_mistake(self):
        recipe_step = 'Place tortilla on cutting board.'
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
        TestRuleBasedClassifier.MODEL_TAGGER_PATH = sys.argv.pop()
    else:
        TestRuleBasedClassifier.MODEL_TAGGER_PATH = '/Users/rlopez/PTG/experiments/models/recipe_tagger'

    unittest.main()
