import sys
import unittest
from tim_reasoning import BertClassifier


class TestBertClassifier(unittest.TestCase):
    MODEL_PATH = None

    @classmethod
    def setUpClass(cls):
        cls.classifier = BertClassifier(cls.MODEL_PATH)

    def test_is_mistake(self):
        recipe_step = 'Place tortilla on cutting board.'
        detected_action = 'Grab tortilla from plate'
        expected = False
        actual = self.classifier.is_mistake(recipe_step, detected_action)
        self.assertEqual(actual, expected)

        recipe_step = 'Clean the knife by wiping with a paper towel.'
        detected_action = 'Grab tortilla from plate'
        expected = True
        actual = self.classifier.is_mistake(recipe_step, detected_action)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        TestBertClassifier.MODEL_PATH = sys.argv.pop()
    else:
        TestBertClassifier.MODEL_PATH = '/Users/rlopez/PTG/experiments/models/bert_classifier/'

    unittest.main()
