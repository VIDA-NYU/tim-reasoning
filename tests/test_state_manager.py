import sys
import unittest
from tim_reasoning import StateManager


class TestStateManager(unittest.TestCase):
    CONFIGS = None

    @classmethod
    def setUpClass(cls):
        recipe = {
                  "title": "Pinwheels",
                  "ingredients": [
                    "1 8-inch flour tortilla",
                    "Jar of nut butter or allergy-friendly alternative (such as sunbutter, soy butter, or seed butter)",
                    "Jar of jelly, jam, or fruit preserves"
                  ],
                  "tools": [
                    "cutting board",
                    "butter knife",
                    "paper towel",
                    "toothpicks",
                    "~12-inch strand of dental floss",
                    "plate"
                  ],
                  "steps": [
                    "Place tortilla on cutting board.",
                    "Use a butter knife to scoop nut butter from the jar. Spread nut butter onto tortilla, leaving 1/2-inch uncovered at the edges.",
                    "Clean the knife by wiping with a paper towel.",
                    "Use the knife to scoop jelly from the jar. Spread jelly over the nut butter.",
                    "Clean the knife by wiping with a paper towel.",
                    "Roll the tortilla from one end to the other into a log shape, about 1.5 inches thick. Roll it tight enough to prevent gaps, but not so tight that the filling leaks.",
                    "Secure the rolled tortilla by inserting 5 toothpicks about 1 inch apart.",
                    "Trim the ends of the tortilla roll with the butter knife, leaving 1⁄2 inch margin between the last toothpick and the end of the roll. Discard ends.",
                    "Slide floss under the tortilla, perpendicular to the length of the roll. Place the floss halfway between two toothpicks.",
                    "Cross the two ends of the floss over the top of the tortilla roll. Holding one end of the floss in each hand, pull the floss ends in opposite directions to slice.",
                    "Continue slicing with floss to create 5 pinwheels.",
                    "Place the pinwheels on a plate."
                  ]
            }

        cls.state_manager = StateManager(recipe, cls.CONFIGS)

    def test_start_steps(self):
        expected_step_id = 0
        expected_step_description = 'Place tortilla on cutting board.'
        step_data = self.state_manager.start_steps()
        actual_step_description = step_data['step_description']
        self.assertEqual(actual_step_description, expected_step_description)
        actual_step_id = step_data['step_id']
        self.assertEqual(actual_step_id, expected_step_id)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        configs = {'rule_classifier_path': sys.argv.pop(), 'bert_classifier_path': sys.argv.pop()}
    else:
        configs = {'rule_classifier_path': '/Users/rlopez/PTG/experiments/models/recipe_tagger',
                   'bert_classifier_path': '/Users/rlopez/PTG/experiments/models/bert_classifier/'}

    TestStateManager.CONFIGS = configs
        
    unittest.main()
