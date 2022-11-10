import argparse
import unittest
from tim_reasoning import StateManager


class TestStateManager(unittest.TestCase):
    CONFIGS = None

    @classmethod
    def setUpClass(cls):
        cls.state_manager = StateManager(cls.CONFIGS)
        cls.recipe = {
            "_id": "pinwheels",
            "name": "Pinwheels",
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
            "instructions": [
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

    def test_start_steps(self):
        expected_step_id = 0
        expected_step_description = 'Place tortilla on cutting board.'
        step_data = self.state_manager.start_recipe(self.recipe)
        actual_step_description = step_data['step_description']
        self.assertEqual(actual_step_description, expected_step_description)
        actual_step_id = step_data['step_id']
        self.assertEqual(actual_step_id, expected_step_id)

    def test_get_entities(self):
        self.state_manager.start_recipe(self.recipe)
        expected_entities = {'ingredients': {'tortilla'}, 'tools': {'cutting board'}}
        actual_entities = self.state_manager.get_entities()[0]['step_entities']  # Only for step 1
        self.assertEqual(actual_entities, expected_entities)

    def test_check_status(self):
        self.state_manager.start_recipe(self.recipe)
        detected_actions = [('place tortilla on chopping board', 0.7), ('scoop peanut butter, knife', 0.8), ('spread-on peanut butter, tortilla', 0.8), ('wipe knife', 0.8), ('scoop grape jelly, knife', 0.8)]
        detected_objects = [{'xyxyn': [0.27401072, 0.43337104, 0.37002426, 0.6817775], 'confidence': 0.79647225, 'class_id': 9, 'label': 'Jar of jelly'}, {'xyxyn': [0.16237056, 0.5151739, 0.28028676, 0.7697359], 'confidence': 0.5592432, 'class_id': 8, 'label': 'Jar of peanut butter'}, {'xyxyn': [0.16281287, 0.5172336, 0.25855953, 0.6642615], 'confidence': 0.32714963, 'class_id': 12, 'label': 'jar lid'}]
        self.state_manager.check_status(detected_actions, detected_objects)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tagger')
    parser.add_argument('-b', '--bert')
    args, unknown_args = parser.parse_known_args()

    configs = {'tagger_model_path': '/Users/rlopez/PTG/experiments/models/recipe_tagger',
               'bert_classifier_path': '/Users/rlopez/PTG/experiments/models/bert_classifier'}

    if args.tagger:
        configs['tagger_model_path'] = args.tagger
    if args.bert:
        configs['bert_classifier_path'] = args.bert

    TestStateManager.CONFIGS = configs
    unittest.main(argv=[parser.prog]+unknown_args)  # Send the remaining arguments to unittest
