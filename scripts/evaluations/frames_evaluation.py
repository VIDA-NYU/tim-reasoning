import os
import logging
import ptgctl
import ptgctl.util
import numpy as np
from os.path import join
from tim_reasoning import StateManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)
#ptgctl.log.setLevel('WARNING')

RECIPE_SID = 'event:recipe:id'
SESSION_SID = 'event:session:id'
UPDATE_STEP_SID = 'event:recipe:step'
ACTIONS_CLIP_SID = 'clip:action:steps'
ACTIONS_EGOVLP_SID = 'egovlp:action:steps'
OBJECTS_SID = 'detic:image:v2'
REASONING_STATUS_SID = 'reasoning:check_status'
REASONING_ENTITIES_SID = 'reasoning:entities'

PERCEPTION_OUTPUTS_PATH = '/Users/rlopez/PTG/experiments/datasets/NYU_PTG/perception_outputs'
CONFIGS = {'tagger_model_path': join(os.environ['REASONING_MODELS_PATH'], 'recipe_tagger'),
           'bert_classifier_path': join(os.environ['REASONING_MODELS_PATH'], 'bert_classifier')}


class ReasoningApp:

    def __init__(self):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'reasoning',
                              password=os.getenv('API_PASS') or 'reasoning')

        self.state_manager = StateManager(CONFIGS)

    def start_recipe(self, recipe_id):
        logger.info(f'Starting recipe, ID={str(recipe_id)}')
        if recipe_id is not None:
            recipe = self.api.recipes.get(recipe_id)
            logger.info(f'Loaded recipe: {str(recipe)}')
            step_data = self.state_manager.start_recipe(recipe)
            logger.info(f'First step: {str(step_data)}')

            return step_data

    def run_reasoning(self):
        video_id = '2022.07.26-20.35.03'
        recipe_id = 'pinwheels'
        perception_actions = np.load(join(PERCEPTION_OUTPUTS_PATH, 'matrix', video_id + '.npy'))
        perception_indexes = np.load(join(PERCEPTION_OUTPUTS_PATH, 'classes', recipe_id, 'classes.npy'))
        perception_actions = [list(zip(perception_indexes, y_i)) for y_i in perception_actions]
        self.start_recipe(recipe_id)

        for detected_actions in perception_actions[:3]:
            detected_actions = sorted(detected_actions, key=lambda x: x[1], reverse=True)
            logger.info(f'Perception actions: {str(detected_actions)}')
            recipe_status = self.state_manager.check_status(detected_actions, [])
            step = str(recipe_status['step_id'] + 1)
            print(step)
            #logger.info(f'Reasoning outputs: {str(recipe_status)}')

    def run(self):
        self.run_reasoning()


if __name__ == '__main__':
    import fire
    fire.Fire(ReasoningApp)
