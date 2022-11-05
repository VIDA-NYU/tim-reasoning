import os
import logging
import time
import ptgctl
import ptgctl.util
import numpy as np
import pandas as pd
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
        perception_actions = np.load(join(PERCEPTION_OUTPUTS_PATH, 'action_frames', video_id + '.npy'))
        perception_indexes = np.load(join(PERCEPTION_OUTPUTS_PATH, 'action_names', recipe_id, 'classes.npy'))
        perception_actions = [list(zip(perception_indexes, y_i)) for y_i in perception_actions]
        steps_groundtruth = self.load_groundtruth(recipe_id, video_id)
        results = {'action': [], 'true_step': [], 'predicted_step': []}
        self.start_recipe(recipe_id)

        for index, detected_actions in enumerate(perception_actions):
            detected_actions = sorted(detected_actions, key=lambda x: x[1], reverse=True)
            logger.info(f'Perception actions: {str(detected_actions)}')
            recipe_status = self.state_manager.check_status(detected_actions, [])
            predicted_step = str(recipe_status['step_id'] + 1)
            results['action'].append(steps_groundtruth['action'][index])
            results['true_step'].append(steps_groundtruth['step'][index])
            results['predicted_step'].append(predicted_step)
            print('True:', steps_groundtruth['step'][index])
            print('Predicted', predicted_step)
            #logger.info(f'Reasoning outputs: {str(recipe_status)}')

        count = 0.0
        for true_step, predicted_step in zip(results['true_step'], results['predicted_step']):
            if str(true_step) == predicted_step:
                count += 1

        print('Total Accuracy:', count/len(results['true_step']))

        results = pd.DataFrame.from_dict(results)
        results.to_csv(f'/Users/rlopez/PTG/tim-reasoning/scripts/evaluations/resource/results_{recipe_id}.csv', index=False)

    def load_groundtruth(self, recipe_id, video_id):
        annotations_path = '/Users/rlopez/PTG/tim-reasoning/scripts/evaluations/resource'
        annotations = pd.read_csv(join(annotations_path, f'groundtruth_{recipe_id}.csv'), keep_default_na=False)
        annotations = annotations[annotations['video_id'] == video_id]
        frame_classes = np.load(join(PERCEPTION_OUTPUTS_PATH, 'frame_classes', video_id + '.npy'))
        perception_indexes = np.load(join(PERCEPTION_OUTPUTS_PATH, 'action_names', recipe_id, 'classes.npy'))

        previous_value = None
        index = 0
        current_step = 1  # We assume that user should start always in the first step
        no_action = -1
        id_step = no_action
        steps_groundtruth = {'action': [], 'step': []}

        for frame_class in frame_classes:
            if frame_class != 0:
                if frame_class != previous_value:
                    id_step = annotations.iloc[index]['step_id']
                    id_step = int(float(id_step)) if len(id_step) > 0 else id_step
                    if id_step == '':
                        # If there are no step annotated, it should be the current step
                        id_step = current_step
                    previous_value = frame_class
                    current_step = id_step
                    index += 1
                steps_groundtruth['step'].append(id_step)

            else:
                previous_value = no_action
                #steps_groundtruth['step'].append(no_action)
                # Add the current step instead of no action because for the UI we always output the current step with an error message
                steps_groundtruth['step'].append(current_step)

            steps_groundtruth['action'].append(perception_indexes[frame_class])

        return steps_groundtruth

    def run(self):
        self.run_reasoning()


if __name__ == '__main__':
    import fire
    fire.Fire(ReasoningApp)
