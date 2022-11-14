import os
import logging
import json
import numpy as np
import pandas as pd
from os.path import join
from tim_reasoning import StateManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

RECIPES_PATH =  '/Users/rlopez/PTG/tim-reasoning/tim_reasoning/resource/mit_recipes'
PERCEPTION_OUTPUTS_PATH = '/Users/rlopez/PTG/experiments/datasets/NYU_PTG/perception_outputs'
ANNOTATED_VIDEOS_PATH = '/Users/rlopez/PTG/tim-reasoning/scripts/evaluations/resource'

CONFIGS = {'tagger_model_path': join(os.environ['REASONING_MODELS_PATH'], 'recipe_tagger'),
           'bert_classifier_path': join(os.environ['REASONING_MODELS_PATH'], 'bert_classifier')}


class ReasoningApp:

    def __init__(self):
        self.state_manager = StateManager(CONFIGS)

    def start_recipe(self, recipe_id):
        logger.info(f'Starting recipe, ID={str(recipe_id)}')
        if recipe_id is not None:
            with open(join(RECIPES_PATH, f'recipe_{recipe_id}.json')) as fin:
                recipe = json.load(fin)
                logger.info(f'Loaded recipe: {str(recipe)}')
                step_data = self.state_manager.start_recipe(recipe)
                logger.info(f'First step: {str(step_data)}')

                return step_data

    def run_reasoning(self):
        video_id = 'coffee-test-1'
        recipe_id = 'coffee'
        perception_actions = np.load(join(PERCEPTION_OUTPUTS_PATH, 'action_frames', video_id + '.npy'))
        perception_indexes = np.load(join(PERCEPTION_OUTPUTS_PATH, 'action_names', recipe_id, 'classes.npy'))
        #perception_actions = [list(zip(perception_indexes, y_i)) for y_i in perception_actions]
        steps_groundtruth = self.load_groundtruth(recipe_id, video_id)
        print(f'Total frames: {len(perception_actions)}, Total steps: {len(steps_groundtruth["step"])} Total actions:{len(steps_groundtruth["action"])}')
        results = {'action': [], 'true_step': [], 'predicted_step': []}
        self.start_recipe(recipe_id)
        batch_size = 20
        counter = 0
        batch = np.zeros((batch_size, len(perception_indexes)))
        valid_frames = 0
        indexes = []

        for index, detected_actions in enumerate(perception_actions):
            if steps_groundtruth['action'][index] == 'no action':
                continue  # Ignore no action for this evaluation
            valid_frames += 1

            if counter < batch_size:
                batch[counter] = detected_actions
                indexes.append(index)
                counter += 1

                if counter == batch_size:
                    detected_actions = batch.mean(0)
                    real_index = indexes.pop(0)
                    batch = np.delete(batch, 0, axis=0)
                    batch = np.append(batch, np.zeros((1, len(perception_indexes))), axis=0)
                    counter -= 1
                else:
                    continue

            detected_actions = list(zip(perception_indexes, detected_actions))
            detected_actions = sorted(detected_actions, key=lambda x: x[1], reverse=True)
            logger.info(f'Perception actions: {str(detected_actions)}')
            recipe_status = self.state_manager.check_status(detected_actions, [])
            predicted_step = str(recipe_status['step_id'] + 1)

            results['action'].append(steps_groundtruth['action'][real_index])
            results['true_step'].append(steps_groundtruth['step'][real_index])
            results['predicted_step'].append(predicted_step)
            print('True:', steps_groundtruth['step'][real_index])
            print('Predicted', predicted_step)

        count = 0.0
        for true_step, predicted_step in zip(results['true_step'], results['predicted_step']):
            if str(true_step) == str(predicted_step):
                count += 1
        print(len(perception_actions), valid_frames, len(results['true_step']))
        print('Total Accuracy:', count/len(results['true_step']))

        results = pd.DataFrame.from_dict(results)
        results.to_csv(f'/Users/rlopez/PTG/tim-reasoning/scripts/evaluations/resource/results_{recipe_id}.csv', index=False)

    def load_groundtruth(self, recipe_id, video_id):
        action_names = np.load(join(PERCEPTION_OUTPUTS_PATH, 'action_names', recipe_id, 'classes.npy'))
        frame_labels = np.load(join(PERCEPTION_OUTPUTS_PATH, 'frame_labels', video_id + '.npy'))
        frame_labels = [action_names[i] for i in frame_labels]
        step_labels = np.load(join(PERCEPTION_OUTPUTS_PATH, 'step_labels', video_id + '.npy'))
        step_labels = [int(i) for i in step_labels]
        steps_groundtruth = {'action': frame_labels, 'step': step_labels}

        return steps_groundtruth

    def run(self):
        self.run_reasoning()


if __name__ == '__main__':
    import fire
    fire.Fire(ReasoningApp)
