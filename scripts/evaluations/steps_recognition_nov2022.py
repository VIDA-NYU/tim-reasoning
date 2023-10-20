import os
import logging
import json
import numpy as np
import pandas as pd
from os.path import join
from tim_reasoning import StateManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

RECIPES_PATH = '/Users/rlopez/PTG/tim-reasoning/tim_reasoning/resource/mit_recipes'
PERCEPTION_OUTPUTS_PATH = '/Users/rlopez/PTG/experiments/datasets/NYU_PTG/perception_outputs'
RESULTS_PATH = '/Users/rlopez/PTG/tim-reasoning/scripts/evaluations/resource'

CONFIGS = {'tagger_model_path': join(os.environ['REASONING_MODELS_PATH'], 'recipe_tagger'),
           'bert_classifier_path': join(os.environ['REASONING_MODELS_PATH'], 'bert_classifier')}


BATCH_SIZE = 20
state_manager = StateManager(CONFIGS)


def evaluate_videos():
    recipe_id = 'pinwheels'
    video_ids = ['2022.07.26-22.21.56', '2022.07.26-20.35.03']
    #recipe_id = 'coffee'
    #video_ids = ['coffee-test-1', 'coffee-test-2']
    #recipe_id = 'mugcake'
    #video_ids = ['mugcake-10.13', 'peterx-mugcake']

    results = {'video': [], 'action': [], 'true_step': [], 'predicted_step': []}

    for video_id in video_ids:
        video_results = evaluate_video(recipe_id, video_id)
        results['video'] += video_results['video']
        results['action'] += video_results['action']
        results['true_step'] += video_results['true_step']
        results['predicted_step'] += video_results['predicted_step']

    results = pd.DataFrame.from_dict(results)
    results.to_csv(join(RESULTS_PATH, f'results_{recipe_id}.csv'), index=False)


def evaluate_video(recipe_id, video_id):
    perception_indexes = np.load(join(PERCEPTION_OUTPUTS_PATH, 'action_names', recipe_id, 'classes.npy'))
    perception_actions = np.load(join(PERCEPTION_OUTPUTS_PATH, 'action_frames', video_id + '.npy'))
    steps_groundtruth = load_groundtruth(recipe_id, video_id)
    #print(f'Total frames: {len(perception_actions)}, Total steps: {len(steps_groundtruth["step"])} Total actions:{len(steps_groundtruth["action"])}')
    results = {'video': [], 'action': [], 'true_step': [], 'predicted_step': []}
    start_recipe(recipe_id)
    batch_size = BATCH_SIZE
    batch_counter = 0
    batch = np.zeros((batch_size, len(perception_indexes)))
    valid_frames = 0
    indexes = []

    for index, detected_actions in enumerate(perception_actions):
        if steps_groundtruth['action'][index] == 'no action':
            continue  # Ignore no action for this evaluation
        valid_frames += 1

        if batch_counter < batch_size:
            batch[batch_counter] = detected_actions
            indexes.append(index)
            batch_counter += 1

            if batch_counter == batch_size:
                detected_actions = batch.mean(0)
                real_index = indexes.pop(0)
                batch = np.delete(batch, 0, axis=0)
                batch = np.append(batch, np.zeros((1, len(perception_indexes))), axis=0)
                batch_counter -= 1
            else:
                continue

        detected_actions = list(zip(perception_indexes, detected_actions))
        detected_actions = sorted(detected_actions, key=lambda x: x[1], reverse=True)
        logger.info(f'Perception actions: {str(detected_actions)}')
        recipe_status = state_manager.check_status(detected_actions, [])
        predicted_step = recipe_status['step_id'] + 1
        results['action'].append(steps_groundtruth['action'][real_index])
        results['true_step'].append(steps_groundtruth['step'][real_index])
        results['predicted_step'].append(predicted_step)
        results['video'].append(video_id)
        print('True:', steps_groundtruth['step'][real_index])
        print('Predicted', predicted_step)

    for index in range(batch_size-1):  # For the remaining final frames
        detected_actions = batch[index]
        real_index = indexes[index]
        detected_actions = list(zip(perception_indexes, detected_actions))
        detected_actions = sorted(detected_actions, key=lambda x: x[1], reverse=True)
        logger.info(f'Perception actions: {str(detected_actions)}')
        recipe_status = state_manager.check_status(detected_actions, [])
        predicted_step = recipe_status['step_id'] + 1
        results['action'].append(steps_groundtruth['action'][real_index])
        results['true_step'].append(steps_groundtruth['step'][real_index])
        results['predicted_step'].append(predicted_step)
        results['video'].append(video_id)
        print('True:', steps_groundtruth['step'][real_index])
        print('Predicted', predicted_step)

    assert valid_frames == len(results['true_step'])
    print(f'Total frames: {len(perception_actions)}, valid frames: {valid_frames}')

    correct_predictions = sum([t == p for t, p in zip(results['true_step'], results['predicted_step'])])
    print('Total Accuracy:', float(correct_predictions)/valid_frames)

    return results


def start_recipe(recipe_id):
    logger.info(f'Starting recipe, ID={str(recipe_id)}')
    if recipe_id is not None:
        with open(join(RECIPES_PATH, f'recipe_{recipe_id}.json')) as fin:
            recipe = json.load(fin)
            logger.info(f'Loaded recipe: {str(recipe)}')
            state_manager.start_recipe(recipe)


def load_groundtruth(recipe_id, video_id):
    action_names = np.load(join(PERCEPTION_OUTPUTS_PATH, 'action_names', recipe_id, 'classes.npy'))
    frame_labels = np.load(join(PERCEPTION_OUTPUTS_PATH, 'frame_labels', video_id + '.npy'))
    frame_labels = [action_names[i] for i in frame_labels]
    step_labels = np.load(join(PERCEPTION_OUTPUTS_PATH, 'step_labels', video_id + '.npy'))
    step_labels = [int(i) for i in step_labels]
    steps_groundtruth = {'action': frame_labels, 'step': step_labels}

    return steps_groundtruth


if __name__ == '__main__':
    evaluate_videos()