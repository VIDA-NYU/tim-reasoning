import json
import pandas as pd
import numpy as np
from os.path import join, dirname

RECIPES_PATH = join(dirname(__file__), 'resource', 'mit_recipes')
ANNOTATED_VIDEOS_PATH = join(dirname(__file__), 'resource', 'annotated_videos')


def load_recipe_entity_labels(recipe_id):
    with open(join(RECIPES_PATH, f'recipe_{recipe_id}.json')) as fin:
        recipe_data = json.load(fin)

        recipe_object_labels = {'ingredients': list(recipe_data['ingredient_objects'].keys()),
                                'tools': list(recipe_data['tool_objects'].keys())}

        return recipe_object_labels


def has_common_words(word1, word2):
    words1 = set(word1.split())
    words2 = set(word2.split())
    common = words1 & words2

    if len(common) > 0:
        return True
    else:
        return False


def create_matrix(recipe_id, exclude=None):
    annotations = pd.read_csv(join(ANNOTATED_VIDEOS_PATH, f'recipe_{recipe_id}.csv'), keep_default_na=False)
    annotations = annotations[annotations['video_id'] != exclude]  # For testing
    annotations = annotations[annotations['step_id'] != '']
    no_action_label = 'no action'
    annotations['narration'] = annotations['narration'].replace([''], no_action_label)
    unique_steps = {s: i for i, s in enumerate(annotations['step_id'].unique())}
    unique_actions = {a: i for i, a in enumerate(annotations['narration'].unique())}

    if no_action_label not in unique_actions:
        unique_actions[no_action_label] = len(unique_actions)

    matrix = np.zeros((len(unique_steps), len(unique_actions)))

    for _, row in annotations.iterrows():
        action_index = unique_actions[row['narration']]
        step_index = unique_steps[row['step_id']]
        duration = row['stop_sec'] - row['start_sec']
        matrix[step_index][action_index] += duration

    step_times = matrix.sum(axis=1)
    matrix = matrix / step_times[:, np.newaxis]

    # There is no way to know how many recipes there are in the train data, a video can contain multiple recipes
    num_train_videos = {'pinwheels': 6, 'coffee': 6, 'mugcake': 4, 'tourniquet': 4}

    return {'indexes': unique_actions, 'matrix': matrix, 'step_times': step_times/num_train_videos[recipe_id]}
