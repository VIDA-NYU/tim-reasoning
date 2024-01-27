import json
import pandas as pd
import numpy as np
from os.path import join, dirname
from Levenshtein import distance as levenshtein_distance

RECIPES_PATH = join(dirname(__file__), 'resource', 'mit_recipes')
ANNOTATED_VIDEOS_PATH = join(dirname(__file__), 'resource', 'annotated_videos')


def load_recipe_entity_labels(recipe_id):
    with open(join(RECIPES_PATH, f'recipe_{recipe_id}.json')) as fin:
        recipe_data = json.load(fin)
        if 'ingredient_objects' in recipe_data:
            recipe_object_labels = {'ingredients': list(recipe_data['ingredient_objects'].keys()),
                                    'tools': list(recipe_data['tool_objects'].keys())}
        else:
            recipe_object_labels = {'ingredients': recipe_data.get('ingredients',[]), 'tools': recipe_data.get('tools', [])}

        return recipe_object_labels


def map_entity_labels(entity_labels, detected_entities):
    entity_types = entity_labels.keys()
    new_names = {}

    for entity_type in entity_types:
        new_names_tmp = set()
        for detected_entity in detected_entities[entity_type]:
            if detected_entity in entity_labels[entity_type]:
                new_names_tmp.add(detected_entity)
            else:
                min_distance = float('inf')
                best_label = None
                for entity_label in entity_labels[entity_type]:
                    # if detected_entity in entity_label or entity_label in detected_entity:
                    if has_common_words(detected_entity, entity_label):
                        distance = levenshtein_distance(detected_entity, entity_label)
                        if distance < min_distance:
                            min_distance = distance
                            best_label = entity_label

                if best_label is not None:
                    new_names_tmp.add(best_label)
        new_names[entity_type] = list(new_names_tmp)

    return new_names


def has_common_words(word1, word2):
    words1 = set(word1.split())
    words2 = set(word2.split())
    common = words1 & words2

    if len(common) > 0:
        return True
    else:
        return False


def create_matrix(recipe_id, exclude=None):
    try:
        annotations = pd.read_csv(join(ANNOTATED_VIDEOS_PATH, f'recipe_{recipe_id}.csv'), keep_default_na=False)
    except Exception:
        return {'indexes': {}, 'matrix': np.zeros((0,0)), 'step_times': 0}
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
    num_train_videos = {'pinwheels': 6, 'coffee': 6, 'mugcake': 4, 'tourniquet': 4, 'm5': 1}

    return {'indexes': unique_actions, 'matrix': matrix, 'step_times': step_times/num_train_videos[recipe_id]}
