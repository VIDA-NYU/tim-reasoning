import json
import pandas as pd
from os.path import join, dirname
from Levenshtein import distance as levenshtein_distance

RECIPES_PATH = join(dirname(__file__), 'resource', 'mit_recipes')
ANNOTATED_VIDEOS_PATH = join(dirname(__file__), 'resource', 'annotated_videos')


def load_recipe_entity_labels(recipe_id):
    with open(join(RECIPES_PATH, f'recipe_{recipe_id}.json')) as fin:
        recipe_data = json.load(fin)

        recipe_object_labels = {'ingredients': list(recipe_data['ingredient_objects'].keys()),
                                'tools': list(recipe_data['tool_objects'].keys())}

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


def create_matrix(recipe_id, normalize=True):
    # TODO: Add "no action" label
    annotations = pd.read_csv(join(ANNOTATED_VIDEOS_PATH, f'recipe_{recipe_id}.csv'), keep_default_na=False)
    annotations = annotations[annotations['step_id'] != 'NA']
    annotations['narration'] = annotations['narration'].replace(['NA'], 'no action')
    unique_steps = annotations['step_id'].unique()
    unique_actions = annotations['narration'].unique()
    matrix = {}

    for step in unique_steps:
        matrix[step] = {}
        for action in unique_actions:
            matrix[step][action] = 0

    for index, row in annotations.iterrows():
        action = row['narration']
        step = row['step_id']
        duration = row['stop_sec'] - row['start_sec']
        matrix[step][action] += duration

    if normalize:
        for step, actions in matrix.items():
            total_duration = float(sum([x for x in matrix[step].values()]))
            matrix[step] = {k: v / total_duration for k, v in matrix[step].items()}

    return matrix
