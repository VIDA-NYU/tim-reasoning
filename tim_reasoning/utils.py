import json
from os import listdir
from os.path import join, dirname
from Levenshtein import distance as levenshtein_distance

RECIPES_PATH = join(dirname(__file__), 'resource', 'mit_recipes')


def load_recipe_entity_labels(recipe_name):
    recipe_files = [f for f in listdir(RECIPES_PATH) if f.endswith('.json')]

    for recipe_file in recipe_files:
        with open(join(RECIPES_PATH, recipe_file)) as fin:
            recipe_data = json.load(fin)
            if recipe_data['name'] == recipe_name:
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