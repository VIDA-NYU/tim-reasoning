import json
from os.path import join, dirname
from Levenshtein import distance as levenshtein_distance

RECIPES_PATH = join(dirname(__file__), 'resource', 'mit_recipes')


def load_recipe_entity_labels(recipe_id):
    recipe_path = join(RECIPES_PATH, f'recipe_{recipe_id}.json')

    with open(recipe_path) as fin:
        recipe_data = json.load(fin)
        recipe_object_labels = {'ingredients': list(recipe_data['ingredients_annotated'].keys()),
                                'tools': list(recipe_data['tools_annotated'].keys())}

        return recipe_object_labels


def map_entity_labels(entity_labels, detected_entities):
    entity_types = entity_labels.keys()
    new_names = {}

    for entity_type in entity_types:
        new_names[entity_type] = []
        for detected_entity in detected_entities[entity_type]:
            if detected_entity in entity_labels[entity_type]:
                new_names[entity_type].append(detected_entity)
            else:
                min_distance = float('inf')
                best_label = None
                for entity_label in entity_labels[entity_type]:
                    if detected_entity in entity_label or entity_label in detected_entity:
                        distance = levenshtein_distance(detected_entity, entity_label)
                        if distance < min_distance:
                            min_distance = distance
                            best_label = entity_label

                if best_label is not None:
                    new_names[entity_type].append(best_label)

    return new_names
