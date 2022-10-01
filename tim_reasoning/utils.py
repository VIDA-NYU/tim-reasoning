import json
from os.path import join, dirname

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

    for entity_type in entity_types:
        new_names = []
        for entity_name in detected_entities[entity_type]:
            if entity_name in entity_labels[entity_type]:
                new_names.append(entity_name)
                continue

        detected_entities[entity_type] = new_names

    return detected_entities
