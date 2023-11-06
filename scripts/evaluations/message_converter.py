import pandas as pd
from os.path import join, dirname

RESOURCE_PATH = join(dirname(__file__), 'resource')
OBJECTS_OF_INTEREST = {'bowl', 'mug', 'tortilla', 'plate'}


def read_unique_states(target_objects):
    unique_states_df = pd.read_csv(join(RESOURCE_PATH, 'unique_states.csv'))

    unique_states = []
    for _, row in unique_states_df.iterrows():
        if row['object'].strip() not in target_objects:
            continue
        state_id = row['object'].strip() + '_' + row['state'].strip()
        unique_states.append(state_id)

    return unique_states


def read_unique_objects():
    unique_objects_df = pd.read_csv(join(RESOURCE_PATH, 'unique_objects.csv'))

    return [o + '_hoi' for o in unique_objects_df['object'].unique()]


def convert_message(message):
    object_name = message['label']

    if object_name not in OBJECTS_OF_INTEREST:
        return None

    unique_objects = read_unique_objects()
    unique_states = read_unique_states([object_name])
    all_columns = unique_states + unique_objects
    all_columns_indices = {v: i for i, v in enumerate(all_columns)}

    perception_predictions = [0] * len(all_columns)
    states_to_add = {}

    for state, confidence in message['state'].items():
        state_id = object_name + '_' + state
        states_to_add[state_id] = confidence

    for state_id, state_confidence in states_to_add.items():
        index = all_columns_indices[state_id]
        perception_predictions[index] = state_confidence

    hoi_id = object_name + '_hoi'
    hoi_confidence = message['hand_object_interaction']
    index = all_columns_indices[hoi_id]
    perception_predictions[index] = hoi_confidence

    output = {'id': message['id'], 'object_name': object_name, 'states': perception_predictions}

    return output


if __name__ == '__main__':
    message = {
    "pos": [
      -0.2149151724097291,
      -0.4343880843796524,
      -0.6208099189217009
    ],
    "xyxyn": [
      0.1,
      0.1,
      0.2,
      0.2
    ],
    "label": "mug",
    "status": "tracked",
    "id": 3256,
    "last_seen": 1699306850,
    "state": {
        "empty": 0.1,
        "dripper[empty]": 0.2,
        "dripper+filter": 0.039999999999999994,
        "dripper+filter+coffee[wet]": 0.3,
        "dripper+filter+coffee[drained]": 0.4,
        "tea-bag": 0.5,
        "tea-bag+water": 0.88,
    },
    "hand_object_interaction": 0.27,
    }

    output = convert_message(message)
    print(output)

