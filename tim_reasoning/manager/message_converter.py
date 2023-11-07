import pandas as pd
from os.path import join, dirname

RESOURCE_PATH = join(dirname(__file__), 'resource')
OBJECTS_OF_INTEREST = {'bowl', 'mug', 'tortilla'}


class MessageConverter:
    def __init__(self) -> None:
        pass

    def read_unique_states(self, target_objects):
        unique_states_df = pd.read_csv(join(RESOURCE_PATH, 'unique_states.csv'))

        unique_states = []
        for _, row in unique_states_df.iterrows():
            if row['object'].strip() not in target_objects:
                continue
            state_id = row['object'].strip() + '_' + row['state'].strip()
            unique_states.append(state_id)

        return unique_states

    def read_unique_objects(self):
        unique_objects_df = pd.read_csv(join(RESOURCE_PATH, 'unique_objects.csv'))

        return [o + '_hoi' for o in unique_objects_df['object'].unique()]

    def convert_message(self, message, entire_message):
        object_name = message['label']

        if object_name not in OBJECTS_OF_INTEREST:
            return None

        all_hois = {}
        for obj_message in entire_message:
            hoi_id = obj_message['label'] + '_hoi'
            hoi_confidence = message.get('hand_object_interaction', 0)
            all_hois[hoi_id] = hoi_confidence

        unique_objects = self.read_unique_objects()
        unique_states = self.read_unique_states([object_name])
        all_columns = unique_states + unique_objects
        all_columns_indices = {v: i for i, v in enumerate(all_columns)}

        perception_predictions = [0] * len(all_columns)
        states_to_add = {}

        for state, confidence in message.get('state', {}).items():
            state_id = object_name + '_' + state
            states_to_add[state_id] = confidence

        for state_id, state_confidence in states_to_add.items():
            index = all_columns_indices[state_id]
            perception_predictions[index] = state_confidence

        for hoi_id, hoi_confidence in all_hois.items():
            index = all_columns_indices[hoi_id]
            perception_predictions[index] = hoi_confidence

        output = {
            'id': message['id'],
            'object_name': object_name,
            'states': perception_predictions,
        }

        return output
