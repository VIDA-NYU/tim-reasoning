import os
import json
import copy
import logging
import random
import numpy as np
import pandas as pd
from os.path import join, dirname

logger = logging.getLogger(__name__)

random.seed(0)
RESOURCE_PATH = join(dirname(__file__), 'resource')


def curate_perception_annotations(
    annotations, video_id, save_curated_annotation=True
):
    video_annotations = copy.deepcopy(
        annotations[annotations['video_name'] == video_id]
    )
    video_annotations = video_annotations[pd.notnull(video_annotations['time'])]
    video_annotations['start_time'] = video_annotations['time'].apply(to_seconds)
    video_annotations['end_time'] = (
        video_annotations['start_time']
        .shift(-1)
        .astype('Int64', errors='ignore')
        #.sub(1)
    )
    #print(video_id, video_annotations)
    task_name = video_annotations.iloc[0]['recipe']
    step_id = 0
    actual_step = None

    curated_annotations = {
        'task_name': [],
        'video_id': [],
        'start_time': [],
        'end_time': [],
        'step': [],
    }

    for _, row in video_annotations.iterrows():
        current_step = row['step']
        if not pd.isna(current_step):
            step_id += 1
            actual_step = current_step

        curated_annotations['task_name'].append(row['recipe'])
        curated_annotations['video_id'].append(row['video_name'])
        curated_annotations['start_time'].append(row['start_time'])
        curated_annotations['end_time'].append(row['end_time'])
        curated_annotations['step'].append(int(actual_step))

    curated_annotations['end_time'][-1] = (
        curated_annotations['start_time'][-1] + 30000
    )  # Last element doesn't have and end time, just add 30000 secs
    curated_annotations_df = pd.DataFrame.from_dict(curated_annotations)

    if save_curated_annotation:
        curated_annotations_df.to_csv(
            join(RESOURCE_PATH, f'{task_name}_curated_annotation.csv'), index=False
        )

    logger.debug('Perception outputs curated.')

    return curated_annotations_df


def to_seconds(minutes_seconds):
    minutes, seconds = map(int, minutes_seconds.split(':'))
    only_seconds = minutes * 60 + seconds

    return only_seconds


def read_unique_states(target_object=None):
    unique_states_df = pd.read_csv(join(RESOURCE_PATH, 'unique_states.csv'))

    unique_states = []
    for _, row in unique_states_df.iterrows():
        if row['object'].strip() != target_object:
            continue
        state_id = row['object'].strip() + '_' + row['state'].strip()
        unique_states.append(state_id)

    return unique_states


def read_unique_objects():
    unique_objects_df = pd.read_csv(join(RESOURCE_PATH, 'unique_objects.csv'))

    return [o + '_hoi' for o in unique_objects_df['object'].unique()]


def remap_labels(label):
    RENAME = {
        '[partial]': '',
        '[full]': '',
        'floss-underneath': 'ends-cut',
        'floss-crossed': 'ends-cut',
        'raisins[cooked]': 'raisins',
        'oatmeal[cooked]+raisins': 'oatmeal+raisins',
        'teabag': 'tea-bag',
        '+stirrer': '',
        '[stirred]': '',
        'water+honey': 'water',
        'with-quesadilla': 'with-food',
        'with-pinwheels': 'with-food',
        'oatmeal+raisin+banana+cinnamon+honey': 'oatmeal+raisins+banana+cinnamon+honey',
    }
    for old, new in RENAME.items():
        label = label.replace(old, new)
    return label


def make_data(raw_annotations, video_id, target_object='bowl'):
    curated_annotations = curate_perception_annotations(raw_annotations, video_id)
    unique_objects = read_unique_objects()
    unique_states = read_unique_states(target_object)
    all_columns = unique_states + unique_objects + ['step']

    all_data = {f: [] for f in all_columns}

    with open(join(RESOURCE_PATH, f'perception_outputs/{video_id}_detic-image-misc.json'), 'r') as read_file:
        detic_data = json.load(read_file)

    time_ranges = []
    for _, row in curated_annotations.iterrows():
        time_ranges.append((row['start_time'], row['end_time'], row['step']))

    cache_states = []
    for entry in detic_data:
        objects = entry['objects']
        if len(objects) == 0:
            continue
        timestamp = entry['timestamp']
        step = None
        for time_range in time_ranges:
            if time_range[0] <= timestamp <= time_range[1]:
                step = time_range[2]
                break

        states_to_add = {s: 0 for s in unique_states}
        objects_to_add = {o: 0 for o in unique_objects}
        counter_valid_states = 0
        for obj in objects:
            obj_name = obj['label']
            states = obj.get('state', {})
            for state, proba in states.items():
                counter_valid_states += 1
                #state = remap_labels(state)
                state_id = obj_name + '_' + state
                if state_id in states_to_add:
                    states_to_add[state_id] = proba

            hoi_value = obj.get('hand_object_interaction', 0)
            objects_to_add[obj_name + '_hoi'] = hoi_value

        if counter_valid_states == 0:
            if len(cache_states) > 1:
                last_states = cache_states[-5:]
                tmp_states = {s: [] for s in unique_states}

                for states in last_states:
                    for state, proba in states.items():
                        tmp_states[state].append(proba)

                for state in tmp_states:
                    last_probas = tmp_states[state]
                    tmp_states[state] = np.mean(last_probas)

                states_to_add = tmp_states

        else:
            cache_states.append(states_to_add)

        for state_id, proba in states_to_add.items():
            all_data[state_id].append(proba)

        for object_id, hoi_value in objects_to_add.items():
            all_data[object_id].append(hoi_value)

        all_data['step'].append(step)

    all_data_df = pd.DataFrame.from_dict(all_data)
    all_data_df.to_csv(join(RESOURCE_PATH, f'matrices/data_{video_id}.csv'), index=False)


def create_matrices():
    raw_annotations_path = join(RESOURCE_PATH, 'raw_annotations_new.csv')
    raw_annotations = pd.read_csv(raw_annotations_path)

    videos = {
        'oatmeal': [
            'oatmeal_2023.06.16-20.08.08',
            'oatmeal_2023.06.30-20.10.42',
            'oatmeal_2023.06.16-20.33.26',
            'oatmeal_2023.06.16-20.18.26',
            'oatmeal_2023.11.03-18.51.03',
            'oatmeal_2023.11.03-19.04.04',
            'oatmeal_2023.11.03-18.43.03'
        ]
    }

    for video_id in videos['oatmeal']:
        print(video_id)
        make_data(raw_annotations, video_id)


create_matrices()
