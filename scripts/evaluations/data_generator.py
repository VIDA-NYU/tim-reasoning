import json
import copy
import logging
import random
import pandas as pd
import numpy as np
from datetime import datetime
from os.path import join, dirname

logger = logging.getLogger(__name__)

random.seed(20)
RESOURCE_PATH = join(dirname(__file__), 'resource')
OBJECTS = {
    'pinwheels': ['tortilla'],
    'quesadilla': ['tortilla'],
    'oatmeal': ['bowl'],
    'coffee': ['mug'],
    'tea': ['mug'],
}
FPS = 1
CURRENT_TIME = int(datetime.now().timestamp())
PERCEPTION_OUTPUT_TEMPLATE = {
    "pos": [-0.2149151724097291, -0.4343880843796524, -0.6208099189217009],
    "xyxyn": [0.1, 0.1, 0.2, 0.2],
    "label": "",
    "status": "tracked",
    "id": "1",
    "last_seen": "",
    "state": {},
    "hand_object_interaction": 0.27,
}

MAPPING_IDS = None


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
        .sub(1)
    )
    task_name = video_annotations.iloc[0]['recipe']
    step_id = 0
    tracked_objects = OBJECTS[task_name]
    current_states = {o: None for o in tracked_objects}
    curated_annotations = {
        'task_name': [],
        'video_id': [],
        'start_time': [],
        'end_time': [],
        'step': [],
    }
    curated_annotations.update({o: [] for o in tracked_objects})

    for _, row in video_annotations.iterrows():
        current_step = row['step']
        if not pd.isna(current_step):
            step_id += 1

        for tracked_object in tracked_objects:
            current_state = row[tracked_object]
            if not pd.isna(current_state):
                current_states[tracked_object] = current_state
            curated_annotations[tracked_object].append(
                current_states[tracked_object]
            )
        curated_annotations['task_name'].append(row['recipe'])
        curated_annotations['video_id'].append(row['video_name'])
        curated_annotations['start_time'].append(row['start_time'])
        curated_annotations['end_time'].append(row['end_time'])
        curated_annotations['step'].append(step_id)

    curated_annotations['end_time'][-1] = (
        curated_annotations['start_time'][-1] + 3
    )  # Last element doesn't have and end time, just add 3 secs
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


def select_tracked_objects(row, objects):
    tracked_objects = {}

    for obj in objects:
        if row[obj] != '':
            tracked_objects[obj] = row[obj]

    return tracked_objects


def get_unique_states(annotations, objects):
    unique_states = {}

    for obj in objects:
        states = annotations[obj].unique()
        unique_states[obj] = list(states)

    return unique_states


def format_annotations(raw_annotations, video_id):
    curated_annotations = curate_perception_annotations(raw_annotations, video_id)
    task_name = curated_annotations.iloc[0]['task_name']
    unique_states = get_unique_states(curated_annotations, OBJECTS[task_name])
    annotated_video = {
        'task_name': task_name,
        'task_id': video_id,
        'records': {},
        'unique_states': unique_states,
    }

    for _, row in curated_annotations.iterrows():
        tracked_objects = select_tracked_objects(row, OBJECTS[task_name])
        step_id = row['step']
        if step_id not in annotated_video['records']:
            annotated_video['records'][step_id] = []
        annotated_video['records'][step_id].append(
            {
                'start_time': row['start_time'],
                'end_time': row['end_time'],
                'objects': tracked_objects,
            }
        )

    logger.debug('Annotations formatted.')

    return annotated_video


def make_groundtruth_outputs(annotated_video):
    perception_outputs = []

    for step_id, step_annotations in annotated_video['records'].items():
        session_annotation = {
            'task_id': annotated_video['task_id'],
            'task_name': annotated_video['task_name'],
            'step_id': step_id,
        }
        for step_annotation in step_annotations:
            step_outputs = make_groundtruth_step_outputs(
                session_annotation,
                step_annotation,
                annotated_video['unique_states'],
                PERCEPTION_OUTPUT_TEMPLATE,
                FPS,
            )
            perception_outputs += step_outputs

    logger.debug('Perception outputs generated.')
    return perception_outputs


def make_groundtruth_step_outputs(
    session_annotation, step_annotation, unique_states, output_template, fps
):
    objects = step_annotation['objects']
    start_time = step_annotation['start_time']
    end_time = step_annotation['end_time']
    ranges = list(range(start_time, (end_time + 1) * fps))
    step_outputs = []

    for time_secs in ranges:
        time_stamp = CURRENT_TIME + time_secs
        for object_name, object_state in objects.items():
            object_output = copy.deepcopy(output_template)
            object_output['id'] = MAPPING_IDS[object_name]
            object_output['groundtruth'] = session_annotation
            object_output['label'] = object_name
            object_output['last_seen'] = time_stamp
            state_probas = {s: 0.0 for s in unique_states[object_name]}
            state_probas[object_state] = 1.0
            object_output['state'] = state_probas
            step_outputs.append(object_output)

    return step_outputs


def save_outputs(outputs, file_name):
    file_path = join(RESOURCE_PATH, f'{file_name}.json')
    with open(file_path, 'w') as fout:
        json.dump(outputs, fout, indent=2)

    logger.debug(f'Perception outputs saved at {file_path}')


def complete_state_probas(initial_state_probas, unique_states):
    all_state_probas = {}

    if initial_state_probas is None:
        initial_state_probas = {}

    total_proba = 0

    for state_name, state_proba in initial_state_probas.items():
        try:
            unique_states.remove(state_name)
        except:
            continue
        all_state_probas[state_name] = state_proba
        total_proba += state_proba

    remaining_proba = 1 - total_proba
    remaining_proba /= len(unique_states)

    for unique_state in unique_states:
        all_state_probas[unique_state] = remaining_proba

    return all_state_probas


def generate_perturbed_indices(list_size, percentage):
    sample_size = int(list_size * percentage)
    indices = random.sample(range(0, list_size), sample_size)
    boolean_values = [False] * list_size

    for index in indices:
        boolean_values[index] = True

    return boolean_values


def make_noisy_step_outputs(
    session_annotation,
    step_annotation,
    unique_states,
    output_template,
    error_rate,
    fps,
):
    objects = step_annotation['objects']
    start_time = step_annotation['start_time']
    end_time = step_annotation['end_time']
    ranges = list(range(start_time, (end_time + 1) * fps))
    indices_to_perturb = generate_perturbed_indices(len(ranges), error_rate)
    step_outputs = []

    for index, time_secs in enumerate(ranges):
        time_stamp = CURRENT_TIME + time_secs
        for object_name, object_state in objects.items():
            object_output = copy.deepcopy(output_template)
            object_output['id'] = MAPPING_IDS[object_name]
            object_output['groundtruth'] = session_annotation
            object_output['label'] = object_name
            object_output['last_seen'] = time_stamp

            if indices_to_perturb[index]:
                object_states = copy.deepcopy(unique_states[object_name])
                initial_state_probas = {
                    random.choice(object_states): round(random.uniform(0, 1), 2)
                }
                state_probas = complete_state_probas(
                    initial_state_probas, object_states
                )
            else:
                state_probas = {s: 0.0 for s in unique_states[object_name]}
                state_probas[object_state] = 1.0

            object_output['state'] = state_probas

            step_outputs.append(object_output)

    return step_outputs


def make_noisy_outputs(annotated_video, noise_config):
    steps_with_noises = noise_config['steps']
    error_rate = noise_config['error_rate']
    fps = noise_config['fps'] if 'fps' in noise_config else FPS
    perception_outputs = []

    for step_id, step_annotations in annotated_video['records'].items():
        session_annotation = {
            'task_id': annotated_video['task_id'],
            'task_name': annotated_video['task_name'],
            'step_id': step_id,
        }

        for step_annotation in step_annotations:
            if step_id in steps_with_noises:
                step_outputs = make_noisy_step_outputs(
                    session_annotation,
                    step_annotation,
                    annotated_video['unique_states'],
                    PERCEPTION_OUTPUT_TEMPLATE,
                    error_rate,
                    fps,
                )
            else:
                step_outputs = make_groundtruth_step_outputs(
                    session_annotation,
                    step_annotation,
                    annotated_video['unique_states'],
                    PERCEPTION_OUTPUT_TEMPLATE,
                    fps,
                )
            perception_outputs += step_outputs

    return perception_outputs


def skip_steps(perception_outputs, steps_to_skip):
    new_perception_outputs = []

    for output in perception_outputs:
        actual_step = output['groundtruth']['step_id']
        if actual_step not in steps_to_skip:
            new_perception_outputs.append(output)

    return new_perception_outputs


def reorder_steps(perception_outputs, steps_to_reorder):
    new_perception_outputs = []
    map_steps = {}

    for output in perception_outputs:
        actual_step = output['groundtruth']['step_id']

        if actual_step not in map_steps:
            map_steps[actual_step] = []

        map_steps[actual_step].append(output)

    new_order_steps = np.array(list(map_steps.keys()))
    indices = [s-1 for s in steps_to_reorder]
    shuffled_indices = indices.copy()
    random.shuffle(shuffled_indices)
    new_order_steps[indices] = new_order_steps[shuffled_indices]

    for step in new_order_steps:
        new_perception_outputs += map_steps[step]

    return new_perception_outputs


def generate_data(task_name, video_id=None, noise_config=None, error_config=None):
    if error_config is None:
        error_config = {'name': None}
    global MAPPING_IDS
    MAPPING_IDS = {
        'tortilla': random.randint(0, 1000),
        'knife': random.randint(1000, 2000),
        'plate': random.randint(2000, 3000),
        'bowl': random.randint(3000, 4000),
        'mug': random.randint(3000, 4000),
    }

    raw_annotations_path = join(RESOURCE_PATH, 'raw_annotations.csv')
    raw_annotations = pd.read_csv(raw_annotations_path)
    formatted_annotations = format_annotations(raw_annotations, video_id)

    if noise_config is None:
        perception_outputs = make_groundtruth_outputs(formatted_annotations)
    else:
        perception_outputs = make_noisy_outputs(
            formatted_annotations, noise_config
        )

    if error_config['name'] == 'skip_steps':
        perception_outputs = skip_steps(perception_outputs, error_config['steps'])

    elif error_config['name'] == 'reorder_steps':
        perception_outputs = reorder_steps(perception_outputs, error_config['steps'])

    save_outputs(perception_outputs, f'{task_name}_perception_outputs')

    return perception_outputs
