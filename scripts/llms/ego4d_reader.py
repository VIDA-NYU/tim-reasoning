import json
from os.path import join, dirname

FOLDER_PATH = join(dirname(__file__), './ego4d/annotations')


def read_annotations(data_split='goalstep_train'):
    file_path = join(FOLDER_PATH, f'{data_split}.json')
    with open(file_path, 'r') as file:
        raw_annotations = json.load(file)

    annotations = {}

    for annotation in raw_annotations['videos']:
        video_id = annotation.pop('video_uid')
        annotations[video_id] = annotation

    return annotations


def get_goal(video_data):
    goal = video_data['goal_description']

    return goal


def get_steps(video_data):
    steps = [x['step_description'] for x in video_data['segments']]
    if len(steps) == 0:
        return None

    return steps

def get_summary_sentences(video_data):
    steps = video_data['summary']
    if len(steps) == 0:
        return None

    return steps

def aggregate_steps(video_data):
    steps = '. '.join(
        [f'{i}. {x["step_description"].capitalize()}' for i, x in enumerate(video_data['segments'], 1)])
    if len(steps) == 0:
        return None

    return steps


def make_general_summary(video_data):
    steps = '. '.join([f'{i}: {x.capitalize()}' for i, x in enumerate(video_data['summary'], 1)])
    if len(steps) == 0:
        return None

    summary = f'In the video the user follow these steps: {steps}'

    return summary


def make_detailed_summary(video_data):
    steps = aggregate_steps(video_data)
    if len(steps) == 0:
        return None

    summary = f'In the video the user follow these steps: {steps}'

    return summary


def get_valid_annotations(*args):
    all_annotations = read_annotations(*args)
    valid_annotations = {}

    for video_id, video_data in all_annotations.items():
        if len(video_data['segments']) > 0 and len(video_data['summary']) > 0:
            valid_annotations[video_id] = video_data

    print(f'Found {len(valid_annotations)} valid annotations out of {len(all_annotations)}.')

    return valid_annotations


def get_step_timestamps(video_data):
    step_timestamps = []

    for segment in video_data['segments']:
        description = segment['step_description']
        start_time = segment['start_time'] 
        end_time = segment['end_time']
        step_timestamps.append({'description': description, 'start_time': start_time, 'end_time': end_time})

    return step_timestamps


if __name__ == '__main__':
    import random
    random.seed(27)
    videos = ['09bccca1-368b-4776-9433-3c8835837110', '1938c632-f575-49dd-8ae0-e48dbb467920']
    valid_annotations = get_valid_annotations()
    #print(list(valid_annotations.keys())[:100])
    video_id = videos[0]
    video_data = valid_annotations[video_id]
    general_summary = make_general_summary(video_data)
    steps = get_steps(video_data)
    #print('Summary:', general_summary)
    #print('Steps:', steps)

    print('--------- NO MODIFICATION ---------')
    steps = '. '.join([f'{i}: {x.capitalize()}' for i, x in enumerate(steps, 1)])
    print(f'In the video the person should follow these steps: {steps}. Which step is missing in the video?', '\n')
    print(f'In the video the person should follow these steps: {steps}. Were the steps done in order?', '\n')

    print('---------- MISSING STEPS ----------')
    steps = get_steps(video_data)
    new_step = 'Cut the oranges'
    position_to_insert = random.randrange(len(steps) - 1 )
    steps.insert(position_to_insert, new_step)
    steps = '. '.join([f'{i}: {x.capitalize()}' for i, x in enumerate(steps, 1)])
    print(f'In the video the person should follow these steps: {steps}. Which step is missing in the video?', '\n')

    print('--------- REORDERED STEPS ---------')
    steps = get_steps(video_data)
    random.shuffle(steps)
    steps = '. '.join([f'{i}: {x.capitalize()}' for i, x in enumerate(steps, 1)])
    print(f'In the video the person should follow these steps: {steps}. Were the steps done in order?', '\n')

    step_timestamps = get_step_timestamps(video_data)
    print(step_timestamps)
