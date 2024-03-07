import json
from os.path import join, dirname

FOLDER_PATH = join(dirname(__file__), './ego4d/annotations')


def read_annotations(file_path):
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


def read_train_data():
    annotations = read_annotations(join(FOLDER_PATH, 'goalstep_train.json'))

    return annotations


if __name__ == '__main__':
    annotations = read_train_data()
    video_id = '1938c632-f575-49dd-8ae0-e48dbb467920'
    video_data = annotations[video_id]
    general_summary = make_general_summary(video_data)
    detailed_summary = make_detailed_summary(video_data)
    print(general_summary)
    print(detailed_summary)
