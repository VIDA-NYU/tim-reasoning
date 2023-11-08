import json
import logging
import random
import pandas as pd
from os.path import join, dirname
from tim_reasoning import SessionManager
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

random.seed(0)
RESOURCE_PATH = join(dirname(__file__), 'resource')
PATIENCE = 5


def run_reasoning(video_id, file_name):
    sm = SessionManager(patience=PATIENCE)

    with open(join(RESOURCE_PATH, f'perception_outputs/{video_id}-deticmemory.json'), 'r') as read_file:
        detic_data = json.load(read_file)

    data = {'task': [], 'step': [], 'obj_id':[], 'obj_name':[]}
    for entry in detic_data:
        entry = entry['values']
        for obj_entry in entry:
            _, dashboard = sm.handle_message([obj_entry], entry)
            if len(dashboard) > 0:
                predicted_task_id = dashboard['task_id']
                predicted_task_name = dashboard['task_name']
                predicted_step = dashboard['step_num']
                new_name = predicted_task_name + '_' + str(predicted_task_id)
                data['task'].append(new_name)
                data['step'].append(predicted_step)
                data['obj_id'].append(dashboard['object_id'])
                data['obj_name'].append(dashboard['object_name'])

    data_df = pd.DataFrame.from_dict(data)
    data_df.to_csv(f'{file_name}.csv', index_label='index')


def plot(file_name):
    df = pd.read_csv(f'{file_name}.csv')

    # Extract index column as x-values
    x = df['index']
    x = df.index.values

    # Initialize plot
    fig, ax = plt.subplots()

    # Plot each task separately
    for task in df['task'].unique():
        # Filter dataframe by task
        task_df = df[df['task'] == task]

        # Plot just x and y columns
        ax.plot(task_df['index'], task_df['step'], label=task)

    ax.set_xlabel('Index')
    ax.set_ylabel('Step')
    ax.set_title('Plot by Task')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    video_id = '2023.11.07-04.32.13'
    file_name = 'data'
    run_reasoning(video_id, file_name)
    plot(file_name)
