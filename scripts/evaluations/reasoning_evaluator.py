import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname
from tim_reasoning import SessionManager
from data_generator import generate_task, generate_multiple_sessions

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

RESOURCE_PATH = join(dirname(__file__), 'resource')
PATIENCE = 3


def run_reasoning(session_id, session, save_reasoning_outputs=True):
    results = {
        'true_task': [],
        'predicted_task': [],
        'true_step': [],
        'predicted_step': [],
    }
    sm = SessionManager(patience=PATIENCE)
    all_outputs = []

    for perception_output in session:
        actual_step = perception_output['groundtruth']['step_id']
        actual_task = perception_output['groundtruth']['task_name']
        reasoning_outputs = sm.handle_message(message=[perception_output])

        if reasoning_outputs['active_tasks'][0] is None:
            continue

        all_outputs.append(reasoning_outputs)

        for reasoning_output in reasoning_outputs['active_tasks']:
            predicted_step = reasoning_output['step_id']
            predicted_task = reasoning_output['task_name']
            results['true_task'].append(actual_task)
            results['true_step'].append(actual_step)
            results['predicted_step'].append(predicted_step)
            results['predicted_task'].append(predicted_task)

    results_df = pd.DataFrame.from_dict(results)

    if save_reasoning_outputs:
        with open(
            join(RESOURCE_PATH, f'{session_id}_reasoning_outputs.json'), 'w'
        ) as fout:
            json.dump(all_outputs, fout, indent=2)

        file_path = join(RESOURCE_PATH, f'{session_id}_reasoning_results.csv')
        results_df.to_csv(file_path, index=False)

    return results_df


def calculate_accuracy(results, session_id):
    results['match_task'] = results['true_task'] == results['predicted_task']
    task_matches = 0
    try:
        task_matches = results['match_task'].value_counts()[True]
    except:
        pass
    task_accuracy = task_matches / len(results)
    logger.debug(f'Task recognition accuracy in {session_id}: {round(task_accuracy, 3)}')

    results['match_step'] = (results['true_task'] == results['predicted_task']) & (
            results['true_step'] == results['predicted_step']
    )

    step_matches = 0
    try:
        step_matches = results['match_step'].value_counts()[True]
    except:
        pass
    step_accuracy = step_matches / len(results)
    logger.debug(f'Step recognition accuracy in {session_id}: {round(step_accuracy, 3)}')

    performance_by_step = (
        results.groupby(['true_task', 'true_step'])['match_step'].mean().round(3)
    )

    logger.debug(f'Accuracy for each step in {session_id}:')
    for step_id, step_performance in enumerate(performance_by_step, 1):
        logger.debug(f'Step {step_id}: {step_performance}')

    return task_accuracy, step_accuracy


def visualize_results(results, session_id, task_name, error_name):
    steps = {f'Step {i}': i for i in (results['true_step'].unique())}
    results.loc[results.predicted_task != task_name, 'predicted_step'] = 0  # Put 0 if it's another recipe step
    plot = results.plot(legend=True)
    plot.set_yticks(
        [0] + list(steps.values()), labels=['Other Recipe'] + list(steps.keys())
    )
    plt.title(
        f'Model Patience {PATIENCE}, task {task_name} with error {str(error_name)}'
    )

    plt.savefig(join(RESOURCE_PATH, f'{task_name}_{session_id}_error{str(error_name)}_P{PATIENCE}_plot.png'))


def evaluate_reasoning(num_sessions, seed_tasks, add_noise, error_name, save_plots=False):
    sessions = generate_multiple_sessions(num_sessions, seed_tasks,  add_noise, error_name)
    task_accuracies = []
    step_accuracies = []

    for session_id, session in sessions:
        results = run_reasoning(session_id, session)
        task_accuracy, step_accuracy = calculate_accuracy(results, session_id)
        task_accuracies.append(task_accuracy)
        step_accuracies.append(step_accuracy)
        if save_plots:
            task_to_visualize = session[0]['groundtruth']['task_name']
            #  The visualization supports only 1 task
            visualize_results(results, session_id, task_to_visualize, error_name)

    logger.debug(f'Global task recognition accuracy: {round(np.mean(task_accuracies), 2)} +/- {round(np.std(task_accuracies), 2)}')
    logger.debug(f'Global step recognition accuracy: {round(np.mean(step_accuracies), 2)} +/- {round(np.std(step_accuracies), 2)}')


if __name__ == '__main__':
    seed_tasks = [
        ('pinwheels', 'pinwheels_2023.04.04-18.33.59'),
        ('quesadilla', 'quesadilla_2023.06.16-18.57.48'),
        ('oatmeal', 'oatmeal_2023.06.16-20.33.26'),
        ('coffee', 'coffee_mit-eval'),
        ('tea', 'tea_2023.06.16-18.43.48')
    ]
    error_name = None  # Other options are: 'skip_steps' and 'reorder_steps'
    add_noise = True
    num_sessions = 100
    evaluate_reasoning(num_sessions, seed_tasks, add_noise, error_name)

