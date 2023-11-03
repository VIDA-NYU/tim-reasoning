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


def calculate_accuracy(results):
    results['match_task'] = results['true_task'] == results['predicted_task']
    task_matches = 0
    try:
        task_matches = results['match_task'].value_counts()[True]
    except:
        pass
    task_accuracy = task_matches / len(results)
    logger.debug(f'Task recognition accuracy: {round(task_accuracy, 3)}')

    results['match_step'] = (results['true_task'] == results['predicted_task']) & (
            results['true_step'] == results['predicted_step']
    )

    step_matches = 0
    try:
        step_matches = results['match_step'].value_counts()[True]
    except:
        pass
    step_accuracy = step_matches / len(results)
    logger.debug(f'Step recognition accuracy: {round(step_accuracy, 3)}')

    performance_by_step = (
        results.groupby(['true_task', 'true_step'])['match_step'].mean().round(3)
    )

    logger.debug('Accuracy for each step:')
    for step_id, step_performance in enumerate(performance_by_step, 1):
        logger.debug(f'Step {step_id}: {step_performance}')

    return task_accuracy, step_accuracy


def evaluate_reasoning(num_sessions):
    sessions = generate_multiple_sessions(num_sessions, error_name=None)
    task_accuracies = []
    step_accuracies = []

    for session_id, session in sessions:
        results = run_reasoning(session_id, session)
        task_accuracy, step_accuracy = calculate_accuracy(results)
        task_accuracies.append(task_accuracy)
        step_accuracies.append(step_accuracy)

    logger.debug(f'Task recognition accuracy: {round(np.mean(task_accuracies), 2)} +/- {round(np.std(task_accuracies), 2)}')
    logger.debug(f'Step recognition accuracy: {round(np.mean(step_accuracies), 2)} +/- {round(np.std(step_accuracies), 2)}')


if __name__ == '__main__':
    evaluate_reasoning(100)

