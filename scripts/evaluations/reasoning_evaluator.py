import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join, dirname
from tim_reasoning import SessionManager
from data_generator import generate_data

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

RESOURCE_PATH = join(dirname(__file__), 'resource')


def run_reasoning(recipe_id, video_id, noise_config):
    perception_outputs = generate_data(recipe_id, video_id, noise_config)
    results = {'true_task': [], 'predicted_task': [], 'true_step': [], 'predicted_step': []}
    sm = SessionManager(patience=1)
    all_outputs = []

    for perception_output in perception_outputs:
        actual_step = perception_output['session']['step_id']
        actual_task = perception_output['session']['task_id']
        outputs_reasoning = sm.handle_message(message=[perception_output])

        if outputs_reasoning[0] is None:
            continue

        all_outputs.append(outputs_reasoning)

        for output_reasoning in outputs_reasoning:
            predicted_step = output_reasoning['step_id']
            predicted_task = output_reasoning['task_id']
            results['true_task'].append(actual_task)
            results['true_step'].append(actual_step)
            results['predicted_step'].append(predicted_step)
            results['predicted_task'].append(predicted_task)

    results_df = pd.DataFrame.from_dict(results)
    file_path = join(RESOURCE_PATH, f'{recipe_id}_reasoning_results.csv')
    results_df.to_csv(file_path, index=False)
    logger.debug(f'Reasoning results saved at {file_path}')

    with open('reasoning_output.json', 'w') as fout:
        json.dump(all_outputs, fout, indent=2)
    return results_df


def visualize_results(results):
    steps = {f'Step {i}': i for i in (results['true_step'].unique())}
    plot = results[['predicted_step', 'true_step']].plot()
    plot.set_yticks([0] + list(steps.values()), labels=['Other Recipe'] + list(steps.keys()))

    plt.show()


def evaluate_reasoning(recipe_id, video_id, noise_config=None, plot_results=True):
    results = run_reasoning(recipe_id, video_id, noise_config)

    results['match_task'] = (results['true_task'] == results['predicted_task'])
    total_accuracy = results['match_task'].value_counts()[True] / len(results)
    logger.debug(f'Task recognition accuracy: {round(total_accuracy, 3)}')

    results['match_step'] = ((results['true_task'] == results['predicted_task']) & (results['true_step'] == results['predicted_step']))
    total_accuracy = results['match_step'].value_counts()[True] / len(results)
    logger.debug(f'Task recognition accuracy: {round(total_accuracy, 3)}')

    performance_by_step = results.groupby(['true_task', 'true_step'])['match_step'].mean().round(3)

    print(performance_by_step, type(performance_by_step))
    logger.debug('Accuracy for each step:')
    for step_id, step_performance in enumerate(performance_by_step, 1):
        logger.debug(f'Step {step_id}: {step_performance}')

    if plot_results:
        visualize_results(results)


if __name__ == '__main__':
    recipe_id = 'pinwheels'
    video_id = 'pinwheels_2023.04.04-18.33.59'
    #recipe_id = 'quesadilla'
    #video_id = 'quesadilla_2023.06.16-18.57.48'
    #recipe_id = 'outmeal'
    #video_id = 'oatmeal_2023.06.16-20.33.26'
    noise_config = {'steps': [1], 'error_rate': 0.4}
    noise_config = None
    evaluate_reasoning(recipe_id, video_id, noise_config)
