import logging
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join, dirname
from tim_reasoning import SessionManager
from data_generator import generate_data

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

RESOURCE_PATH = join(dirname(__file__), 'resource')


def run_reasoning(recipe_id, video_id):
    perception_outputs = generate_data(recipe_id, video_id)
    results = {'task_id': [], 'true_step': [], 'predicted_step': []}
    sm = SessionManager(patience=1)

    for perception_output in perception_outputs:
        actual_step = perception_output['session']['step_id']
        output_reasoning = sm.handle_message(message=[perception_output])[0]

        if output_reasoning is None:
            continue

        predicted_step = output_reasoning['step_id']
        #predicted_session = output_reasoning['session_id']
        predicted_task = output_reasoning['task_id']
        results['true_step'].append(actual_step)
        results['predicted_step'].append(predicted_step)
        results['task_id'].append(predicted_task)

    results_df = pd.DataFrame.from_dict(results)
    file_path = join(RESOURCE_PATH, f'{recipe_id}_reasoning_results.csv')
    results_df.to_csv(file_path, index=False)
    logger.debug(f'Reasoning results saved at {file_path}')

    return results_df


def visualize_results(results):
    steps = {f'Step {i}': i for i in (results['true_step'].unique())}
    plot = results[['predicted_step', 'true_step']].plot()
    plot.set_yticks([0] + list(steps.values()), labels=['Other Recipe'] + list(steps.keys()))

    plt.show()


def evaluate_reasoning(recipe_id, video_id, plot_results=True):
    results = run_reasoning(recipe_id, video_id)
    results.loc[results.task_id != recipe_id, 'predicted_step'] = 0  # Put 0 if it's another recipe step
    counts = results['predicted_step'].eq(results['true_step'])\
        .value_counts().rename({True: 'match', False: 'no match'})
    total_accuracy = counts['match'] / len(results)

    logger.debug(f'Average accuracy: {round(total_accuracy, 3)}')

    results['match'] = results['predicted_step'].eq(results['true_step'])
    performance_by_step = results.groupby('true_step')['match'].mean().round(3)

    logger.debug('Accuracy for each step:')
    for step_id, step_performance in enumerate(performance_by_step, 1):
        logger.debug(f'Step {step_id}: {step_performance}')

    if plot_results:
        visualize_results(results)


if __name__ == '__main__':
    recipe_id = 'pinwheels'
    video_id = 'pinwheels_2023.04.04-18.33.59'
    evaluate_reasoning(recipe_id, video_id)
