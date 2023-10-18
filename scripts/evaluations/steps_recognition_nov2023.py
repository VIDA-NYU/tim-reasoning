
import json
import pandas as pd
from os.path import join, dirname
from tim_reasoning import SessionManager

RESULTS_PATH = join(dirname(__file__), './resource')
PERCEPTION_OUTPUTS_PATH = join(dirname(__file__), '../data_generator/resource/perception_object_states')


def generate_predictions(file_name):
    output_path = join(PERCEPTION_OUTPUTS_PATH, f'{file_name}.json')

    with open(output_path, 'r') as fin:
        perception_outputs = json.load(fin)

    results = {'task_index': [], 'true_step': [], 'predicted_step': []}
    sm = SessionManager(patience=1)

    for perception_output in perception_outputs:
        actual_step = perception_output['session']['step_id']
        output_reasoning = sm.handle_message(message=[perception_output])[0]

        if output_reasoning is None:
            continue  # output_reasoning = {'step_id': 1, 'session_id': 0}  # Fake the None values
        print('reasoning output', output_reasoning)
        predicted_step = output_reasoning['step_id']
        predicted_session = output_reasoning['session_id']
        results['true_step'].append(actual_step)
        results['predicted_step'].append(predicted_step)
        results['task_index'].append(predicted_session)
        results_df = pd.DataFrame.from_dict(results)
        results_df.to_csv(join(RESULTS_PATH, f'results_{file_name}.csv'), index=False)


perception_outputs_file = 'pinwheels_session'
generate_predictions(perception_outputs_file)
