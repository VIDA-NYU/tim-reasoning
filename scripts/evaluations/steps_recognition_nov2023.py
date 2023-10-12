
import json
import pandas as pd
from os.path import join, dirname
from tim_reasoning import SessionManager

RESULTS_PATH = '/Users/rlopez/PTG/tim-reasoning/scripts/evaluations/resource'
recipe_id = 'pinwheels'
video_name = 'pinwheels_2023.03.30-16.38.48'

output_path = join(dirname(__file__), f'../data_generator/resource/perception_object_states/{recipe_id}_session.json')

with open(output_path, 'r') as fin:
    perception_outputs = json.load(fin)


results = {'video': [], 'true_step': [], 'predicted_step': []}
sm = SessionManager()

for perception_output in perception_outputs:
    actual_step = perception_output['session']['step_id']
    print(f'Actual step: {actual_step}')
    sm.handle_message(message=[perception_output])
    fake_output_reasoning = {
        "session_id": 1,
        "task_id":  "pinwheels",
        "step_id": 1,
        "step_status": "IN_PROGRESS",
        "step_description": "wipe off knife with a paper towel",
        "error_status": True,
        "error_description": "Errors detected in the step"
        ""
    }
    predicted_step = fake_output_reasoning['step_id']

    results['true_step'].append(actual_step)
    results['predicted_step'].append(predicted_step)
    results['video'].append(video_name)

    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(join(RESULTS_PATH, f'results_{recipe_id}.csv'), index=False)
