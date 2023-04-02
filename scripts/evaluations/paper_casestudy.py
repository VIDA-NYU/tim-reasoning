import json
import datetime
from os.path import join, dirname


#  List of steps with start and end second
label_ranges = {
    '0': {'start_min': '0:03', 'end_min': '0:07'},
    '1': {'start_min': '0:14', 'end_min': '0:57'},
    '2': {'start_min': '1:02', 'end_min': '1:11'},
    '3': {'start_min': '1:16', 'end_min': '1:43'},
    '4': {'start_min': '1:47', 'end_min': '1:54'},
    '5': {'start_min': '1:58', 'end_min': '2:07'},
    '6': {'start_min': '2:11', 'end_min': '2:37'},
    '7': {'start_min': '2:41', 'end_min': '2:55'},
    '8': {'start_min': '3:10', 'end_min': '3:18'},
    '9': {'start_min': '3:19', 'end_min': '3:29'},
    '10': {'start_min': '3:34', 'end_min': '4:11'},
    '11': {'start_min': '4:16', 'end_min': '4:23'}
}

labels_by_steps = {}
labels_by_seconds = {}
for step_id, step_ranges in label_ranges.items():
    start_min = step_ranges['start_min'].split(':')
    start_sec = int(start_min[0]) * 60 + int(start_min[1])
    end_min = step_ranges['end_min'].split(':')
    end_sec = int(end_min[0]) * 60 + int(end_min[1])
    labels_by_steps[step_id] = []
    for second in range(start_sec, end_sec+1):
        labels_by_steps[step_id].append(second)
        labels_by_seconds[second] = step_id

output_id = 'pinwheels_2023.03.20-19.05.44_confidence_0_3.json'
output_path = join(dirname(__file__),f'resource/{output_id}')

with open(output_path) as fin:
    reasoning_outputs = json.load(fin)

start_unix_time = int(reasoning_outputs[0]['timestamp'][:-2])
start_datetime = datetime.datetime.fromtimestamp(round(start_unix_time / 1000))

predictions_by_seconds = {}
predictions_by_steps = {}
for reasoning_output in reasoning_outputs:
    current_unix_time = int(reasoning_output['timestamp'][:-2])
    current_datetime = datetime.datetime.fromtimestamp(round(current_unix_time / 1000))
    current_second = int((current_datetime - start_datetime).total_seconds())
    predicted_step = str(reasoning_output['step_id'])

    if predicted_step not in predictions_by_steps:
        predictions_by_steps[predicted_step] = []

    predictions_by_seconds[current_second] = predicted_step
    predictions_by_steps[predicted_step].append(current_second)

corrects = 0
for second in labels_by_seconds:
    if labels_by_seconds[second] == predictions_by_seconds.get(second, None):
        corrects += 1

print('Total accuracy:', round(corrects/float(len(labels_by_seconds)), 2))

for step_id in labels_by_steps:
    actual_seconds = labels_by_steps[step_id]
    predicted_seconds = predictions_by_steps.get(step_id, [])
    common_seconds = len(set(actual_seconds) & set(predicted_seconds))
    step_accuracy = round(common_seconds / float(len(actual_seconds)), 2)
    print(f'Accuracy step {step_id}: {step_accuracy}')
