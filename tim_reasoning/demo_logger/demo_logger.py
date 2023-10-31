import csv
from datetime import datetime


class DemoLogger:
    def __init__(self, output_file):
        self.output_file = output_file
        self.writer = None
        self.start_time = None
        self.recipe_map = {
            "pinwheels": "A",
            "coffee": "B",
            "tea": "C",
            "oatmeal": "D",
            "quesadilla": "E",
        }

    def start_trial(self):
        self.start_time = datetime.now()
        self.writer = csv.writer(open(self.output_file, 'w'))
        self.writer.writerow(
            [
                'timestamp',
                'team_name',
                'recipe_id',
                'step_number',
                'current_status',
                'optional_comment',
            ]
        )

    def log_message(self, message):
        if not self.writer:
            raise Exception("Must call start_trial() before logging messages!")

        timestamp = datetime.now()
        if message is None:
            self.writer.writerow([timestamp, "NYU", "null", "null", "null"])
        else:
            task_id = message.get('task_id')
            task_name = self.recipe_map[message.get('task_name')]
            step_id = message.get('step_id')
            step_status = self._get_step_status(message)

            self.writer.writerow([timestamp, "NYU", task_name, step_id, step_status])

    def _get_step_status(self, message):
        if message.get('error_status'):
            return "error"
        elif message.get('step_status') == 'IN_PROGRESS':
            return "active"
        else:
            return "null"
