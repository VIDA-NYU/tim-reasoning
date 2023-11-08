import os
import csv
from os.path import join
from datetime import datetime

LOGS_FOLDER = os.getenv('REASONING_LOGS_PATH', '')


class DemoLogger:
    def __init__(self, output_file):
        self.output_file = join(LOGS_FOLDER, output_file)
        self.writer = None
        self.start_time = None
        self.last_status = None
        self.recipe_map = {
            "pinwheels": "A",
            "coffee": "B",
            "tea": "C",
            "oatmeal": "D",
            "quesadilla": "E",
        }

    def start_trial(self):
        self.start_time = datetime.now()
        self.writer = csv.writer(
            open(self.output_file, 'w'), quoting=csv.QUOTE_NONNUMERIC, quotechar="'"
        )
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
        # new_status = ["NYU", "null", "null", "null"]
        if message is not None:
            task_name = self.recipe_map[message.get('task_name')]
            step_id = message.get('step_id')
            step_status = self._get_step_status(message)
            new_status = ["NYU", task_name, step_id, step_status]
            if step_status == "error":
                new_status += [message.get('error_description')]
            self.write_in_file(timestamp, new_status)

    def _get_step_status(self, message):
        if message.get('error_status'):
            return "error"
        elif message.get('step_status') == 'IN_PROGRESS':
            return "active"
        else:
            return "null"

    def write_in_file(self, timestamp, new_status):
        if new_status != self.last_status:
            self.last_status = new_status
            self.writer.writerow([timestamp] + new_status)
