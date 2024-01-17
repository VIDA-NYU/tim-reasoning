import joblib
import numpy as np

from tim_reasoning import MessageConverter
from os.path import join, dirname

TOTAL_STEPS_PINWHEELS = 12
TOTAL_STEPS_TEA = 7


class Task:
    def __init__(self, obj_id):
        self.object_ids = set()
        self.current_step = None
        self.task_name = None
        self.object_ids.add(obj_id)

    def get_task_name(self):
        return self.task_name

    def get_current_step(self):
        return self.current_step

    def get_object_ids(self):
        return self.object_ids


class RunML:
    def __init__(self) -> None:
        self.tasks = {}  # <id_task, Task>
        self.curr_task_id = 0
        self.mc = MessageConverter()
        self.object_info = {}  # id: [[states], [states] ... ]
        self.window_size = 5
        self.current_task_name = None
        self.current_step = None

    def load_model(self, object_name):
        loaded_rf = None
        if object_name == "bowl":
            model_location = join(dirname(__file__), "../resource/models/oatmeal_rf.joblib")
            loaded_rf = joblib.load(model_location)
        elif object_name == "tortilla":
            model_location = join(
                dirname(__file__), "../resource/models/pinwheels_quesadilla_rf.joblib"
            )
            loaded_rf = joblib.load(model_location)
        elif object_name == "mug":
            model_location = join(dirname(__file__), "../resource/models/tea_coffee_rf.joblib")
            loaded_rf = joblib.load(model_location)
        return loaded_rf

    def create_dashboard_output(self, task_id, task_name, step_num, object_id, object_name):

        return {
            "object_id": object_id,
            "task_id": task_id,
            "task_name": task_name,
            "object_name": object_name,
            "step_num": int(step_num),
        }

    def identify_task_step(self, object_name, pred_step_num):
        step_num = None
        task_name = None

        if object_name == "bowl":
            step_num = pred_step_num
            task_name = "oatmeal"
        elif object_name == "tortilla":
            if pred_step_num > TOTAL_STEPS_PINWHEELS:
                step_num = pred_step_num - TOTAL_STEPS_PINWHEELS
                task_name = "quesadilla"
            else:
                step_num = pred_step_num
                task_name = "pinwheels"
        elif object_name == "mug":
            if pred_step_num > TOTAL_STEPS_TEA:
                step_num = pred_step_num - TOTAL_STEPS_TEA
                task_name = "coffee"
            else:
                step_num = pred_step_num
                task_name = "tea"

        self.current_task_name = task_name
        self.current_step = step_num

    def get_task_id(self, object_id):
        for task_id, task in self.tasks.items():
            if object_id in task.get_object_ids():
                self.tasks[task_id].current_step = self.current_step
                return task_id

            elif self.current_task_name == task.get_task_name() and self.current_step == task.get_current_step():
                self.tasks[task_id].current_step = self.current_step
                self.tasks[task_id].object_ids.add(object_id)  # Add the new object id
                return task_id

        new_task = Task(object_id)
        self.curr_task_id += 1
        self.tasks[self.curr_task_id] = new_task
        self.tasks[self.curr_task_id].task_name = self.current_task_name
        self.tasks[self.curr_task_id].current_step = self.current_step

        return self.curr_task_id

    def run(self, object_name, object_id, data: list):
        model = self.load_model(object_name)
        if model is None:
            return {}

        pred_step_num = model.predict(np.array([data]))[0]
        #pred_proba = model.predict_proba(np.array([data]))
        # print('>>>> step', pred_step_num)
        # print('>>>> proba', pred_proba)
        # if pred_proba.max() <= 0.4:
        #    return {}

        self.identify_task_step(object_name, pred_step_num)

        task_id = self.get_task_id(object_id=object_id)

        return self.create_dashboard_output(
            task_id, self.current_task_name, self.current_step, object_id, object_name
        )

    def run_message(self, message, entire_message):
        message_output = self.mc.convert_message(message, entire_message)
        if message_output is None:
            return {}
        object_id = message_output["id"]
        object_name = message_output["object_name"]
        data = message_output["states"]
        if object_id in self.object_info:
            self.object_info[object_id].pop(0)
            self.object_info[object_id].append(data)
        else:
            self.object_info[object_id] = [data[:] for _ in range(self.window_size)]
        # Average the data now
        # Transpose the list of lists to work with columns
        transposed = list(map(list, zip(*self.object_info[object_id])))

        # Calculate the average for each column
        averages = [sum(col) / len(col) for col in transposed]

        return self.run(object_name, object_id, averages)
