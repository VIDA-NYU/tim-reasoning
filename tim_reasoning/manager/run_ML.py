import joblib
import numpy as np

from tim_reasoning import MessageConverter
from os.path import join, dirname

TOTAL_STEPS_PINWHEELS = 12
TOTAL_STEPS_TEA = 7


class RunML:
    def __init__(self) -> None:
        self.tasks = {}
        self.curr_task_id = 0
        self.mc = MessageConverter()

    def load_model(self, object_name):
        loaded_rf = None
        if object_name == "bowl":
            model_location = join(dirname(__file__), "../../oatmeal_rf.joblib")
            loaded_rf = joblib.load(model_location)
        elif object_name == "tortilla":
            model_location = join(dirname(__file__), "../../pinwheels_quesadilla_rf.joblib")
            loaded_rf = joblib.load(model_location)
        elif object_name == "mug":
            model_location = join(dirname(__file__), "../../tea_coffee_rf.joblib")
            loaded_rf = joblib.load(model_location)
        return loaded_rf

    def create_dashboard_output(
        self, object_id, pred_step_num, object_name, task_id
    ):
        step_num = None
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

        return {
            "object_id": object_id,
            "task_id": task_id,
            "task_name": task_name,
            "object_name": object_name,
            "step_num": int(step_num),
        }

    def get_task_id(self, object_id):
        if object_id in self.tasks:
            task_id = self.tasks[object_id]
        else:
            self.curr_task_id += 1
            self.tasks[object_id] = self.curr_task_id
            task_id = self.tasks[object_id]
        return task_id

    def run(self, object_name, object_id, data: list):
        model = self.load_model(object_name)
        if model == None: return {}
        task_id = self.get_task_id(object_id=object_id)
        pred_step_num = model.predict(np.array([data]))[0]
        return self.create_dashboard_output(
            object_id, pred_step_num, object_name, task_id
        )

    def run_message(self, message):
        message_output = self.mc.convert_message(message=message)
        if message_output == None: return {}
        object_id = message_output["id"]
        object_name = message_output["object_name"]
        data = message_output["states"]
        return self.run(object_name, object_id, data)
