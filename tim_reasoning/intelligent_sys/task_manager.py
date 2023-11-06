import json
import numpy as np
import sys
from os.path import join, dirname


RECIPE_DATA_FOLDER = join(dirname(__file__), "../../data/recipe")


class TaskManager:
    """Class for task manager that track a specific recipe"""

    _id = 0

    def __init__(self, task_name: str) -> None:
        self._id = TaskManager._id
        TaskManager._id += 1

        self.completed_steps = []
        self.task_name = task_name
        self.task_errors = {
            "missing": [],
            "reorder": [],
        }  # Track errors for the task tracker
        self.object_id = None
        self.object_label = None
        self.current_step_number = 1

    def get_current_step_number(self) -> int:
        """Returns Task graph's current step number

        Returns:
            int: current step number
        """
        min_missing_step = (
            min(self.task_errors["missing"])
            if self.task_errors["missing"]
            else sys.maxsize
        )
        self.current_step_number = min(self.current_step_number, min_missing_step)
        return self.current_step_number

    def _is_dependencies_completed(self, step_num: int):
        """Returns a boolean value indicating whether or not all of the dependencies are completed.

        Note: checking only one level down from current state,
        as it is sequential check, we dont need to recursive check at
        each step. If a future step occurs, we throw error, or all deps
        not covered we throw error.

        Args:
            step_num (int): the current step number track
        """
        if (step_num - 1) in self.completed_steps:
            return True
        else:
            missing_steps = list(np.arange(max(self.completed_steps), step_num))
            if missing_steps:
                self.task_errors["missing"].extend(missing_steps)
            self.task_errors["missing"] = list(set(self.task_errors["missing"]))
            return False

    def _read_json(self, json_file: str):
        """Read and return json data

        Args:
            json_file (str): json file location

        Returns:
            json object: data
        """
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def get_recipe(
        self,
        recipe_file_name: str = "recipe.json",
        recipe_folder: str = RECIPE_DATA_FOLDER,
    ):
        json_data = self._read_json(json_file=f"{recipe_folder}/{recipe_file_name}")
        instructions = json_data[self.task_name]["steps"]
        return instructions

    def get_recipe_length(
        self,
        recipe_file_name: str = "recipe.json",
        recipe_folder: str = RECIPE_DATA_FOLDER,
    ):
        json_data = self._read_json(json_file=f"{recipe_folder}/{recipe_file_name}")
        instructions = json_data[self.task_name]["steps"]
        total_steps = len(instructions)
        if total_steps > 0:
            return len(instructions)
        else:
            return -1

    def get_current_recipe_step(
        self,
        recipe_file_name: str = "recipe.json",
        recipe_folder: str = RECIPE_DATA_FOLDER,
    ) -> str:
        """Returns the next (current) recipe step instructions

        Args:
            recipe_file_name (str, optional): recipe json file name. Defaults to "recipe.json".
            recipe_folder (str, optional):
                recipe json folder location. Defaults to RECIPE_DATA_FOLDER.

        Returns:
            str: next step if exists
        """
        # get json instructions for current recipe
        instructions = self.get_recipe(recipe_file_name, recipe_folder)

        # next step number
        current_step = self.get_current_step_number()
        if isinstance(current_step, int):
            step_num = str(self.current_step_number)
        # if next instruction exists
        if step_num in instructions:
            return instructions[step_num]
        # else if its last step
        else:
            return "This recipe is complete."
