import json
import sys
from tim_reasoning import Pddl2GraphConverter, Json2GraphConverter
from tim_reasoning.pddl2graph.node import Node
from tim_reasoning.reasoning_errors import ReasoningErrors
from os.path import join, dirname


PDDL_DATA_FOLDER = join(dirname(__file__), "../../data/pddl/gpt-generated")
RECIPE_DATA_FOLDER = join(dirname(__file__), "../../data/recipe")


class TaskTracker:
    """Class for task manager that track a specific recipe"""

    _id = 0

    def __init__(
        self,
        recipe: str,
        data_folder: str = PDDL_DATA_FOLDER,
        if_json_converter: bool = True,
        verbose: str = False,
    ):
        self._id = TaskTracker._id
        TaskTracker._id += 1
        self.recipe = recipe
        self.task_graph = self._setup_task_graph(
            recipe,
            data_folder,
            RECIPE_DATA_FOLDER,
            if_json_converter,
            verbose,
        )
        self.completed_nodes = {}
        self.current_step_number = 1
        self.object_ids = []  # ids are unique
        self.object_labels = []  # object_labels can be duplicates
        self.completed = False  # whether the Task is completed or not
        self.task_errors = {
            "missing": [],
            "reorder": [],
        }  # Track errors for the task tracker
        self.object_positions = []

    def _is_dependencies_completed(self, node: Node):
        """Returns a boolean value indicating whether or not all of the dependencies are completed.

        Note: checking only one level down from current state,
        as it is sequential check, we dont need to recursive check at
        each step. If a future step occurs, we throw error, or all deps
        not covered we throw error.

        Args:
            node (Node): given node (currently tracked)
        """
        for dep in node.dependencies:
            dep_id = dep.get_id()
            if dep_id not in self.completed_nodes:
                return False
        return True

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

    def _update_step_number(self, node_step: int):
        """Update the step number of task graph

        Args:
            node_step (int): last node added's step_number
        """
        min_missing_step = (
            min(self.task_errors["missing"])
            if self.task_errors["missing"]
            else sys.maxsize
        )

        max_step = max(self.current_step_number, node_step + 1)
        self.current_step_number = min(max_step, min_missing_step)

    def _build_not_started_output_dict(self, current_step_num: int = 1):
        current_step_num = str(current_step_num)
        # get json instructions for current recipe
        instructions = self.get_recipe()

        if current_step_num in instructions:
            step_description = instructions[current_step_num]
        # else if its last step
        else:
            self.completed = True
            step_description = "This recipe is complete."
        return {
            "task_id": self.get_id(),
            "task_name": self.recipe,
            "step_id": int(current_step_num),
            "step_status": "NOT_STARTED",
            "step_description": step_description,
            "error_status": False,
            "error_description": "",
            "total_steps": self.get_recipe_length(),
            "object_ids": self.get_object_ids(),
            "object_labels": self.get_object_labels(),
        }

    def _build_output_dict(self, instruction):
        return {
            "task_id": self.get_id(),
            "task_name": self.recipe,
            "step_id": self.get_current_step_number(),
            "step_status": "IN_PROGRESS",
            "step_description": instruction,
            "error_status": False,
            "error_description": "",
            "total_steps": self.get_recipe_length(),
            "object_ids": self.get_object_ids(),
            "object_labels": self.get_object_labels(),
        }

    def _build_completed_output_dict(
        self, instruction: str = "This recipe is complete."
    ):
        return {
            "task_id": self.get_id(),
            "task_name": self.recipe,
            "step_id": self.get_current_step_number(),
            "step_status": "COMPLETED",
            "step_description": instruction,
            "error_status": False,
            "error_description": "",
            "total_steps": self.get_recipe_length(),
            "object_ids": self.get_object_ids(),
            "object_labels": self.get_object_labels(),
        }

    def _build_error_dict(self, error):
        return {
            "task_id": self.get_id(),
            "task_name": self.recipe,
            "step_id": self.get_current_step_number(),
            "step_status": "ERROR",
            "step_description": self.get_next_recipe_step(),
            "error_status": True,
            "error_description": str(error),
            "total_steps": self.get_recipe_length(),
            "object_ids": self.get_object_ids(),
            "object_labels": self.get_object_labels(),
        }

    def _get_current_missing_dependencies(self, node: Node, missing_deps: list):
        if node:
            for dep in node.dependencies:
                dep_id = dep.get_id()
                if dep_id not in self.completed_nodes:
                    missing_deps.append(dep.step_number)
                missing_deps = self._get_current_missing_dependencies(
                    dep, missing_deps
                )
        return missing_deps

    def _setup_task_graph(
        self,
        recipe: str,
        instructions_folder: str,
        recipe_data_folder: str,
        if_json_converter: bool,
        verbose: bool,
    ):
        if if_json_converter:
            json2graph_converter = Json2GraphConverter()
            task_graph = json2graph_converter.convert(
                recipe=recipe,
                instructions_folder=instructions_folder,
                recipe_data_folder=recipe_data_folder,
                verbose=verbose,
            )
            return task_graph
        else:
            instructions_folder = f'{instructions_folder}/{recipe}'
            pddl2graph_converter = Pddl2GraphConverter()
            task_graph = pddl2graph_converter.convert(
                recipe=recipe,
                pddl_folder=instructions_folder,
                recipe_data_folder=recipe_data_folder,
                verbose=verbose,
            )
            return task_graph

    def _handle_track_missing_dependencies(self, node, node_id, objects, object_ids):
        # find all the missing dependencies
        missing_deps = self._get_current_missing_dependencies(node, [])
        missing_deps = set(missing_deps)
        already_missing = set(self.task_errors["missing"])
        # find the new missed errors
        new_missed = list(missing_deps - already_missing)
        if len(new_missed) > (self.current_step_number * 0.35):
            error = ReasoningErrors.FUTURE_STEP
            output = self._build_error_dict(
                error="Received future step way ahead in future."
            )
            return error, output
        # we need to add this to completed list, update step num
        self.add_completed_node(
            node=node,
            node_id=node_id,
            objects=objects,
            object_ids=object_ids,
        )
        # create the error output

        # update the missing errors
        self.task_errors["missing"] = list(missing_deps)
        # if there are no new missings steps, just output the current step
        if not new_missed:
            next_recipe_step = self.get_next_recipe_step()
            return next_recipe_step, self._build_output_dict(
                instruction=next_recipe_step
            )
        else:
            error = ReasoningErrors.MISSING_PREVIOUS
            error_output = self._build_error_dict(
                error=f"Missing step: {str(new_missed)[1:-1]}"
            )
            return (error, error_output)

    def get_id(self):
        return self._id

    def get_object_ids(self):
        """Returns the object ids in the current task graph"""
        return self.object_ids

    def get_object_labels(self):
        """Returns the object_labels in the current task graph"""
        return self.object_labels

    def get_task_errors(self) -> dict:
        """Returns the dictionary of errors so far

        Returns:
            dict: dictionary of errors
        """
        return self.task_errors

    def get_current_step_number(self) -> int or ReasoningErrors:
        """Returns Task graph's current step number

        Returns:
            int or ReasoningErrors: current step number
        """
        if self.current_step_number == 0:
            return 1
        return self.current_step_number

    def get_recipe(
        self,
        recipe_file_name: str = "recipe.json",
        recipe_folder: str = RECIPE_DATA_FOLDER,
    ):
        json_data = self._read_json(json_file=f"{recipe_folder}/{recipe_file_name}")
        instructions = json_data[self.recipe]["steps"]
        return instructions

    def get_recipe_length(
        self,
        recipe_file_name: str = "recipe.json",
        recipe_folder: str = RECIPE_DATA_FOLDER,
    ):
        json_data = self._read_json(json_file=f"{recipe_folder}/{recipe_file_name}")
        instructions = json_data[self.recipe]["steps"]
        total_steps = len(instructions)
        if total_steps > 0:
            return len(instructions)
        else:
            return -1

    def get_next_recipe_step(
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
        current_step = str(current_step)
        # if next instruction exists
        if current_step in instructions:
            return instructions[current_step]
        # else if its last step
        else:
            self.completed = True
            return "This recipe is complete."

    def get_current_instruction_output(self):
        """Returns the instruction output dict for current step without tracking"""
        current_step = self.get_current_step_number()

        if current_step == 1:
            return self._build_not_started_output_dict()

        instruction = self.get_next_recipe_step()

        if self.completed:
            return self._build_completed_output_dict()

        return self._build_output_dict(instruction)

    def add_completed_node(self, node, node_id, objects, object_ids):
        if node_id not in self.completed_nodes:
            # Add to completed nodes
            self.completed_nodes[node_id] = node

            # Update step number
            self._update_step_number(node_step=node.step_number)
            for object_id, object_label in zip(object_ids, objects):
                if object_id not in self.object_ids:
                    self.object_ids.append(object_id)
                    self.object_labels.append(object_label)

    def set_current_step(self, step_num: int):
        """Set current step number and mark previous as completed

        Args:
            step_num (int): The step number to set as current

        Raises:
            ValueError: If step_num is less than 1
        """
        if step_num < 1:
            raise ValueError("Step number must be greater than 0")

        self.current_step_number = step_num

        # Mark steps before current as completed
        for node_id, node in self.task_graph.nodes.items():
            if node.step_number < step_num:
                self.add_completed_node(node, node_id, [], [])

        next_recipe_step = self.get_next_recipe_step()
        return next_recipe_step, self._build_output_dict(
            instruction=next_recipe_step
        )

    def track(self, state: str, objects: list, object_ids: list):
        """Track the steps using task graph, add to completed list and raise errors

        Args:
            state (str): current state
            objects (list): current objects

        Returns:
            tuple
        """
        # Finds the the given object_label in the uncompleted nodes (regardless of the step_num)
        node_id, node = self.task_graph.find_node(
            state=state, objects=objects, visited_nodes=self.completed_nodes
        )
        if self.current_step_number > 0:
            # if object not found in unvisited steps
            if not node:
                # Partial match is either with COMPLETED nodes or FUTURE (partially) nodes
                (
                    max_match,
                    partial_node_id,
                    partial_node,
                ) = self.task_graph.find_partial_node(state=state, objects=objects)
                if max_match > 0.1:
                    # possibility of a future node (but incomplete)
                    # or possibility of a previously completed node
                    return (
                        ReasoningErrors.PARTIAL_STATE,
                        self._build_output_dict(
                            instruction=self.get_next_recipe_step()
                        ),
                    )
                else:
                    # This means that the state or object is totally new
                    error = ReasoningErrors.INVALID_STATE
                    return error, self._build_error_dict(
                        error=f"Unseen action or object might be detected"
                    )
            elif node.step_number < self.get_current_step_number():
                # this will only occur when the
                # previous step's node was incomplete and we found it
                if node.step_number in self.task_errors["missing"]:
                    # remove from missing errors
                    self.task_errors["missing"] = set(
                        self.task_errors["missing"]
                    ) - set([node.step_number])
                    # add to reorder
                    self.task_errors["reorder"].append(node.step_number)
                    (
                        instruction,
                        output,
                    ) = ReasoningErrors.REORDER_ERROR, self._build_error_dict(
                        error=f"Step {node.step_number} completed late."
                    )
                    return instruction, output
            # Otherwise if we found the node, check dependencies of the found node
            if not self._is_dependencies_completed(node=node):
                # check if the new step is not too far ahead in future
                # and update the missing nodes
                instruction, output = self._handle_track_missing_dependencies(
                    node=node,
                    node_id=node_id,
                    objects=objects,
                    object_ids=object_ids,
                )
                # update the current_step_number
                # self._update_step_number(node_step=node.step_number)
                return instruction, output
        if self.current_step_number == 0 and not node:
            # get the first recipe step
            instructions = self.get_recipe()
            instruction = instructions["1"]
            return instruction, self._build_not_started_output_dict(
                self.current_step_number
            )
        # if the task is not started, check dependency
        elif self.current_step_number == 0 and node:
            if not self._is_dependencies_completed(node=node):
                instruction, output = self._handle_track_missing_dependencies(
                    node=node,
                    node_id=node_id,
                    objects=objects,
                    object_ids=object_ids,
                )
                return instruction, output
        else:
            # even if the node is already completed the below function would not do anything
            self.add_completed_node(
                node=node, node_id=node_id, objects=objects, object_ids=object_ids
            )
        next_recipe_step = self.get_next_recipe_step()
        return next_recipe_step, self._build_output_dict(
            instruction=next_recipe_step
        )
