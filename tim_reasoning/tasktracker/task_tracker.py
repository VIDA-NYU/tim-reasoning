from tim_reasoning import Pddl2GraphConverter, Json2GraphConverter
from tim_reasoning.pddl2graph.node import Node
from tim_reasoning.reasoning_errors import ReasoningErrors

PDDL_DATA_FOLDER = "data/pddl/gpt-generated"
RECIPE_DATA_FOLDER = "data/recipe"


class TaskTracker:
    """Class for task manager that track a specific recipe"""

    def __init__(
        self,
        recipe: str,
        data_folder: str = PDDL_DATA_FOLDER,
        if_json_converter: bool = True,
        verbose: str = False,
    ):
        self.recipe = recipe
        self.task_graph = self.setup_task_graph(
            recipe,
            data_folder,
            RECIPE_DATA_FOLDER,
            if_json_converter,
            verbose,
        )
        self.completed_nodes = {}
        self.current_step_number = 0

    def setup_task_graph(
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
        import json

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _update_step_number(self, node_step: int):
        """Update the step number of task graph

        Args:
            node_step (int): last node added's step_number
        """
        self.current_step_number = max(self.current_step_number, node_step)

    def get_current_step_number(self) -> int or ReasoningErrors:
        """Returns Task graph's current step number

        Returns:
            int or ReasoningErrors: current step number
        """
        if self.current_step_number == 0:
            return ReasoningErrors.NOT_STARTED
        return self.current_step_number

    def get_next_recipe_step(
        self,
        recipe_file_name: str = "recipe.json",
        recipe_folder: str = RECIPE_DATA_FOLDER,
    ) -> str or None:
        """Returns the next recipe step instructions

        Args:
            recipe_file_name (str, optional): recipe json file name. Defaults to "recipe.json".
            recipe_folder (str, optional):
                recipe json folder location. Defaults to RECIPE_DATA_FOLDER.

        Returns:
            str or None: next step if exists else None
        """
        # get json instructions for current recipe
        json_data = self._read_json(json_file=f"{recipe_folder}/{recipe_file_name}")
        instructions = json_data[self.recipe]["steps"]

        # next step number
        step_num = str(self.current_step_number + 1)
        # if next instruction exists
        if step_num in instructions:
            return instructions[step_num]
        # else if its last step
        return None

    def track(self, state: str, objects: list) -> ReasoningErrors or None:
        """Track the steps using task graph, add to completed list and raise errors

        Args:
            state (str): current state
            objects (list): current objects

        Returns:
            ReasoningErrors or None: errors or none
        """
        node_id, node = self.task_graph.find_node(state=state, objects=objects)
        if not node:
            return ReasoningErrors.INVALID_STATE

        # check_dependencies
        if not self._is_dependencies_completed(node=node):
            # raise error
            return ReasoningErrors.MISSING_PREVIOUS

        # Add to completed nodes
        self.completed_nodes[node_id] = node

        # Update step number
        self._update_step_number(node_step=node.step_number)
        # return none when no errors found to keep on going
        return None
