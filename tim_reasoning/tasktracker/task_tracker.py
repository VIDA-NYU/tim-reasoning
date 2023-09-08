from tim_reasoning.pddl2graph.node import Node
from tim_reasoning.pddl2graph.pddl2graph_converter import Pddl2GraphConverter
from tim_reasoning.reasoning_errors import ReasoningErrors

DATA_FOLDER = 'data/pddl/gpt-generated'


class TaskTracker:
    """Class for task manager that track a specific recipe"""

    def __init__(self, recipe: str, data_folder: str = DATA_FOLDER) -> None:
        self.pddl2graph_converter = Pddl2GraphConverter()
        self.task_graph = self.pddl2graph_converter.convert(
            pddl_folder=f'{data_folder}/{recipe}', verbose=False
        )
        self.completed_nodes = {}
        self.current_step_number = 0

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
