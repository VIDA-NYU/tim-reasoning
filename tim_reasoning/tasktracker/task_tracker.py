from tim_reasoning.pddl2graph.node import Node
from tim_reasoning.pddl2graph.pddl2graph_converter import Pddl2GraphConverter
from tim_reasoning.reasoning_errors import ReasoningErrors

DATA_FOLDER = 'data/pddl/gpt-generated'


class TaskTracker:
    """Class for task manager that track a specific recipe"""

    def __init__(self, recipe: str) -> None:
        self.pddl2graph_converter = Pddl2GraphConverter()
        self.task_graph = self.pddl2graph_converter.convert(f'{DATA_FOLDER}/{recipe}')
        self.completed_nodes = []

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
            if dep not in self.completed_nodes:
                return False
        return True

    def track(self, state: str, objects: list) -> ReasoningErrors:
        """Track the steps using task graph and raise errors

        Args:
            state (str): current state
            objects (list): current objects

        Returns:
            ReasoningErrors: _description_
        """
        _, node = self.task_graph.find_node(state=state, objects=objects)
        if not node:
            return ReasoningErrors.INVALID_STATE

        # check_dependencies
        if not self._is_dependencies_completed(node=node):
            # raise error
            return ReasoningErrors.MISSING_PREVIOUS

        # append to completed node
        self.completed_nodes.append(node)
        # return none when no errors found to keep on going
        return None
