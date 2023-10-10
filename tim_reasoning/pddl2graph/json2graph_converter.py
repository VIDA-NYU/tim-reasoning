import json
import os

from tim_reasoning import DependencyGraph, Logger
from tim_reasoning.pddl2graph.node import Node


class Json2GraphConverter:
    def __init__(self):
        self.graph = DependencyGraph()
        self.log = Logger(name="Json2GraphConverter")

    def _add_previous_dependencies(self, previous_nodes, current_nodes):
        for node in current_nodes:
            node.add_dependencies(previous_nodes)

    def _create_nodes(self, parsed_step, step_number):
        nodes = []
        for goal in parsed_step['goals']:
            node = Node(
                state=goal['state'], objects=goal['objects'], step_number=step_number
            )
            nodes.append(node)
        return nodes

    def _get_recipe_length(self, recipe_data_folder, recipe_name) -> int:
        with open(f'{recipe_data_folder}/recipe.json', encoding="utf-8") as file:
            recipe_data = json.load(file)
        steps = recipe_data[recipe_name]['steps']
        return len(steps)

    def _generate_graph(
        self, json_dir: str, total_steps: int, verbose: bool = False
    ):
        for step_number in range(1, total_steps + 1):
            step_path = f'{json_dir}/step{step_number}.json'
            if not os.path.exists(step_path):
                if verbose:
                    self.log.info(
                        f"JSON step file doesn't exist for Step {step_number}"
                    )
                continue
            with open(step_path, encoding="utf-8") as f:
                data = f.read()
            steps = json.loads(data)

            if verbose:
                self.log.info(f"Parsing step {step_number}")

            nodes = self._create_nodes(steps, step_number)
            self.graph.add_nodes(nodes)

            if step_number > 1:
                self._add_previous_dependencies(previous_nodes, nodes)

            previous_nodes = nodes

    def convert(
        self,
        recipe: str,
        instructions_folder: str = 'data/step_goals',
        recipe_data_folder: str = 'data/recipe',
        verbose=False,
    ):
        TOTAL_STEPS = self._get_recipe_length(
            recipe_data_folder=recipe_data_folder, recipe_name=recipe
        )
        assert TOTAL_STEPS > 0
        json_dir = f"{instructions_folder}/{recipe}"
        self._generate_graph(
            json_dir=json_dir, total_steps=TOTAL_STEPS, verbose=verbose
        )

        return self.graph
