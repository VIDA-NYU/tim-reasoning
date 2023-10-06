import json
import os

from glob import glob
from unified_planning.io import PDDLReader
from tim_reasoning import Logger
from tim_reasoning.pddl2graph.dependency_graph import DependencyGraph
from tim_reasoning.pddl2graph.node import Node
from tim_reasoning.pddl2graph.problem_parser import ProblemParser


class Pddl2GraphConverter:
    def __init__(self) -> None:
        self.graph = DependencyGraph()
        self.problem_parser = ProblemParser()
        self.reader = PDDLReader()
        self.log = Logger(name="Pddl2GraphConverter")

    def _create_2_nodes(self, parsed, step_number):
        nodes = []
        for i in range(len(parsed['state'])):
            nodes.append(
                Node(
                    state=parsed['state'][i],
                    objects=parsed['objects'][i],
                    step_number=step_number,
                )
            )
        return nodes

    def _create_single_node(self, parsed, step_number):
        return [
            Node(
                state=parsed['state'][0],
                objects=parsed['objects'][0],
                step_number=step_number,
            )
        ]

    def _create_step_nodes(self, parsed, step_number: int):
        if parsed['operand'] == 'AND':
            nodes = self._create_2_nodes(parsed, step_number)
        else:
            nodes = self._create_single_node(parsed, step_number)
        return nodes

    def _get_recipe_length(self, recipe_data_folder, recipe_name) -> int:
        with open(f'{recipe_data_folder}/recipe.json', encoding="utf-8") as file:
            recipe_data = json.load(file)
        steps = recipe_data[recipe_name]['steps']
        return len(steps)

    def _goals_to_nodes(self, goals: list, step_number: int):
        step_nodes = []
        for goal in goals:
            nodes = self._create_step_nodes(goal, step_number)
            step_nodes.extend(nodes)
        return step_nodes

    def _add_previous_step_dependencies(
        self, previous_step_nodes, current_step_nodes
    ):
        for curr_node in current_step_nodes:
            curr_node.add_dependencies(previous_step_nodes)

    def _parse_pddl(
        self,
        domain_file: str,
        total_steps: int,
        pddl_folder: str,
        verbose: bool = False,
    ):
        for step_count in range(1, total_steps + 1):
            if verbose:
                self.log.info(f"Parsing for step {step_count}/{total_steps}")

            pddl_step_file = f'{pddl_folder}/step{step_count}.pddl'
            if not os.path.exists(pddl_step_file):
                if verbose:
                    self.log.info(
                        f"PDDL step file doesn't exist for Step {step_count}"
                    )
                continue
            problem = self.reader.parse_problem(
                domain_filename=domain_file,
                problem_filename=pddl_step_file,
            )
            # parse all the goals for the current step
            parsed_goals = self.problem_parser.parse_goals(problem=problem)
            if verbose:
                self.log.info(f"Parsed goals are = \n{parsed_goals}\n")

            # finding final object states for current step
            current_step_nodes = self._goals_to_nodes(
                goals=parsed_goals, step_number=step_count
            )
            # add these to graph
            self.graph.add_nodes(current_step_nodes)

            # after traversing 2nd step, i would add dependency of step n to step n-1
            if step_count != 1 and previous_step_nodes:
                # Add current step nodes
                self._add_previous_step_dependencies(
                    previous_step_nodes=previous_step_nodes,
                    current_step_nodes=current_step_nodes,
                )

            previous_step_nodes = current_step_nodes

    def convert(
        self,
        recipe: str,
        pddl_folder: str,
        recipe_data_folder: str = 'data/recipe',
        verbose: bool = False,
    ) -> DependencyGraph:
        """Converts PDDL to a DependencyGraph

        Args:
            pddl_folder (str): Folder path containing all the pddl files.

        Returns:
            DependencyGraph: graph object
        """

        DOMAIN_FILE = f'{pddl_folder}/domain.pddl'
        assert os.path.exists(DOMAIN_FILE)
        TOTAL_STEPS = self._get_recipe_length(
            recipe_data_folder=recipe_data_folder, recipe_name=recipe
        )
        assert TOTAL_STEPS > 0
        # create the graph and add dependencies
        self._parse_pddl(
            domain_file=DOMAIN_FILE,
            total_steps=TOTAL_STEPS,
            pddl_folder=pddl_folder,
            verbose=verbose,
        )
        return self.graph