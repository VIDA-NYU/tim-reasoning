import os
from .dependency_graph import DependencyGraph
from .node import Node
from .problem_parser import ProblemParser
from glob import glob
from unified_planning.io import PDDLReader

class Pddl2GraphConverter:
    def __init__(self) -> None:
        self.graph = DependencyGraph()
        self.problem_parser = ProblemParser()
        self.reader = PDDLReader()

    def _create_2_nodes(self, parsed):
        nodes = []
        for i in range(len(parsed['state'])):
            nodes.append(
                Node(
                    state=parsed['state'][i],
                    objects=parsed['objects'][i]
                )
            )
        return nodes

    def _create_single_node(self, parsed):
        return [
            Node(
                    state=parsed['state'][0],
                    objects=parsed['objects'][0]
            )
        ]

    def _create_step_nodes(self, parsed):
        if parsed['operand'] == 'AND':
            nodes = self._create_2_nodes(parsed)
        else:
            nodes = self._create_single_node(parsed)
        return nodes

    def _goals_to_nodes(self, goals):
        step_nodes = []
        for goal in goals:
            nodes = self._create_step_nodes(goal)
            step_nodes.extend(nodes)
        return step_nodes

    def _add_previous_step_dependencies(self, previous_step_nodes, current_step_nodes):
        for curr_node in current_step_nodes:
            curr_node.add_dependencies(previous_step_nodes)

    def _parse_pddl(self,
                    domain_file: str,
                    total_steps: str,
                    pddl_folder: str,
                    verbose: bool = False
                    ):
        for step_count in range(1, total_steps + 1):
            print(f"Parsing for step {step_count}/{total_steps}")
            
            problem = self.reader.parse_problem(
                domain_filename=domain_file,
                problem_filename=f'{pddl_folder}/step{step_count}.pddl'
            )
            # parse all the goals for the cuyrrent step
            parsed_goals = self.problem_parser.parse_goals(problem=problem)
            if verbose:
                print(f"Parsed goals are = \n{parsed_goals}\n")
            
            # finding final object states for current step
            current_step_nodes = self._goals_to_nodes(parsed_goals)
            # add these to graph
            self.graph.add_nodes(current_step_nodes)
            
            # after traversing 2nd step, i would add dependency of step n to step n-1
            if step_count != 1 and previous_step_nodes:
                # Add current step nodes to 
                self._add_previous_step_dependencies(
                    previous_step_nodes=previous_step_nodes,
                    current_step_nodes=current_step_nodes
                )

            previous_step_nodes = current_step_nodes
        return self.graph

    def convert(self, pddl_folder: str):
        DOMAIN_FILE = f'{pddl_folder}/domain.pddl'
        assert os.path.exists(DOMAIN_FILE)
        TOTAL_STEPS = len(glob(f'{pddl_folder}/step*.pddl'))
        assert TOTAL_STEPS > 0
        
        # create the graph and add dependencies
        graph = self._parse_pddl(
            domain_file=DOMAIN_FILE,
            total_steps=TOTAL_STEPS,
            pddl_folder=pddl_folder)
        return graph
