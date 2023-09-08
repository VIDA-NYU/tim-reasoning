"""
Each step can have multiple goals and when they have
AND operand, they can 2 goals. 
So, we assume goals can be list of list of object states
"""
from unified_planning.model.fnode import FNode
from unified_planning.model.problem import Problem


class ProblemParser:
    def __init__(self) -> None:
        pass

    def parse_goals(self, problem: Problem) -> list:
        """Parse goals for each step

        Args:
            problem (Problem): unified_planning parsed problem

        Returns:
            list: returns list of tuples of goals (that have actions and objects)
        """

        goals = []
        for goal in problem.goals:
            parsed = self._parse_goal(goal)
            goal_dict = self._parsed_to_dict(parsed)
            goals.append(goal_dict)
        return goals

    def _parsed_to_dict(self, parsed):
        return {
            "operand": parsed[0],
            "state": parsed[1],  # list
            "objects": parsed[2],  # list
        }

    def _parse_goal(self, goal: FNode) -> tuple:
        """Parse an individual goal.

        Args:
            goal (FNode): The goal to parse.

        Returns:
            tuple: The parsed goal tuple.
        """

        if goal.is_and():
            return ('AND', *self._parse_args(goal.args))
        else:
            return ('NONE', *self._parse_single(goal))

    def _parse_args(self, args) -> tuple:
        """Parse goals that are AND expressions.

        Args:
            args (list): The AND expression arguments.

        Returns:
            tuple: The parsed actions and objects.
        """

        actions, objects_list = [], []
        for arg in args:
            if arg.is_not():
                arg_prime = arg.arg(0)
                action = arg_prime.fluent().name
                # append to actions
                actions.append(f"not-{action}")

                # Get the objects
                objs = self._get_objects(arg_prime)
                objects_list.append(objs)

            elif arg.is_fluent_exp():
                action = arg.fluent().name
                # append to actions
                actions.append(action)

                # Get the objects
                objs = self._get_objects(arg)
                objects_list.append(objs)
        return actions, objects_list

    def _parse_single(self, goal: FNode) -> tuple:
        """Parse a single goal expression.

        Args:
            goal (FNode): The single goal expression

        Returns:
            tuple: The parsed action and objects
        """

        if goal.is_fluent_exp():
            action = goal.fluent().name
            return [action], [self._get_objects(goal)]
        return [], []

    def _get_objects(self, goal: FNode) -> list:
        """Get objects for a goal expression.

        Args:
            goal (FNode): The goal expression.

        Returns:
            list: The list of object names.
        """
        objs = []
        for a in goal.args:
            if a.is_object_exp():
                objs.append(a.object().name)
        return objs
