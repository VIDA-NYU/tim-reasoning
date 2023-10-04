import unittest
from unittest.mock import patch

from tim_reasoning import DependencyGraph, Json2GraphConverter
from tim_reasoning.pddl2graph.node import Node


class TestJson2GraphConverter(unittest.TestCase):
    def setUp(self):
        self.converter = Json2GraphConverter()
        self.recipe = "tea"  # sample recipe to test
        self.nodes_len = 5  # total number of nodes in tea recipe
        self.total_steps = 7
        self.recipe_folder = "data/recipe"
        self.instructions_folder = 'data/step_goals'
        self.json_dir = f"{self.instructions_folder}/{self.recipe}"

    def test_create_nodes(self):
        # Test creating nodes from parsed JSON step
        parsed_step = {'goals': [{'state': 'chopped', 'objects': ['onion']}]}
        nodes = self.converter._create_nodes(parsed_step, 1)
        self.assertEqual(len(nodes), 1)
        self.assertIsInstance(nodes[0], Node)
        self.assertEqual(nodes[0].state, 'chopped')
        self.assertEqual(nodes[0].objects, ['onion'])
        self.assertEqual(nodes[0].step_number, 1)

    def test_add_previous_dependencies(self):
        # Test adding dependencies from previous nodes
        graph = DependencyGraph()
        prev_nodes = [
            Node('measured', ['water'], 1),
            Node('contains', ['kettle', 'water'], 1),
        ]
        graph.add_nodes(prev_nodes)
        cur_nodes = [
            Node('contains', ['mug', 'tea-bag'], 2),
            Node('steeped', ['tea-bag'], 5),
        ]
        graph.add_nodes(cur_nodes)

        self.converter._add_previous_dependencies(prev_nodes, cur_nodes)

        self.assertIn(prev_nodes[0], cur_nodes[0].dependencies)
        self.assertIn(prev_nodes[1], cur_nodes[0].dependencies)
        self.assertIn(prev_nodes[0], cur_nodes[1].dependencies)
        self.assertIn(prev_nodes[1], cur_nodes[1].dependencies)

    def test_get_recipe_length(self):
        # Test getting number of steps from recipe JSON
        num_steps = self.converter._get_recipe_length(
            self.recipe_folder, self.recipe
        )
        self.assertEqual(num_steps, self.total_steps)

    def test_generate_graph(self):
        # Test generating full dependency graph
        self.converter._generate_graph(
            json_dir=self.json_dir,
            total_steps=self.total_steps,
        )

        self.assertIsInstance(self.converter.graph, DependencyGraph)
        self.assertEqual(len(self.converter.graph.nodes), self.nodes_len)

    def test_convert(self):
        # End-to-end test
        converter = Json2GraphConverter()
        graph = converter.convert(
            self.recipe, self.instructions_folder, self.recipe_folder
        )

        self.assertIsInstance(graph, DependencyGraph)
        self.assertEqual(len(graph.nodes), self.nodes_len)


if __name__ == '__main__':
    unittest.main()
