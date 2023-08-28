import unittest
from unittest.mock import patch

from tim_reasoning import Pddl2GraphConverter
from tim_reasoning import DependencyGraph

class TestPddl2GraphConverter(unittest.TestCase):

    def setUp(self):
        self.converter = Pddl2GraphConverter()

    def test_create_2_nodes(self):
        parsed = {
            'state': ['measured', 'contains'],
            'objects': ['water', ['mug', 'honey']]
        }
        nodes = self.converter._create_2_nodes(parsed)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].state, 'measured')
        self.assertEqual(nodes[1].state, 'contains')

    def test_create_single_node(self):
        parsed = {
            'state': ['measured'],
            'objects': [['a', 'b']]
        }
        nodes = self.converter._create_single_node(parsed)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].state, 'measured')
        self.assertEqual(nodes[0].objects, ['a', 'b'])

    def test_create_step_nodes(self):
        parsed = {
            'operand': 'AND', 
            'state': ['measured', 'contains'],
            'objects': ['water', ['mug', 'honey']]
        }
        nodes = self.converter._create_step_nodes(parsed)
        self.assertEqual(len(nodes), 2)

        parsed = {
            'operand': 'NONE',
            'state': ['measured'], 
            'objects': ['a']
        }
        nodes = self.converter._create_step_nodes(parsed)
        self.assertEqual(len(nodes), 1)

    def test_goals_to_nodes(self):
        goals = [
            {'operand': 'AND', 'state': ['measured', 'contains'],
             'objects': ['water', ['mug', 'honey']]
            },
            {'operand': 'NONE', 'state': ['measured'], 'objects': ['a']}
        ]
        nodes = self.converter._goals_to_nodes(goals)
        self.assertEqual(len(nodes), 3)

if __name__ == '__main__':
    unittest.main()
