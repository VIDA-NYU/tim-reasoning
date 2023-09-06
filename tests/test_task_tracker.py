import unittest
from unittest.mock import MagicMock

from tim_reasoning.pddl2graph.node import Node
from tim_reasoning.reasoning_errors import ReasoningErrors
from tim_reasoning import TaskTracker


class TestTaskTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = TaskTracker(recipe='tea')
        self.node1 = Node('measured', ['water'], 1)
        self.node2 = Node('contains', ['kettle', 'water'], 1)
        self.node3 = Node('contains', ['mug', 'tea-bag'], 2)
        self.node_future = Node('steeped', ['tea-bag'], 5)

    def test_is_dependencies_completed_true(self):
        self.node2.add_dependency(self.node1)
        self.tracker.completed_nodes = [self.node1]
        result = self.tracker._is_dependencies_completed(self.node2)
        self.assertTrue(result)

    def test_is_dependencies_completed_false(self):
        self.node2.add_dependency(self.node1)
        self.tracker.completed_nodes = []
        result = self.tracker._is_dependencies_completed(self.node2)
        self.assertFalse(result)

    def test_track_invalid_state(self):
        result = self.tracker.track(state='non-existent', objects=['non-existent'])
        self.assertEqual(result, ReasoningErrors.INVALID_STATE)

    def test_track_missing_dependency(self):
        self.tracker.completed_nodes = []
        self.tracker.completed_nodes.append(self.node1)
        self.tracker.completed_nodes.append(self.node2)
        result = self.tracker.track(state='steeped', objects=['tea-bag'])
        self.assertEqual(result, ReasoningErrors.MISSING_PREVIOUS)

    def test_track_success(self):
        self.tracker.completed_nodes = []
        self.tracker.completed_nodes.append(self.node1)
        self.tracker.completed_nodes.append(self.node2)

        self.tracker.task_graph.find_node = MagicMock(return_value=(3, self.node3))
        result = self.tracker.track('contains', ['mug', 'tea-bag'])
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
