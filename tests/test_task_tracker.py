import unittest
from unittest.mock import MagicMock

from tim_reasoning.pddl2graph.node import Node
from tim_reasoning.reasoning_errors import ReasoningErrors
from tim_reasoning import TaskTracker


class TestTaskTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = TaskTracker(recipe='tea', if_json_converter=False)
        self.node1 = Node('measured', ['water'], 1)
        self.node2 = Node('contains', ['kettle', 'water'], 1)
        self.node3 = Node('contains', ['mug', 'tea-bag'], 2)
        self.node_future = Node('steeped', ['tea-bag'], 5)

    def test_is_dependencies_completed_true(self):
        self.node2.add_dependency(self.node1)
        node1_id = self.node1.get_id()
        self.tracker.completed_nodes[node1_id] = self.node1
        result = self.tracker._is_dependencies_completed(self.node2)
        self.assertTrue(result)

    def test_is_dependencies_completed_false(self):
        self.node2.add_dependency(self.node1)
        self.tracker.completed_nodes = {}
        result = self.tracker._is_dependencies_completed(self.node2)
        self.assertFalse(result)

    def test_track_invalid_state(self):
        result = self.tracker.track(
            state='non-existent', objects=['non-existent'], object_ids=[1]
        )
        self.assertEqual(result, ReasoningErrors.INVALID_STATE)

    def test_track_missing_dependency(self):
        self.tracker.completed_nodes = {}
        node1_id = self.node1.get_id()
        node2_id = self.node2.get_id()
        self.tracker.completed_nodes[node1_id] = self.node1
        self.tracker.completed_nodes[node2_id] = self.node2
        result = self.tracker.track(
            state='steeped', objects=['tea-bag'], object_ids=[1]
        )
        self.assertEqual(result, ReasoningErrors.MISSING_PREVIOUS)

    def test_track_success(self):
        self.tracker.completed_nodes = {}
        node1_id = self.node1.get_id()
        node2_id = self.node2.get_id()
        self.tracker.completed_nodes[node1_id] = self.node1
        self.tracker.completed_nodes[node2_id] = self.node2

        self.tracker.task_graph.find_node = MagicMock(return_value=(3, self.node3))
        result = self.tracker.track(
            'contains', ['mug', 'tea-bag'], object_ids=[1, 2]
        )
        self.assertEqual(result, "Check the temperature of the water.")

    def test_track_success_without_mock(self):
        self.tracker.track(self.node1.state, self.node1.objects, object_ids=[1])
        self.tracker.track(self.node2.state, self.node2.objects, object_ids=[2, 3])
        result = self.tracker.track(
            'contains', ['mug', 'tea-bag'], object_ids=[1, 2]
        )
        step_number = self.tracker.get_current_step_number()
        self.assertEqual(result, "Check the temperature of the water.")
        self.assertEqual(step_number, 2)

    def test_track_success_without_mock_with_json(self):
        self.node1 = Node('unstacked', ['mug'], 1)
        self.node2 = Node("tea-bag", ["mug"], 2)
        tracker = TaskTracker(
            recipe="tea",
            data_folder="data/step_goals/",
            if_json_converter=True,
            verbose=True,
        )
        tracker.track(self.node1.state, self.node1.objects, object_ids=[1])
        tracker.track(self.node2.state, self.node2.objects, object_ids=[1])
        result = tracker.track("liquid+tea-bag", ['mug'], object_ids=[1])
        step_number = tracker.get_current_step_number()
        self.assertEqual(result, 'Steep for 3 minutes.')
        self.assertEqual(step_number, 4)

    def test_get_current_step_number_success(self):
        self.tracker = TaskTracker(recipe='tea', if_json_converter=False)
        self.tracker.completed_nodes = {}
        self.tracker.track(self.node1.state, self.node1.objects, object_ids=[1])
        self.tracker.track(self.node2.state, self.node2.objects, object_ids=[2, 3])
        step_number = self.tracker.get_current_step_number()
        self.assertEqual(step_number, 1)

    def test_get_current_step_number_not_started(self):
        self.tracker.completed_nodes = {}
        step_number = self.tracker.get_current_step_number()
        self.assertEqual(step_number, ReasoningErrors.NOT_STARTED)

    def test_get_next_recipe_step_success(self):
        self.tracker.current_step_number = 1
        next_step = self.tracker.get_next_recipe_step()
        self.assertEqual(next_step, "Place tea bag in mug.")


if __name__ == '__main__':
    unittest.main()
