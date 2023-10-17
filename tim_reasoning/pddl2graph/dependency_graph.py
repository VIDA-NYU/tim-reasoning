from tim_reasoning import Logger
from tim_reasoning.pddl2graph.node import Node
import nltk

nltk.download('punkt')


class DependencyGraph:
    def __init__(self):
        self.nodes = {}
        self.log = Logger(name="DependencyGraph")

    def add_node(self, node: Node):
        node_id = node.get_id()
        self.nodes[node_id] = node

    def add_nodes(self, nodes: list):
        for node in nodes:
            self.add_node(node)

    def find_node(
        self, state: str, objects: list, visited_nodes: dict
    ) -> (int, Node):
        for node in self.nodes.values():
            if node.get_id() not in visited_nodes:
                if node.state == state and set(node.objects) == set(objects):
                    return node.get_id(), node
        return None, None

    def find_partial_node(self, state: str, objects: list) -> float:
        max_match = 0
        match_node_id = None
        match_node = None
        state_tokens = set(nltk.word_tokenize(state))
        state_tokens = set([token for token in state_tokens if len(token) > 1])
        for node in self.nodes.values():
            if set(node.objects) == set(objects):
                # Calculate percentage of state string that matches
                node_tokens = set(nltk.word_tokenize(node.state))
                node_tokens = set([token for token in node_tokens if len(token) > 1])

                common_tokens = state_tokens.intersection(node_tokens)
                common_tokens = [token for token in common_tokens if len(token) > 1]
                # Calculate percentage of common tokens
                match_percent = len(common_tokens) / len(state_tokens)
                if match_percent > max_match:
                    max_match = match_percent
                    match_node_id = node.get_id()
                    match_node = node
        # return the max percent match, node_id, node
        return max_match, match_node_id, match_node

    def get_dependencies(self, node_id):
        node = self.nodes[node_id]
        return node.dependencies

    def print_dependencies(self):
        # Track indegrees
        indegrees = {node_id: 0 for node_id in self.nodes}
        for node in self.nodes.values():
            for dep in node.dependencies:
                dep_node_id = dep.get_id()
                indegrees[dep_node_id] += 1

        # Start with nodes having 0 indegree
        queue = [node_id for node_id in self.nodes if indegrees[node_id] == 0]

        # Process nodes
        while queue:
            node_id = queue.pop(0)
            node = self.nodes[node_id]

            self.log.info(f"Node {node_id} {node.state, node.objects}")
            # Print node info

            for dep in node.dependencies:
                dep_node_id = dep.get_id()
                indegrees[dep_node_id] -= 1
                if indegrees[dep_node_id] == 0:
                    queue.append(dep_node_id)
            self.log.info()
