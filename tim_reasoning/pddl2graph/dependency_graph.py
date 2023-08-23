from .node import Node
class DependencyGraph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node: Node):
        self.nodes[node._id] = node
    
    def add_nodes(self, nodes: list):
        for node in nodes:
            self.add_node(node)

    def get_dependencies(self, node_id):
        node = self.nodes[node_id]
        return node.dependencies

    def print_dependenciesz(self):
        # Track indegrees
        indegrees = {node_id: 0 for node_id in self.nodes}
        for node in self.nodes.values():
            for dep in node.dependencies:
                indegrees[dep._id] += 1

        # Start with nodes having 0 indegree
        queue = [node_id for node_id in self.nodes if indegrees[node_id] == 0]

        # Process nodes 
        while queue:
            node_id = queue.pop(0)
            node = self.nodes[node_id]

            print(f"Node {node_id} {node.state, node.objects}")
            # Print node info

            for dep in node.dependencies:
                indegrees[dep._id] -= 1
                if indegrees[dep._id] == 0:
                    queue.append(dep._id)
                
            print()