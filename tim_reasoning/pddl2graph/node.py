class Node:
    _id = 0

    def __init__(self, state: str, objects: list) -> None:
        self._id = Node._id
        Node._id += 1
        self.state = state
        self.objects = objects
        self.dependencies = []

    def get_id(self) -> int:
        """Return Node ID

        Returns:
            int: Node ID
        """
        return self._id

    def add_dependency(self, node):
        """Add dependency to current node

        Args:
            node (Node): the node on which current node is dependent
        """
        self.dependencies.append(node)

    def add_dependencies(self, nodes: list):
        """Add multiple dependencies

        Args:
            nodes (list): list of nodes on which current node is dependent
        """
        for node in nodes:
            self.add_dependency(node)


# Notes from reviewers
# duplicate objects
# connection to tracked objects

# -----
# action recognition is done by mainly object states
# - object states
# detecting actions is unstable
