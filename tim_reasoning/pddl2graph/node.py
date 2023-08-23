class Node:
    _id = 0

    def __init__(self, state: str, objects: list) -> None:
        self._id = Node._id
        Node._id += 1
        self.state = state
        self.objects = objects
        self.dependencies = []
    
    def add_dependency(self, node):
        self.dependencies.append(node)
    
    def add_dependencies(self, nodes):
        for node in nodes:
            self.add_dependency(node)


# Notes from reviewers
# duplicate objects
# connection to tracked objects 

# -----
# action recognition is done by mainly object states
# - object states
# detecting actions is unstable
