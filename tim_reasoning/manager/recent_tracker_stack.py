class RecentTrackerStack:
    def __init__(self):
        self.items = []

    def push(self, tracker_id: int):
        if tracker_id in self.items:
            self.items.remove(tracker_id)
        self.items.append(tracker_id)

    def pop(self):
        tracker_id = self.items.pop()
        return tracker_id

    def remove(self, tracker_id):
        if tracker_id in self.items:
            self.items.remove(tracker_id)

    def get_recent(self):
        """Returns the most recent tracker tracked i.e. top of stack"""
        return self.items[-1]
