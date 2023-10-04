# from tim_reasoning import TaskTracker


# class SessionManager:
#     def __init__(self) -> None:
#         self.task_trackers = []
#         # the amount of frames to be used as patience
#         # before we actually call the tasktracker to track
#         # and average out the states
#         self.patience = 10

#     def handle_message(self, message: list):
#         # message can be of type list of dictionaries
#         # each dict item is a given object
#         # and has its labels, states, timestamps

import json

from collections import defaultdict
from tim_reasoning import TaskTracker
from tim_reasoning.reasoning_errors import ReasoningErrors


class SessionManager:
    def __init__(
        self,
        unique_objects_file: str = "data/step_goals/unique_objects.json",
        data_folder: str = "data/step_goals/",
        patience: int = 1,
        verbose: bool = True,
    ) -> None:
        self.task_trackers = []
        self.patience = patience
        self.object_states = defaultdict(lambda: defaultdict(list))
        self.unique_objects_file = unique_objects_file
        self.data_folder = data_folder
        self.verbose = verbose

    def create_probable_task_trackers(self, object_id, object_label) -> list:
        # self.task_trackers.append(TaskTracker(obj_label))
        task_trackers = []
        with open(self.unique_objects_file) as f:
            data = f.read()
        data = json.loads(data)
        recipes = data[object_label]
        for recipe in recipes:
            new_task = TaskTracker(
                recipe=recipe,
                data_folder=self.data_folder,
                if_json_converter=True,
            )
            new_task.object_ids.append(object_id)
            new_task.object_labels.append(object_label)
            task_trackers.append(new_task)
        return task_trackers

    def find_task_tracker(self, object_id, object_label):
        task_trackers = []
        for t in self.task_trackers:
            if object_label in t.object_labels and object_id in t.object_ids:
                task_trackers.append(t)
        # else tasktracker not found hence create a new tasktracker
        if not task_trackers:
            task_trackers = self.create_probable_task_trackers(
                object_id=object_id, object_label=object_label
            )
        return task_trackers

    def track_object(self, state, object_id, object_label):
        task_trackers = self.find_task_tracker(object_id, object_label)
        if self.verbose:
            print(f"Task Trackers = {task_trackers}")
        # if we are sure there's only one TaskTracker
        if len(task_trackers) == 1:
            task_tracker = task_trackers[0]
            instruction = task_tracker.track(
                state=state, objects=[object_label], object_ids=[object_id]
            )
            return instruction
        # multipe probable task_graphs possible
        else:
            instructions = []
            for i, task_tracker in enumerate(task_trackers):
                instruction = task_tracker.track(state, [object_label], [object_id])
                # if there's an error in instruction, it means we can drop this task_graph
                if isinstance(instruction, ReasoningErrors):
                    task_trackers.remove(i)
                else:
                    instructions.append(instruction)
            return instructions

    def handle_message(self, message: list):
        for obj in message:
            object_id = obj['id']
            object_label = obj['label']
            # no state, hence we can avoid this
            if "state" not in obj:
                continue
            object_state = obj['state']
            with open(self.unique_objects_file) as f:
                data = f.read()
            important_objects = json.loads(data)

            if object_label in important_objects:
                # Keep track of states over time
                for state, prob in object_state.items():
                    self.object_states[object_id][state].append(prob)

                # After patience reached, average and track
                if len(self.object_states[object_id]) >= self.patience:
                    avg_states = {
                        state: sum(probs) / len(probs)
                        for state, probs in self.object_states[object_id].items()
                    }
                    avg_state = max(avg_states, key=avg_states.get)

                    if self.verbose:
                        print(
                            f"Object id: {object_id}, label: {object_label} with avg state={avg_state}"
                        )
                    instruction = self.track_object(
                        state=avg_state,
                        object_id=object_id,
                        object_label=object_label,
                    )

                    if isinstance(instruction, list):
                        print(instruction)
                    else:
                        print(instruction)
                    # Reset obj state after tracking
                    self.object_states[object_id] = defaultdict(list)
            else:
                print(f"Not tracking : {object_label}")
