import json

from collections import defaultdict
from tim_reasoning import Logger, TaskTracker
from tim_reasoning.reasoning_errors import ReasoningErrors


class SessionManager:
    def __init__(
        self,
        unique_objects_file: str = "data/step_goals/unique_objects.json",
        common_objects_file: str = "data/step_goals/common_objects.json",
        data_folder: str = "data/step_goals/",
        patience: int = 1,
        verbose: bool = True,
    ) -> None:
        self.task_trackers = []
        self.patience = patience
        self.object_states = defaultdict(lambda: defaultdict(list))
        self.unique_objects_file = unique_objects_file
        self.common_objects_file = common_objects_file
        self.data_folder = data_folder
        self.verbose = verbose
        self.log = Logger(name="SessionManager")
        if self.verbose:
            self.log.info("Session Manager initiated.")
        (
            self.important_objects,
            self.common_objects,
        ) = self.get_unique_common_objects()
        self.last_task_tracker_tracked_id = None

    def get_last_task_tracker_tracked_id(self) -> int:
        """Return the id of the most recent (ongoing) task tracker

        Returns:
            int: recent task_tracker id
        """
        return self.last_task_tracker_tracked_id

    def get_task_tracker(self, task_tracker_id: int) -> TaskTracker:
        """Returns the task_tracker for the given id

        Args:
            task_tracker_id (int): id of task_tracker

        Returns:
            TaskTracker: task_tracker
        """
        return next(
            (tt for tt in self.task_trackers if tt.get_id() == task_tracker_id), None
        )

    def get_unique_common_objects(self) -> tuple:
        """Returns unique and common objects that need to be tracked

        Returns:
            tuple: unique_objects, common_objects
        """
        with open(self.unique_objects_file) as f:
            unique_objects = json.load(f)
        with open(self.common_objects_file) as f:
            common_objects = json.load(f)
        return unique_objects, common_objects

    def remove_task_tracker(self, wrong_tracker: TaskTracker):
        """Deletes a Task Tracker from memory when provided

        Args:
            wrong_tracker (TaskTracker): task tracker object to be removed from memory
        """
        task_tracker_id = wrong_tracker.get_id()
        if self.verbose:
            self.log.info(
                f"For `{wrong_tracker.recipe}` Recipe with task id {task_tracker_id},"
                f"got an error, hence, this recipe is not possible, deleting it from memory.\n"
            )
        remove_idx = next(
            (
                i
                for i, task_tracker in enumerate(self.task_trackers)
                if task_tracker.get_id() == task_tracker_id
            ),
            None,
        )
        if remove_idx is None:
            self.log.error("Unknown task tracker id to be removed.")
        else:
            self.task_trackers.pop(remove_idx)

    def update_last_tracker(self, tracker: TaskTracker):
        """Updates the most recent tracker's id

        Args:
            tracker (TaskTracker): task_tracker object just tracked
        """
        self.last_task_tracker_tracked_id = tracker.get_id()

    def create_probable_task_trackers(self, object_id, object_label) -> list:
        """For a given unique object, create probable task trackers

        Args:
            object_id (_type_): object id
            object_label (_type_): object label

        Returns:
            list: list of task_trackers
        """
        with open(self.unique_objects_file) as f:
            recipes = json.load(f)[object_label]
        task_trackers = [
            TaskTracker(
                recipe=recipe,
                data_folder=self.data_folder,
                if_json_converter=True,
            )
            for recipe in recipes
        ]
        for new_task in task_trackers:
            new_task.object_ids.append(object_id)
            new_task.object_labels.append(object_label)
        return task_trackers

    def find_task_tracker(self, object_id, object_label) -> list:
        """Finds or create task_tracker for a given object presence

        Args:
            object_id (_type_): object's id
            object_label (_type_): object's name

        Returns:
            list: list of task_trackers this object is in.
        """
        probable_task_trackers = [
            t
            for t in self.task_trackers
            if object_label in t.object_labels and object_id in t.object_ids
        ]
        # else tasktracker not found hence create a new tasktracker
        if not probable_task_trackers:
            probable_task_trackers = self.create_probable_task_trackers(
                object_id=object_id, object_label=object_label
            )
            if self.verbose:
                self.log.info(
                    f"Created {len(probable_task_trackers)} TaskTrackers for {object_label}_{object_id}"
                )
            # Add newly created task trackers to self.task_trackers
            self.task_trackers.extend(probable_task_trackers)
        return probable_task_trackers

    def track_common_object(self, state, object_id, object_label):
        if self.last_task_tracker_tracked_id is not None:
            last_tracker = self.get_task_tracker(self.last_task_tracker_tracked_id)
            instruction = last_tracker.track(
                state=state, objects=[object_label], object_ids=[object_id]
            )
            self.update_last_tracker(last_tracker)
            return instruction
        return None

    def track_unique_object(self, state, object_id, object_label):
        probable_task_trackers = self.find_task_tracker(object_id, object_label)
        if self.verbose:
            self.log.info(
                f"Found {len(probable_task_trackers)} TaskTrackers for {object_label}_{object_id}"
            )
        # if we are sure there's only one TaskTracker
        if len(probable_task_trackers) == 1:
            task_tracker = probable_task_trackers[0]
            instruction = task_tracker.track(
                state=state, objects=[object_label], object_ids=[object_id]
            )
            self.update_last_tracker(task_tracker)
            return instruction
        # multipe probable task_graphs possible
        else:
            if self.verbose:
                self.log.info("Multipe probable task_graphs possible")
            instructions = []
            for i, task_tracker in enumerate(probable_task_trackers):
                instruction = task_tracker.track(state, [object_label], [object_id])
                self.update_last_tracker(task_tracker)
                # if there's an error in instruction, it means we can drop this task_graph
                if isinstance(instruction, ReasoningErrors):
                    if instruction == ReasoningErrors.PARTIAL_STATE:
                        self.log.info(
                            f"For `{task_tracker.recipe}` Recipe, received partial state completion."
                        )
                        instructions.append(task_tracker.get_next_recipe_step())
                    else:
                        self.remove_task_tracker(wrong_tracker=task_tracker)
                else:
                    if self.verbose:
                        self.log.info(
                            f"For `{task_tracker.recipe}` Recipe, next instruction = `{instruction}`"
                        )
                    instructions.append(instruction)
            return instructions

    def track_object(self, state, object_id, object_label):
        if object_label in self.important_objects:
            return self.track_unique_object(state, object_id, object_label)
        elif object_label in self.common_objects:
            return self.track_common_object(state, object_id, object_label)
        else:
            self.log.error("Unknown object found.")

    def process_object(self, obj):
        object_id = obj['id']
        object_label = obj['label']

        if "state" not in obj:
            return

        object_state = obj['state']

        if (
            object_label in self.important_objects
            or object_label in self.common_objects
        ):
            self.track_object_state(object_id, object_label, object_state)
        else:
            self.log.info(f"Not tracking : {object_label}")

    def track_object_state(self, object_id, object_label, object_state):
        for state, prob in object_state.items():
            self.object_states[object_id][state].append(prob)

        if len(self.object_states[object_id]) >= self.patience:
            avg_state = self.calculate_average_state(object_id, object_label)
            if self.verbose:
                self.log.info(
                    f"Object id: {object_id}, label: {object_label} with avg state={avg_state}"
                )
            instruction = self.track_object(
                state=avg_state,
                object_id=object_id,
                object_label=object_label,
            )
            self.handle_instruction(instruction)
            self.reset_object_states(object_id)

    def calculate_average_state(self, object_id, object_label):
        avg_states = {
            state: sum(probs) / len(probs)
            for state, probs in self.object_states[object_id].items()
        }
        return max(avg_states, key=avg_states.get)

    def handle_instruction(self, instruction):
        if isinstance(instruction, list):
            self.log.info(f"Final Next instructions: {instruction}")
        else:
            self.log.info(f"Final Next instruction: {instruction}")

    def reset_object_states(self, object_id):
        self.object_states[object_id] = defaultdict(list)

    def handle_message(self, message: list):
        for obj in message:
            self.process_object(obj)
