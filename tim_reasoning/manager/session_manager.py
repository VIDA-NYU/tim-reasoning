import json
from collections import defaultdict
from tim_reasoning import DemoLogger, Logger, RecentTrackerStack, TaskTracker
from tim_reasoning.reasoning_errors import ReasoningErrors
from os.path import join, dirname


class SessionManager:
    def __init__(
        self,
        unique_objects_file: str = join(
            dirname(__file__), "../../data/step_goals/unique_objects.json"
        ),
        common_objects_file: str = join(
            dirname(__file__), "../../data/step_goals/common_objects.json"
        ),
        data_folder: str = join(dirname(__file__), "../../data/step_goals/"),
        patience: int = 1,
        verbose: bool = True,
    ) -> None:
        self.task_trackers = []
        self.wrong_task_trackers = []
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
        self.recent_tracker_stack = RecentTrackerStack()
        self.demo_logger = DemoLogger('log.log')
        self.demo_logger.start_trial()
        # self.last_task_tracker_tracked_id = None

    def get_last_task_tracker_tracked_id(self) -> int:
        """Return the id of the most recent (ongoing) task tracker

        Returns:
            int: recent task_tracker id
        """
        return self.recent_tracker_stack.get_recent()

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

    def get_probable_task_trackers(self, object_id, object_label):
        probable_task_trackers = [
            t
            for t in self.task_trackers
            if object_label in t.object_labels and object_id in t.object_ids
        ]
        return probable_task_trackers

    def add_wrong_tracker(self, wrong_tracker):
        for tt in self.wrong_task_trackers:
            if tt.get_id() == wrong_tracker.get_id():
                return
        self.wrong_task_trackers.append(wrong_tracker)

    def remove_task_tracker(self, wrong_tracker: TaskTracker):
        """Deletes a Task Tracker from memory when provided

        Args:
            wrong_tracker (TaskTracker): task tracker object to be removed from memory
        """
        self.add_wrong_tracker(wrong_tracker)
        task_tracker_id = wrong_tracker.get_id()
        # remove the task_tracker from the recent tracker stack
        self.recent_tracker_stack.remove(task_tracker_id)

        if self.verbose:
            self.log.info(
                f"For `{wrong_tracker.recipe}` Recipe with task id {task_tracker_id},"
                f"got an error, hence, this recipe is not possible, deleting it from memory.\n"
            )
        # find the index where it exists and remove it
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

    def handle_wrong_tasks(
        self, probable_task_trackers: list, wrong_tracker_list: list
    ):
        """Removes the wrong tasks only if all of the probable ones were not wrong

        Args:
            probable_task_trackers (list): for given object, possible task_trackers
            wrong_tracker_list (list): errorneous task trackers
        """
        if len(probable_task_trackers) != len(wrong_tracker_list) or set(
            probable_task_trackers
        ) != set(wrong_tracker_list):
            for tt in wrong_tracker_list:
                self.remove_task_tracker(wrong_tracker=tt)

    def update_last_tracker(self, tracker: TaskTracker):
        """Updates the most recent tracker's id

        Args:
            tracker (TaskTracker): task_tracker object just tracked
        """
        self.recent_tracker_stack.push(tracker.get_id())

    def create_new_task_tracker(self, object_id, object_label, recipe):
        new_task = TaskTracker(
            recipe=recipe,
            data_folder=self.data_folder,
            if_json_converter=True,
        )
        # add respective id and labels of object
        new_task.object_ids.append(object_id)
        new_task.object_labels.append(object_label)
        # Add newly created task tracker to self.task_trackers
        self.task_trackers.append(new_task)
        return new_task

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
        task_trackers = []
        for recipe in recipes:
            tt = self.create_new_task_tracker(object_id, object_label, recipe)
            task_trackers.append(tt)
        return task_trackers

    def find_task_tracker(self, object_id, object_label) -> list:
        """Finds or create task_tracker for a given object presence

        Args:
            object_id (_type_): object's id
            object_label (_type_): object's name

        Returns:
            list: list of task_trackers this object is in.
        """
        probable_task_trackers = self.get_probable_task_trackers(
            object_id=object_id, object_label=object_label
        )
        # else tasktracker not found hence create a new tasktracker
        if not probable_task_trackers:
            probable_task_trackers = self.create_probable_task_trackers(
                object_id=object_id, object_label=object_label
            )
            if self.verbose:
                self.log.info(
                    f"Created {len(probable_task_trackers)} TaskTrackers for {object_label}_{object_id}"
                )
        return probable_task_trackers

    def track_common_object(self, state, object_id, object_label):
        if self.get_last_task_tracker_tracked_id() is not None:
            last_tracker = self.get_task_tracker(
                self.get_last_task_tracker_tracked_id()
            )
            instruction, track_output = last_tracker.track(
                state=state, objects=[object_label], object_ids=[object_id]
            )
            self.update_last_tracker(last_tracker)
            return [track_output]
        return [None]

    def track_unique_object(self, state, object_id, object_label):
        probable_task_trackers = self.find_task_tracker(object_id, object_label)
        if self.verbose:
            self.log.info(
                f"Found {len(probable_task_trackers)} TaskTrackers for {object_label}_{object_id}"
            )
        # if we are sure there's only one TaskTracker
        if len(probable_task_trackers) == 1:
            task_tracker = probable_task_trackers[0]
            instruction, track_output = task_tracker.track(
                state=state, objects=[object_label], object_ids=[object_id]
            )
            self.update_last_tracker(task_tracker)
            return [track_output]
        # multipe probable task_graphs possible
        else:
            if self.verbose:
                self.log.info("Multipe probable task_graphs possible")
            output, wrong_tracker_list = [], []
            for i, task_tracker in enumerate(probable_task_trackers):
                instruction, track_output = task_tracker.track(
                    state, [object_label], [object_id]
                )
                self.update_last_tracker(task_tracker)
                # if there's an error in instruction, it means we can drop this task_graph
                if isinstance(instruction, ReasoningErrors):
                    if instruction == ReasoningErrors.PARTIAL_STATE:
                        # Return the current step - ignoring the input
                        self.log.info(
                            f"For `{task_tracker.recipe}` Recipe, received partial state completion."
                        )
                        output.append(track_output)
                    else:
                        # Return the error outputs
                        wrong_tracker_list.append(task_tracker)
                        output.append(track_output)
                else:
                    if self.verbose:
                        self.log.info(
                            f"For `{task_tracker.recipe}` Recipe, next instruction = `{instruction}`"
                        )
                    output.append(track_output)
            # Remove the wrong tasks only if all of the probable ones were not wrong
            self.handle_wrong_tasks(probable_task_trackers, wrong_tracker_list)
            return output

    def track_object(self, state, object_id, object_label):
        if object_label in self.important_objects:
            return self.track_unique_object(state, object_id, object_label)
        elif object_label in self.common_objects:
            return self.track_common_object(state, object_id, object_label)
        else:
            self.log.error("Unknown object found.")

    def track_object_state(self, object_id, object_label, object_state):
        for state, prob in object_state.items():
            self.object_states[object_id][state].append(prob)
        output = [None]
        if len(self.object_states[object_id]) >= self.patience:
            avg_state = self.calculate_average_state(object_id, object_label)
            if self.verbose:
                self.log.info(
                    f"Object id: {object_id}, label: {object_label} with avg state={avg_state}"
                )
            output = self.track_object(
                state=avg_state,
                object_id=object_id,
                object_label=object_label,
            )
            self.handle_instruction(output)
            self.reset_object_states(object_id)

        return output

    def process_object(self, obj):
        object_id = obj['id']
        object_label = obj['label']

        if "state" not in obj:
            return [None]

        object_state = obj['state']

        if (
            object_label in self.important_objects
            or object_label in self.common_objects
        ):
            return self.track_object_state(object_id, object_label, object_state)
        else:
            self.log.info(f"Not tracking : {object_label}")
            return [None]

    def calculate_average_state(self, object_id, object_label):
        avg_states = {
            state: sum(probs) / len(probs)
            for state, probs in self.object_states[object_id].items()
        }
        return max(avg_states, key=avg_states.get)

    def handle_instruction(self, instruction):
        if instruction and self.verbose:
            self.log.info(f"Final Next instructions: {instruction}")

    def reset_object_states(self, object_id):
        self.object_states[object_id] = defaultdict(list)

    def handle_message(self, message: list):
        final_output = []
        for obj in message:
            final_output.extend(self.process_object(obj))
        # Log messages throughout trial
        for output in final_output:
            self.demo_logger.log_message(output)
        return final_output

    def update_step_task(self, step_session):
        step_id, tracker_id = step_session.split('&')
        tt = self.get_task_tracker(int(tracker_id))
        if tt is None:
            self.log.error(f"User set step {step_id} for Invalid Task")
            return None
        else:
            instruction, track_output = tt.set_current_step(step_num=int(step_id))
            if self.verbose:
                self.log.info(
                    f"User set step {step_id} for Task {tt.recipe}, received instruction = {instruction}"
                )
            return [track_output]

    def update_task(self, task_tracker_id: int, recipe_name: str):
        """Updates the recipe for a given task/object with user's feedback

        Args:
            task_tracker_id (int): the task_tracker id whose recipe is wrong
            recipe_name (str): the recipe it needs to be changed (CORRECT recipe)
        """
        # Get the task tracker object with the given ID
        tt = self.get_task_tracker(task_tracker_id)
        if tt is None:
            self.log.error("Session ID (task_tracker_id) doesn't exist.")
            return

        # Get associated object IDs and labels
        object_ids = tt.get_object_ids()
        object_labels = tt.get_object_labels()
        # Map object ID/label pairs to probable task trackers
        probable_tts_map = {}
        for object_id, object_label in zip(object_ids, object_labels):
            if object_label in self.important_objects:
                # Get probable task trackers for object
                if (object_id, object_label) not in probable_tts_map:
                    probable_tts_map[
                        (object_id, object_label)
                    ] = self.get_probable_task_trackers(object_id, object_label)
                else:
                    self.log.error("Error: Duplicate object ids")
        # Check all probable task trackers
        for (object_id, object_label), probable_tts in probable_tts_map.items():
            for probable_tt in probable_tts:
                # If recipe matches, return output for current step
                if probable_tt.get_recipe() == recipe_name:
                    track_output = probable_tt.get_current_instruction_output()
                    return [track_output]
        # Otherwise create new task tracker
        new_tt = self.create_new_task_tracker(
            object_id=object_id,
            object_label=object_label,
            recipe=recipe_name,
        )
        # Return output for new task tracker
        track_output = new_tt.get_current_instruction_output()
        return [track_output]
