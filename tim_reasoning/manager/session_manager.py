import json
from collections import defaultdict
from datetime import datetime
from tim_reasoning import (
    DemoLogger,
    Logger,
    ObjectPositionTracker,
    RecentTrackerStack,
    RunML,
    TaskTracker,
)
from tim_reasoning.reasoning_errors import ReasoningErrors
from os.path import join, dirname


RECIPE_DATA_FOLDER = join(dirname(__file__), "../../data/recipe")


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
        ohi_threshold: float = 0.0,
    ) -> None:
        self.task_trackers = []
        self.wrong_task_trackers = []
        self.patience = patience
        self.ohi_threshold = ohi_threshold
        self.object_states = defaultdict(lambda: defaultdict(list))
        self.unique_objects_file = unique_objects_file
        self.common_objects_file = common_objects_file
        self.data_folder = data_folder
        self.verbose = verbose
        # Internal logger
        self.log = Logger(name="SessionManager")
        (
            self.important_objects,
            self.common_objects,
        ) = self.get_unique_common_objects()
        # Initiate Recent tracker STACK to track most recent tasks we tracked in memory
        self.recent_tracker_stack = RecentTrackerStack()
        self.object_position_tracker = ObjectPositionTracker()
        # PTG Demo logger
        self.demo_logger = self._get_demo_logger()
        self.demo_logger.start_trial()
        # self.last_task_tracker_tracked_id = None
        self.rm = RunML()
        if self.verbose:
            self.log.info("Session Manager initiated.")

    def _get_demo_logger(self):
        now = datetime.now()

        # Format time as string
        time_str = now.strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3]

        # Append trial number
        log_name = f"trial_log_{time_str}.log"
        return DemoLogger(log_name)

    def get_inprogress_task_ids(self):
        inprogress_task_ids = []
        if len(self.task_trackers) > 2:
            for tt in self.task_trackers:
                if tt.get_current_step_number() > 1:
                    inprogress_task_ids.append(tt.get_id())
            return inprogress_task_ids
        else:
            return [tt.get_id() for tt in self.task_trackers]

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

    def get_recipe(
        self,
        task_name: str,
        recipe_file_name: str = "recipe.json",
        recipe_folder: str = RECIPE_DATA_FOLDER,
    ):
        json_file = f"{recipe_folder}/{recipe_file_name}"
        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        instructions = json_data[task_name]["steps"]
        return instructions

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
                f"For `{wrong_tracker.recipe}` recipe with task id {task_tracker_id}, "
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

    def find_task_tracker(
        self, object_id, object_label, object_hand_interaction
    ) -> list:
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
        if (not probable_task_trackers) and (
            object_hand_interaction is None
            or object_hand_interaction > self.ohi_threshold
        ):
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

            final_output = [track_output]

            unique_obj_id, unique_obj_label = None, None
            # traverse through the objects to get unique objects
            for obj_id, obj_label in zip(
                last_tracker.get_object_ids(), last_tracker.get_object_labels()
            ):
                if obj_label in self.important_objects:
                    unique_obj_id = obj_id
                    unique_obj_label = obj_label
                    break
            if unique_obj_id is not None:
                probable_task_trackers = self.get_probable_task_trackers(
                    unique_obj_id, unique_obj_label
                )
                for tt in probable_task_trackers:
                    if tt.get_id() != last_tracker.get_id():
                        _, track_output = tt.track(
                            state=state,
                            objects=[object_label],
                            object_ids=[object_id],
                        )
                        self.update_last_tracker(tt)
                        final_output.extend([track_output])
            return final_output
        return [None]

    def track_unique_object(
        self, state, object_id, object_label, object_hand_interaction
    ):
        probable_task_trackers = self.find_task_tracker(
            object_id, object_label, object_hand_interaction
        )
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
            self.log.info(
                f"Received instruction = {instruction}, and track_output = {track_output} when "
                f"{object_label}_{object_id} tracked in "
                f"task {task_tracker.recipe}_{task_tracker.get_id()}",
            )
            if instruction == ReasoningErrors.INVALID_STATE:
                next_recipe_step = task_tracker.get_next_recipe_step()
                track_output = task_tracker._build_output_dict(
                    instruction=next_recipe_step
                )
            return [track_output]
        # multipe probable task_graphs possible
        else:
            if self.verbose:
                self.log.info("Multiple probable task_graphs possible")
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
                    elif instruction == ReasoningErrors.FUTURE_STEP:
                        output.append(track_output)
                    else:
                        # Return the error outputs
                        smallest_step_num = min(
                            [
                                probable_t.get_current_step_number()
                                for probable_t in probable_task_trackers
                            ]
                        )
                        if (
                            task_tracker.get_current_step_number()
                            == smallest_step_num
                        ):
                            self.log.info(
                                f"{instruction} Error found while tracking {object_label}_{object_id} for task {task_tracker.recipe}_{task_tracker.get_id()}"
                            )
                            wrong_tracker_list.append(task_tracker)
                        output.append(track_output)
                else:
                    if self.verbose:
                        self.log.info(
                            f"For `{task_tracker.recipe}` Recipe, next instruction = `{instruction}`"
                        )
                    output.append(track_output)
            # Remove the wrong tasks only if all of the probable ones were not wrong
            if wrong_tracker_list:
                self.handle_wrong_tasks(probable_task_trackers, wrong_tracker_list)
            return output

    def track_object(self, state, object_id, object_label, object_hand_interaction):
        if object_label in self.important_objects:
            return self.track_unique_object(
                state, object_id, object_label, object_hand_interaction
            )
        elif object_label in self.common_objects:
            return self.track_common_object(state, object_id, object_label)
        else:
            self.log.error("Unknown object found.")

    def track_object_state(
        self, object_id, object_label, object_state, object_hand_interaction
    ):
        for state, prob in object_state.items():
            self.object_states[object_id][state].append(prob)
        output = [None]
        if (
            len(
                self.object_states[object_id][
                    max(
                        self.object_states[object_id],
                        key=lambda k: len(self.object_states[object_id][k]),
                    )
                ]
            )
            >= self.patience
        ):
            avg_state = self.calculate_average_state(object_id, object_label)
            if self.verbose:
                self.log.info(
                    f"Object id: {object_id}, label: {object_label} with avg state={avg_state}"
                )
            output = self.track_object(
                state=avg_state,
                object_id=object_id,
                object_label=object_label,
                object_hand_interaction=object_hand_interaction,
            )
            self.handle_instruction(output)
            self.reset_object_states(object_id)

        return output

    def process_object(self, obj):
        object_id = obj['id']
        object_label = obj['label']
        object_pos = obj['pos'] if 'pos' in obj else None
        object_hand_interaction = (
            obj['hand_object_interaction']
            if 'hand_object_interaction' in obj
            else None
        )
        if "state" not in obj or obj["state"] == {}:
            return [None]

        self.object_position_tracker.set_pos(
            object_id=object_id, object_pos=object_pos
        )
        object_state = obj['state']

        if (
            object_label in self.important_objects
            or object_label in self.common_objects
        ):
            return self.track_object_state(
                object_id, object_label, object_state, object_hand_interaction
            )
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

    def quick_fix_ui_output(
        self, final_output, dashboard_output, object_id, object_name
    ):
        """we need to update the steps in task graph using the prediction from
        ML model

        Args:
            final_output (_type_): this contains not so great step detection
            dashboard_output (_type_): contains the ML predicted steps

        Returns:
            dict: final_output that is provided to the ui
        """
        print('dashboard_output', dashboard_output)
        print('final_output', final_output)
        if "task_name" in dashboard_output and "step_num" in dashboard_output:
            if len(final_output["active_tasks"]) > 0:
                active_task = final_output["active_tasks"][0]
                if active_task is None:
                    #active_task = {}
                    return final_output
                task_name = dashboard_output["task_name"]
                step_num = dashboard_output["step_num"]
                active_task["task_name"] = task_name
                active_task["step_id"] = step_num

                instructions = self.get_recipe(task_name=task_name)
                if len(instructions) > 1:
                    active_task["step_description"] = instructions[str(step_num)]
                    active_task["total_steps"] = len(instructions)
                final_output["active_tasks"] = [active_task]
                # object_id from dashboard_output (which has BETTER `task_name` and `step_num`)
                tts = self.get_probable_task_trackers(
                    object_id=object_id, object_label=object_name
                )
                # find tasktracker for given object_id in the backend
                # change the tts[0]
                target_task_tracker = tts[0]
                # if the task is predicted different modify it
                if target_task_tracker.recipe != task_name:
                    target_task_tracker.set_recipe_name(new_recipe_name=task_name)
                # update the step based on ml model
                final_output = self.update_step(
                    target_task_tracker.get_id(), step_id=step_num + 1
                )
                return final_output

        return final_output

    def handle_message(self, message: list, entire_message: list):
        active_output = []
        # Traverse a single object
        for obj in message:
            # get graph system output
            manager_output = self.process_object(obj)
            # add 3d pos
            active_tasks = self.object_position_tracker.create_active_tasks_output(
                manager_output
            )
            active_output.extend(active_tasks)
        final_output = {
            "active_tasks": active_output,
            "inprogress_task_ids": self.get_inprogress_task_ids(),
        }
        dashboard_output = self.rm.run_message(message[0], entire_message)
        object_id = message[0].get("id")
        object_name = message[0].get("label")
        #print('hi', object_id, object_name, message[0])
        #raise ValueError("")
        final_output = self.quick_fix_ui_output(
            final_output, dashboard_output, object_id, object_name
        )
        # Log messages throughout trial
        if final_output["active_tasks"] is not None:
            for output in final_output["active_tasks"]:
                self.demo_logger.log_message(output)
        return final_output, dashboard_output

    def update_step(self, task_tracker_id, step_id):
        tt = self.get_task_tracker(task_tracker_id)
        if tt is None:
            self.log.error(f"User set step {step_id} for Invalid Task")
            return
        else:
            instruction, track_output = tt.set_current_step(step_num=step_id)
            if self.verbose:
                self.log.info(
                    f"User set step {step_id} for Task {tt.recipe}, received instruction = {instruction}"
                )
            active_tasks = self.object_position_tracker.create_active_tasks_output(
                [track_output]
            )
            final_output = {
                "active_tasks": active_tasks,
                "inprogress_task_ids": self.get_inprogress_task_ids(),
            }
            return final_output

    def update_task(self, correct_task_tracker_id: int, recipe_name: str):
        # Get the task tracker object with the given ID
        tt = self.get_task_tracker(correct_task_tracker_id)
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
        # Check all probable task trackers and remove incorrect ones
        for (object_id, object_label), probable_tts in probable_tts_map.items():
            for probable_tt in probable_tts:
                # If id matches, return output for current step
                if probable_tt.get_id() != correct_task_tracker_id:
                    self.remove_task_tracker(wrong_tracker=probable_tt)

        track_output = tt.get_current_instruction_output()
        active_tasks = self.object_position_tracker.create_active_tasks_output(
            [track_output]
        )
        final_output = {
            "active_tasks": active_tasks,
            "inprogress_task_ids": self.get_inprogress_task_ids(),
        }
        return final_output

    def change_task(self, task_tracker_id: int, recipe_name: str):
        """Changes the recipe for a given task/object with user's feedback

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
                else:
                    self.remove_task_tracker(wrong_tracker=probable_tt)
        # Otherwise create new task tracker
        new_tt = self.create_new_task_tracker(
            object_id=object_id,
            object_label=object_label,
            recipe=recipe_name,
        )
        # Return output for new task tracker
        track_output = new_tt.get_current_instruction_output()
        return [track_output]
