class ObjectPositionTracker:
    def __init__(self) -> None:
        self.object_positions = {}  # key: obj_id, pos: pos

    def get_pos(self, object_id: int) -> list:
        return self.object_positions[object_id]

    def set_pos(self, object_id: int, object_pos: list):
        self.object_positions[object_id] = object_pos

    def get_object_info(self, object_id, object_label):
        return {
            "id": object_id,
            "name": object_label,
            "pos": self.get_pos(object_id=object_id),
        }

    def get_multiple_objects_data(self, object_ids, object_labels):
        objects_data = []
        for object_id, object_label in zip(object_ids, object_labels):
            objects_data.append(self.get_object_info(object_id, object_label))
        return objects_data

    def remove_object_ids_labels(self, modified_output):
        if "object_ids" in modified_output:
            modified_output.pop('object_ids')
        if "object_labels" in modified_output:
            modified_output.pop('object_labels')
        return modified_output

    def create_modified_output(self, task_output):
        modified_output = task_output
        objects_data = self.get_multiple_objects_data(
            task_output['object_ids'], task_output['object_labels']
        )
        modified_output['objects'] = objects_data
        modified_output = self.remove_object_ids_labels(
            modified_output=modified_output
        )
        return modified_output

    def create_active_tasks_output(self, data):
        if data == [None]:
            return data
        else:
            active_tasks_output = []
            for task_output in data:
                modified_output = self.create_modified_output(task_output)
                active_tasks_output.append(modified_output)
            return active_tasks_output
