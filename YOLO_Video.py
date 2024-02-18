from ultralytics import YOLO
import cv2
import math


class PassengerCounter:
    def __init__(self):
        self.passenger_id_map = {}
        self.passenger_id_counter = 1
        self.free_ids = []

    def get_passenger_id(self, bbox):
        for id_, (prev_bbox, _) in self.passenger_id_map.items():
            if self.are_boxes_equal(bbox, prev_bbox):
                return id_
        return None

    def add_passenger(self, bbox):
        if self.free_ids:
            passenger_id = self.free_ids.pop()
        else:
            passenger_id = self.passenger_id_counter
            self.passenger_id_counter += 1
        self.passenger_id_map[passenger_id] = (bbox, False)
        return passenger_id

    def remove_passenger(self, passenger_id):
        del self.passenger_id_map[passenger_id]
        self.free_ids.append(passenger_id)

    def mark_passenger_as_detected(self, passenger_id):
        self.passenger_id_map[passenger_id] = (self.passenger_id_map[passenger_id][0], True)

    def are_boxes_equal(self, bbox1, bbox2):
        return bbox1 == bbox2


passenger_counter = PassengerCounter()


def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    model = YOLO("best.pt")
    classNames = ["Conductor", "Driver", "Passenger"]

    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Check if the passenger is already detected
                passenger_id = passenger_counter.get_passenger_id((x1, y1, x2, y2))

                # Generate unique ID for new passenger
                if passenger_id is None:
                    passenger_id = passenger_counter.add_passenger((x1, y1, x2, y2))

                # Mark passenger as detected
                passenger_counter.mark_passenger_as_detected(passenger_id)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf} ID: {passenger_id}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        yield img

cv2.destroyAllWindows()
