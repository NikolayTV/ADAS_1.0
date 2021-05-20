from scipy.spatial import distance as dist
import numpy as np
from collections import OrderedDict


class Person:
    def __init__(self, person_ID):
        self.person_ID = person_ID  # Person face in frame right now
        self.present = False  # Person face in frame right now
        self.current_face_centroid = (None, None)  # Centroid coords in frame
        self.current_face_bbox = (None, None)  # Centroid coords in frame
        self.present_time = None  # time.time()
        self.last_face_centroid = (None, None)  # Last centroid coords when person was detected
        self.gone_frames = 0
        # Emotions
        self.looking = 0
        self.positive = 0
        self.negative = 0
        self.neutral = 0
        self.current_sentiment = ''
        # Head pose
        self.head_pose = (None, None, None)  # yaw pitch roll
        # Gaze
        self.gaze = (None, None)  # yaw pitch roll


class PersonTracker():
    def __init__(self, maxDisappeared=200):
        self.person_ID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def update(self, rects, tracked_persons):
        if len(rects) == 0:
            for num in range(len(tracked_persons)):
                tracked_persons[num].gone_frames += 1
                if tracked_persons[num].gone_frames > self.maxDisappeared:
                    del tracked_persons[num]
            return tracked_persons

        inputCentroids = [(int((startX + endX) / 2), int((startY + endY) / 2.0)) for startX, startY, endX, endY in rects]

        # Create tracked persons if they not exist
        if len(tracked_persons) == 0:
            tracked_persons = [Person(num) for num in range(len(rects))]
            for num in range(len(rects)):
                tracked_persons[num].current_face_centroid = inputCentroids[num]
                tracked_persons[num].last_face_centroid = inputCentroids[num]
                tracked_persons[num].current_face_bbox = rects[num]
                tracked_persons[num].gone_frames = 0

        # Update tracked persons (update, delete or add new)
        else:
            objectIDs = [person.person_ID for person in tracked_persons]
            objectCentroids = [person.last_face_centroid for person in tracked_persons]

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()  # Индексы строчек по возрастанию с наименьшими расстояниями
            cols = D.argmin(axis=1)[rows]  # Индексы колонок с наименьшими значениями по возрастанию

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                # Check if the distance is too big to consider this as the same object
                if D[row, col] > 100:
                    continue

                objectID = objectIDs[row]
                tracked_persons[objectID].current_face_centroid = inputCentroids[col]
                tracked_persons[objectID].last_face_centroid = inputCentroids[col]
                tracked_persons[objectID].gone_frames = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # Если новых МЕНЬШЕ чем старых УДАЛИТЬ объект
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    tracked_persons[objectID].gone_frames += 1

                    if tracked_persons[objectID].gone_frames > self.maxDisappeared:
                        del tracked_persons[objectID]

            # Если новых БОЛЬШЕ чем старых ДОБАВИТЬ объект
            else:
                for col in unusedCols:
                    new_person_id = len(tracked_persons)
                    tracked_persons.append(Person(new_person_id))
                    tracked_persons[new_person_id].current_face_bbox = rects[col]
                    tracked_persons[new_person_id].current_face_centroid = inputCentroids[col]
                    tracked_persons[new_person_id].last_face_centroid = inputCentroids[col]
                    tracked_persons[new_person_id].gone_frames = 0

        return tracked_persons
