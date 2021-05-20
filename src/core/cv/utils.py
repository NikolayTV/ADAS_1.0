from math import sin, cos, pi
import cv2
import time

def update_fps(frame_start_time):
    """
    Calculate FPS
    """
    now = time.time()
    frame_time = now - frame_start_time
    fps = 1.0 / frame_time
    print('fps', fps)
    frame_start_time = now


def draw_detection_roi(frame, bbox):
    """
    Draw Face detection bounding Box

    Args:
    frame: The Input Frame
    roi: [xmin, xmax, ymin, ymax]
    """
    # Draw face ROI border
    cv2.rectangle(frame,
                  tuple(bbox[:2]), tuple(bbox[2:4]),
                  (0, 220, 0), 2)

def draw_head_pose(frame, bbox, head_pose_angles):
    # Draw headPoseAxes
    # Here head_position_x --> angle_y_fc  # Yaw
    #      head_position_y --> angle_p_fc  # Pitch
    #      head_position_z --> angle_r_fc  # Roll
    yaw, pitch, roll = head_pose_angles

    faceBoundingBoxWidth  = bbox[2] - bbox[0]
    faceBoundingBoxHeight = bbox[3] - bbox[1]

    xCenter = int((bbox[2] + bbox[0]) / 2)
    yCenter = int((bbox[3] + bbox[1]) / 2)

    sinY = sin(yaw * pi / 180.0)
    sinP = sin(pitch * pi / 180.0)
    sinR = sin(roll * pi / 180.0)

    cosY = cos(yaw * pi / 180.0)
    cosP = cos(pitch * pi / 180.0)
    cosR = cos(roll * pi / 180.0)

    axisLength = 0.4 * faceBoundingBoxWidth

    # center to right
    cv2.line(frame, (xCenter, yCenter),
             (((xCenter) + int(axisLength * (cosR * cosY + sinY * sinP * sinR))),
              ((yCenter) + int(axisLength * cosP * sinR))),
             (0, 0, 255), thickness=3)
    # center to top
    cv2.line(frame, (xCenter, yCenter),
             (((xCenter) + int(axisLength * (cosR * sinY * sinP + cosY * sinR))),
              ((yCenter) - int(axisLength * cosP * cosR))),
             (0, 255, 0), thickness=3)

    # Center to forward
    cv2.line(frame, (xCenter, yCenter),
             (((xCenter) + int(axisLength * sinY * cosP)),
              ((yCenter) + int(axisLength * sinP))),
             (255, 0, 0), thickness=3)

def draw_single_person_pose_estimation(frame, persons_keypoints, person_bboxes):
    for pose, bbox in zip(persons_keypoints, person_bboxes):
        draw_detection_roi(frame, bbox)
        print('len(pose)', len(pose))
        for id_kpt, kpt in enumerate(pose):
            cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 3, (0, 0, 255), -1)
