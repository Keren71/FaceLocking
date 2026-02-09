import cv2
from src.config import *
from src.face_lock import FaceLock
from src.action_detection import *
from src.history_logger import HistoryLogger
from src.recognize import recognize_faces

face_lock = FaceLock(TARGET_IDENTITY, MAX_LOST_FRAMES)
logger = None

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = recognize_faces(frame)
    print("Detected faces:", faces)

    for face in faces:
        name = face["name"]
        similarity = face["similarity"]
        bbox = face["bbox"]
        landmarks = face["landmarks"]

        x1, y1, x2, y2 = bbox

        # -------- ALWAYS SHOW RECOGNITION RESULT --------
        label = name if similarity > LOCK_THRESHOLD else "Unknown"
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        # -------- TRY TO LOCK TARGET ONLY --------
        locked_now = face_lock.try_lock(
            name,
            similarity,
            LOCK_THRESHOLD,
            bbox
        )

        if locked_now and logger is None:
            logger = HistoryLogger(name)

        face_lock.update_tracking(name, bbox)

        # -------- SPECIAL ACTIONS ONLY FOR LOCKED FACE --------
        if face_lock.locked and name == face_lock.locked_id:
            left_eye, right_eye, nose, mouth_left, mouth_right = landmarks

            move = detect_head_movement(nose[0], MOVEMENT_THRESHOLD)
            blink = detect_blink(left_eye, right_eye, BLINK_EAR_THRESHOLD)
            smile = detect_smile(
                mouth_left,
                mouth_right,
                SMILE_WIDTH_THRESHOLD
            )

            for action in [move, blink, smile]:
                if action:
                    print("[ACTION]", action)
                    logger.log(action)

            # Label near the locked face
            cv2.putText(
                frame,
                "LOCKED",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

            # Big global indicator on screen
            cv2.putText(
                frame,
                f"TARGET LOCKED: {face_lock.locked_id}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Face Locking System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
