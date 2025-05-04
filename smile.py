import cv2
import os
import datetime

# load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

if face_cascade.empty() or smile_cascade.empty():
    raise IOError("Could not load Haar cascades")

# prepare output
output_dir = "smilescreens"
os.makedirs(output_dir, exist_ok=True)

# open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("Press 'q' to quit.")

prev_smile = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    smiling = False

    for (x, y, w, h) in faces:
        # draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=22,
            minSize=(25, 25)
        )

        # find largest smile
        if len(smiles) > 0:
            max_sw = 0
            for (sx, sy, sw, sh) in smiles:
                if sw > max_sw:
                    max_sw = sw
                    best = (sx, sy, sw, sh)
            # compute relative width
            if max_sw / float(w) >= 0.6:
                smiling = True
                # draw only the “big” smile box
                sx, sy, sw, sh = best
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)

    # on transition from not‑smiling to smiling, save
    if smiling and not prev_smile:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        fn = os.path.join(output_dir, f"smile_{ts}.jpg")
        cv2.imwrite(fn, small)
        print(f"Saved extreme smile → {fn}")

    prev_smile = smiling

    cv2.imshow("Smile Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
