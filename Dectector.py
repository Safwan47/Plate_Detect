from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import pytesseract
import numpy as np
from collections import Counter
import re
from datetime import datetime


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not detected!")
else:
    print("Camera connected successfully.")
cap.release()

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

file_path = r'E:\Tech\Projects\PlateTextDectect.txt'

# Clear the content of the file at the start of a new run
with open(file_path, 'w') as f:
    pass

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("E:/Tech/Projects/PlateTextDectect/best.pt")

classNames = ["license_plate"]

prev_frame_time = 0
new_frame_time = 0
start_time = time.time()

ocr_results = []

# Function to preprocess the image for better OCR accuracy
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2))  # Upscale the image
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Function to filter OCR results based on a typical license plate pattern
def filter_plate_text(text):
    # License plates usually contain 5 to 10 characters, and are a mix of letters and digits
    pattern = r'^[A-Z0-9]{5,10}$'
    if re.match(pattern, text):
        return text
    return None

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        print("Failed to read frame from camera")
        break


    img_resized = cv2.resize(img, (640, 640))
    scale_x = img.shape[1] / 640
    scale_y = img.shape[0] / 640

    results = model(img_resized, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            cls = int(box.cls[0])
            if cls < len(classNames):
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            cropped_plate = img[y1:y2, x1:x2]
            processed_plate = preprocess_image(cropped_plate)

            # OCR Configuration with a stricter whitelist and a different PSM mode
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            plate_text = pytesseract.image_to_string(processed_plate, config=custom_config).strip()

            if plate_text:
                filtered_text = filter_plate_text(plate_text)
                if filtered_text:
                    ocr_results.append(filtered_text)
                    print(f"Detected License Plate Text: {filtered_text}")

    if time.time() - start_time >= 3:
        if ocr_results:
            final_text = Counter(ocr_results).most_common(1)[0][0]
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Final License Plate Text: {final_text}")

            try:
                with open(file_path, 'a') as f:
                    f.write(f'{final_text}\t\t{timestamp}\n')
                print(f"Text '{final_text}' with timestamp written to {file_path}")
            except Exception as e:
                print(f"Error writing to file: {e}")

        ocr_results.clear()
        start_time = time.time()

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {round(fps, 2)}")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
