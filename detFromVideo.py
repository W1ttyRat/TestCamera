import cv2, mss, time, json
from fast_alpr import ALPR
import numpy as np

# Initialize the ALPR system with the specified models
MODEL = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model",
)

# load the image
def load_image(image_path):
    frame = cv2.imread(image_path)
    return frame

# draw the predictions on the image
def draw_predictions(frame):
    annotated_frame = MODEL.draw_predictions(frame)
    return annotated_frame

# get the predictions for the image
def get_predictions(frame):
    predictions = MODEL.predict(frame)
    return predictions

# save the annotated image
def save_annotated_image(annotated_frame):
    output_path = "AnnotatedImages/annotated_image.jpg"
    cv2.imwrite(output_path, annotated_frame)

# process video frames from a video file or webcam
def process_video(source):
    CarSet = set()
    CarList = []

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = get_predictions(frame)
        for r in results:
            plate_text = r.ocr.text
            ocr_confidence = r.ocr.confidence
            det_confidence = r.detection.confidence
            box = r.detection.bounding_box

            if ocr_confidence > 0.98 and det_confidence > 0.82 and plate_text not in CarSet:
                CarSet.add(plate_text)
                CarList.append((plate_text, round(ocr_confidence, 2), round(det_confidence, 2)))

                check_car_info(plate_text)

            #if ocr_confidence > 0.98 and det_confidence > 0.82:
            #    print(f"High confidence for plate: {plate_text}, OCR Confidence: {ocr_confidence:.2f}, Detection Confidence: {det_confidence:.2f}, Box: {box}")
            #else:
            #    print(f"Low confidence for plate: {plate_text}, OCR Confidence: {ocr_confidence:.2f}, Detection Confidence: {det_confidence:.2f}, Box: {box}")
            

        

    cap.release()
    return CarList

# process video frames from a specific monitor for a certain duration
def process_monitor(monitor_index=2, duration_seconds=30):
    CarSet = set()
    CarList = list()

    with mss.mss() as sct:

        start_time = time.time()
        monitor = sct.monitors[monitor_index]

        while True:

            if time.time() - start_time >= duration_seconds:
                break

            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            results = get_predictions(frame)

            for r in results:
                plate_text = r.ocr.text
                ocr_confidence = r.ocr.confidence
                det_confidence = r.detection.confidence
                box = r.detection.bounding_box

                if ocr_confidence > 0.98 and det_confidence > 0.82 and plate_text not in CarSet:
                    CarSet.add(plate_text)
                    CarList.append((plate_text, round(ocr_confidence, 2), round(det_confidence, 2)))


                if ocr_confidence > 0.6 and det_confidence > 0.6:
                    print(f"High confidence for plate: {plate_text}, OCR Confidence: {ocr_confidence:.2f}, Detection Confidence: {det_confidence:.2f}, Box: {box}")
                else:
                    print(f"Low confidence for plate: {plate_text}, OCR Confidence: {ocr_confidence:.2f}, Detection Confidence: {det_confidence:.2f}, Box: {box}")


def save_carlist_json(car_list, output_path="carlist.json"):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([{"plate": p, "ocr_confidence": o, "det_confidence": d} for p, o, d in car_list], f, indent=2)

def check_car_info(plate_text):
    with open("mocklist.json", "r", encoding="utf-8") as f:
        mock_data = json.load(f)
        
        for entry in mock_data:
            #print(entry["plate"])
            if entry["plate"] == plate_text:
                print(f"Car Info for {plate_text}: Make: {entry['make']}, Model: {entry['model']}, Year: {entry['year']}, Color: {entry['color']}")
                return

        #print(f"Checking car info for plate: {plate_text}")

        #if plate_text in mock_data:
         #   car_info = mock_data[plate_text]
          #  print(f"Car Info for {plate_text}: Make: {car_info['make']}, Model: {car_info['model']}, Year: {car_info['year']}, Color: {car_info['color']}")
    print(f"Car Info for {plate_text}: Not found")

if __name__ == "__main__":
    print("Starting ALPR system in 2 seconds...")
    time.sleep(2)
    print("Processing video...")
    
    # For webcam, use 0
    #process_video(0)

    # For video file, provide the path
    car_list = process_video("carvideo2.mp4")

    # For 2nd monitor screen capture
    #process_monitor(2, duration_seconds=25)

    for plate_info in car_list:
        print(f"Detected Plate: {plate_info[0]}, OCR Confidence: {plate_info[1]}, Detection Confidence: {plate_info[2]}")
    print(len(car_list))
    save_carlist_json(car_list, "carlist.json")


