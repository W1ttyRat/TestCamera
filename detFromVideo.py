import cv2
from fast_alpr import ALPR

# Initialize the ALPR system with the specified models
MODEL = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model",
)

# load the image
def load_image(image_path):
    frame = cv2.imread(image_path)
    return frame

def draw_predictions(frame):
    annotated_frame = MODEL.draw_predictions(frame)
    return annotated_frame

def get_predictions(frame):
    predictions = MODEL.predict(frame)
    return predictions

# save the annotated image
def save_annotated_image(annotated_frame):
    output_path = "AnnotatedImages/annotated_image.jpg"
    cv2.imwrite(output_path, annotated_frame)


def process_video(source):
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

            if ocr_confidence > 0.6 and det_confidence > 0.6:
                print(f"High confidence for plate: {plate_text}, OCR Confidence: {ocr_confidence:.2f}, Detection Confidence: {det_confidence:.2f}, Box: {box}")
            else:
                print(f"Low confidence for plate: {plate_text}, OCR Confidence: {ocr_confidence:.2f}, Detection Confidence: {det_confidence:.2f}, Box: {box}")

        

    cap.release()

if __name__ == "__main__":

    # For webcam, use 0
    #process_video(0)

    # For video file, provide the path
    process_video("carvideo.mp4")