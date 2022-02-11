from queue import Queue
import time
from threading import Thread
from utils.utils import *
from utils.detect_card import detect_card
import matplotlib.pyplot as plt

input_path = '/home/cuong/Desktop/test case 3/test_3_10.mp4'

model = detect_card(weights_person='Body_Card/yolov5s.pt',
                    weights_card='Body_Card/model_detect_card.pt',
                    weights_classification='Body_Card/classifier.h5',
                    detect_person=True,
                    use_cuda=False)

def video_capture(frame_detect_queue, frame_origin_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_detect_queue.put(frame)
        frame_origin_queue.put(frame)
    cap.release()
def inference(frame_detect_queue, detections_queue):
    while cap.isOpened():
        image = frame_detect_queue.get()
        bboxes_card, labels_card, scores_card, scores_classification = model.predict(image, show=False)
        detections_queue.put([bboxes_card, labels_card, scores_card, scores_classification])

    cap.release()

def drawing(detections_queue, frame_origin_queue, frame_final_queue):
    while cap.isOpened():
        frame_origin = frame_origin_queue.get()
        bboxes, labels, scores, scores_classification = detections_queue.get()
        if frame_origin is not None:
            image = draw_bboxes(frame_origin, bboxes, labels, scores, scores_classification)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if frame_final_queue.full() is False:
                frame_final_queue.put(image)
            else:
                time.sleep(0.001)
    cap.release()


if __name__ == '__main__':
    frame_detect_queue = Queue(maxsize=1)
    frame_origin_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    frame_final_queue = Queue(maxsize=1)

    save_card = False
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_video = fps if fps <= 120 else 30

    print(fps_video)

    video_filename = '/home/cuong/Desktop/Videos/1.avi' # record detect (no gpu)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_write = cv2.VideoWriter(video_filename, fourcc, fps_video, (width, height))

    Thread(target=video_capture, args=(frame_detect_queue, frame_origin_queue)).start()
    Thread(target=inference, args=(frame_detect_queue, detections_queue)).start()
    Thread(target=drawing, args=(detections_queue, frame_origin_queue, frame_final_queue)).start()
    while True:
        if cap.isOpened():
            image = frame_final_queue.get()
            video_write.write(image)
            image = cv2.resize(image, (1280, 720))
            cv2.imshow('output', image)
            #time.sleep(0.01)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyWindow('output')
                break

    cv2.destroyAllWindows()