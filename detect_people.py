import requests
import cv2
import numpy as np
import os
import threading
import time
from queue import Queue, Empty


def get_camera_urls():
    url = "https://api-vchd-7.uz/api/v1/camera/get-list"
    headers = {
        "Authorization": "Bearer 1|OCbBOk8empoAffzBUzNvhFZWIJTgL03TZQ0DRwZe628649a9",
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        cameras = data.get("results", [])
        return cameras
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return []


def send_video_name_to_api(video_filename):
    url = "https://api-vchd-7.uz/api/v1/video/add"
    headers = {
        "Authorization": "Bearer 1|OCbBOk8empoAffzBUzNvhFZWIJTgL03TZQ0DRwZe628649a9",
        "Content-Type": "application/json"
    }
    data = {
        "file": video_filename
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        print(f"{video_filename} nomli video muvaffaqiyatli yuborildi.")
    else:
        print(f"{video_filename} nomli videoni yuborishda xatolik yuz berdi: {response.status_code}")
        try:
            print(f"Serverdan qaytarilgan xabar: {response.text}")
        except Exception as e:
            print(f"Xato haqida ma'lumotni olishda muammo: {e}")


camera_data = get_camera_urls()

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

helmet_colors = {
    "orange": ([5, 100, 100], [15, 255, 255]),  # To'q sariq rang
    "white": ([0, 0, 180], [180, 30, 255]),  # Oq rang
    "red": ([0, 100, 100], [10, 255, 255])  # Qizil rang
}

output_folder = 'recorded_videos'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_queue = Queue(maxsize=10)
stop_event = threading.Event()

record_time = 30
recently_detected = {}
id_counter = 1


def check_helmet(head_region):
    if head_region.size == 0:
        return False
    hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
    for color, (lower, upper) in helmet_colors.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        if 500 < cv2.countNonZero(mask) < 5000:
            return True
    return False


def get_next_video_index(camera_id):
    existing_files = [f for f in os.listdir(output_folder) if f.endswith('.mp4')]
    indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.startswith(f'no_helmet_{camera_id}')]
    if indices:
        return max(indices) + 1
    return 0


def start_video_recording(width, height, video_index, camera_id):
    fps = 2.0
    video_filename = os.path.join(output_folder, f'no_helmet_{camera_id}_{video_index}.mp4')
    return cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)), video_filename


def generate_id():
    global id_counter
    current_id = id_counter
    id_counter += 1
    return current_id


def camera_capture(camera_id, camera_url, frame_queue):
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print(f"Kamera {camera_url} bilan aloqa o'rnatilmadi. Qayta urinilmoqda...")
        return
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"Strim {camera_url} yo'qolgan. Qayta ulanmoqda...")
            break
        if not frame_queue.full():
            frame_queue.put((camera_id, frame))
    cap.release()


def process_frames():
    video_outs = {}
    start_times = {}
    recording_ids = {}
    frame_buffers = {}

    while not stop_event.is_set() or not frame_queue.empty():
        try:
            camera_id, frame = frame_queue.get(timeout=1)
        except Empty:
            continue

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for detection_layer in detections:
            for detection in detection_layer:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == 'person':
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        current_ids = set()

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                head_region = frame[y:y + h // 3, x:x + w]

                if camera_id not in recording_ids:
                    recording_ids[camera_id] = []

                while len(recording_ids[camera_id]) <= i:
                    recording_ids[camera_id].append(generate_id())

                person_id = recording_ids[camera_id][i]
                current_ids.add(person_id)

                is_wearing_helmet = check_helmet(head_region)
                color = (0, 255, 0) if is_wearing_helmet else (0, 0, 255)
                label = f"ID: {person_id} - {'Kaska bor' if is_wearing_helmet else 'Kaska yoq'}"

                if not is_wearing_helmet:
                    if camera_id not in video_outs:
                        video_index = get_next_video_index(camera_id)
                        video_out, video_filename = start_video_recording(width, height, video_index, camera_id)
                        video_outs[camera_id] = video_out
                        start_times[camera_id] = time.time()
                        frame_buffers[camera_id] = []

                if camera_id in frame_buffers:
                    frame_buffers[camera_id].append(frame)
                    if len(frame_buffers[camera_id]) >= record_time * 2:
                        for buffered_frame in frame_buffers[camera_id]:
                            video_outs[camera_id].write(buffered_frame)
                        video_outs[camera_id].release()
                        send_video_name_to_api(video_filename)
                        del video_outs[camera_id]
                        del frame_buffers[camera_id]

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow(f'Camera {camera_id}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    for camera_id in video_outs.keys():
        if camera_id in frame_buffers:
            for buffered_frame in frame_buffers[camera_id]:
                video_outs[camera_id].write(buffered_frame)
            video_outs[camera_id].release()
            send_video_name_to_api(video_filename)

    cv2.destroyAllWindows()


camera_threads = []
for camera in camera_data:
    camera_id = camera['id']
    camera_url = camera['ip_port']
    camera_thread = threading.Thread(target=camera_capture, args=(camera_id, camera_url, frame_queue))
    camera_threads.append(camera_thread)
    camera_thread.start()

processing_thread = threading.Thread(target=process_frames)
processing_thread.start()


def stop_threads():
    stop_event.set()
    for camera_thread in camera_threads:
        camera_thread.join()
    processing_thread.join()
    cv2.destroyAllWindows()
    print("Dastur to'xtatildi.")


try:
    while any(thread.is_alive() for thread in camera_threads) or processing_thread.is_alive():
        for camera_thread in camera_threads:
            camera_thread.join(timeout=1)
        processing_thread.join(timeout=1)
except KeyboardInterrupt:
    stop_threads()
