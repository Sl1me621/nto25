import cv2

import numpy as np

# Параметры фильтрации
MAX_RELATIVE_SIZE = 0.3    # Максимальная относительная площадь объекта
MIN_ABSOLUTE_AREA = 100    # Минимальная площадь объекта в пикселях
MIN_CENTER_DISTANCE = 0.01  # Минимальное расстояние между объектами
MAX_NEW_OBJECTS_PER_FRAME = 3
CENTER_REGION_RATIO = 0.6  # Центральная область для детекции (30% от кадра)
SIZE_VARIANCE = 0.5        # Допустимое отклонение размера (50%)

# Остальные настройки
weights_path = "yolov4-tiny-obj_best.weights"
config_path = "yolov4-tiny-obj.cfg"
classes = ["many_cracs", "many_holes", "one_crak", "two_holes"]
IOU_THRESHOLD = 0.3
MAX_AGE = 5

# Инициализация модели
net = cv2.dnn.readNet(config_path, weights_path)
yolo_model = cv2.dnn.DetectionModel(net)
yolo_model.setInputParams(size=(256, 256), scale=1/255, swapRB=True)

# Словари для трекинга
active_objects = {}
all_objects = {}
next_track_id = 1

def calculate_distance(center1, center2):
    return np.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)

def is_too_close(new_center, existing_objects, min_distance):
    for obj in existing_objects:
        existing_center = (obj['box'][0] + (obj['box'][2]-obj['box'][0])/2,
                          obj['box'][1] + (obj['box'][3]-obj['box'][1])/2)
        if calculate_distance(new_center, existing_center) < min_distance:
            return True
    return False

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    
    return inter_area / float(box1_area + box2_area - inter_area)

def is_valid_size(box, frame_width, frame_height):
    x, y, w, h = box
    object_area = w * h
    frame_area = frame_width * frame_height
    return (object_area/frame_area <= MAX_RELATIVE_SIZE) and (object_area >= MIN_ABSOLUTE_AREA)

def is_near_center(center, frame_size, ratio):
    """Проверяет находится ли центр в центральной области"""
    w, h = frame_size
    cx, cy = w//2, h//2
    region_w, region_h = w*ratio, h*ratio
    return (cx - region_w/2 < center[0] < cx + region_w/2 and 
            cy - region_h/2 < center[1] < cy + region_h/2)

def is_size_similar(new_area, active_objects, variance):
    """Проверяет схожесть размера с активными объектами"""
    if not active_objects:
        return True
        
    areas = [ (obj['box'][2]-obj['box'][0])*(obj['box'][3]-obj['box'][1]) 
             for obj in active_objects.values()]
    avg_area = np.mean(areas)
    return abs(new_area - avg_area)/avg_area <= variance

def process_video(video_path):
    global active_objects, all_objects, next_track_id

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    
    frame_number = 0
    
    while cap.isOpened():
        frame_number += 1
        ret, frame = cap.read()
        if not ret:
            break

        # Обнаружение объектов
        class_ids, scores, boxes = yolo_model.detect(frame, 0.5, 0.4)  # Увеличены пороги
        filtered_objects = []

        if boxes is not None:
            # Получаем текущие центры активных объектов
            active_centers = [
                (obj['box'][0] + (obj['box'][2]-obj['box'][0])/2,
                obj['box'][1] + (obj['box'][3]-obj['box'][1])/2)
                for obj in active_objects.values()
            ]

            # Сортировка по уверенности
            sorted_indices = np.argsort(scores)[::-1]
            
            for i in sorted_indices:
                if len(filtered_objects) >= MAX_NEW_OBJECTS_PER_FRAME:
                    break
                
                box = boxes[i]
                cls = class_ids[i]
                score = scores[i]
                
                # Базовые проверки
                x, y, w, h = box
                center = (x + w/2, y + h/2)
                area = w * h
                
                if not is_valid_size(box, frame_width, frame_height):
                    continue
                
                if not is_near_center(center, frame_size, CENTER_REGION_RATIO):
                    print(f"Объект вне центра: {classes[cls]}")
                    continue
                
                if not is_size_similar(area, active_objects, SIZE_VARIANCE):
                    print(f"Объект неподходящего размера: {classes[cls]}")
                    continue

                # Проверка расстояния
                too_close = is_too_close(center, active_objects.values(), MIN_CENTER_DISTANCE)
                
                if not too_close:
                    for obj in filtered_objects:
                        if calculate_distance(center, obj['center']) < MIN_CENTER_DISTANCE:
                            too_close = True
                            break
                
                if not too_close:
                    filtered_objects.append({
                        "box": [x, y, x + w, y + h],
                        "class": classes[cls],
                        "score": score,
                        "center": center,
                        "area": area
                    })
                else:
                    print(f"Объект близко: {classes[cls]}")
        current_objects = filtered_objects            

        # Сопоставление объектов
        matched = set()
        for obj in current_objects:
            best_iou = 0
            best_id = None

            for track_id, data in active_objects.items():
                iou = calculate_iou(obj["box"], data["box"])
                if iou > best_iou and iou >= IOU_THRESHOLD:
                    best_iou = iou
                    best_id = track_id

            if best_id is not None:
                active_objects[best_id].update({
                    "box": obj["box"],
                    "age": 0
                })
                matched.add(best_id)
                if best_id in all_objects:
                    all_objects[best_id]["last_seen"] = frame_number
            else:
                new_id = next_track_id
                active_objects[new_id] = {
                    "class": obj["class"],
                    "box": obj["box"],
                    "age": 0
                }
                all_objects[new_id] = {
                    "class": obj["class"],
                    "first_seen": frame_number,
                    "last_seen": frame_number
                }
                print(f"Новый объект: {obj['class']} (ID: {new_id})")
                next_track_id += 1

        # Обновление возраста объектов
        to_delete = []
        for track_id in active_objects:
            if track_id not in matched:
                active_objects[track_id]["age"] += 1
                if active_objects[track_id]["age"] > MAX_AGE:
                    to_delete.append(track_id)

        for track_id in to_delete:
            del active_objects[track_id]

        # Визуализация
        for track_id, data in active_objects.items():
            x1, y1, x2, y2 = data["box"]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (55, 0, 255), 2)
            cv2.putText(frame, f"{data['class']} {track_id}", 
                        (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 0, 255), 2)

        # Отображение кадра
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Фиксация результатов
    print("\nИтоговое количество уникальных объектов:", len(all_objects))
    for track_id, data in all_objects.items():
        print(f"Объект {track_id}: {data['class']} | " 
              f"Появился: {data['first_seen']} кадр, "
              f"Последний раз: {data['last_seen']} кадр")

if __name__ == "__main__":
    video_file = "output4.mp4"
    process_video(video_file)