import cv2
import numpy as np
from collections import defaultdict

# Параметры фильтрации
MAX_RELATIVE_SIZE = 0.3
MIN_ABSOLUTE_AREA = 100
MIN_CENTER_DISTANCE = 30  # В пикселях
MAX_AGE = 15
CENTER_REGION_RATIO = 0.6
SIZE_VARIANCE = 0.3
TRACK_HISTORY = 20  # Глубина истории для трекинга
APPEARANCE_THRESHOLD = 0.8  # Порог схожести внешнего вида

# Инициализация модели
net = cv2.dnn.readNet("yolov4-tiny-obj_best_drone.weights", "yolov4-tiny-obj_drone.cfg")
yolo_model = cv2.dnn.DetectionModel(net)
yolo_model.setInputParams(size=(256, 256), scale=1/255, swapRB=True)
classes = ["many_cracs", "many_holes", "one_crak", "two_holes"]


# Словари для трекинга
active_objects = defaultdict(lambda: {
    'box': None,
    'age': 0,
    'history': [],
    'color_hist': None
})
all_objects = {}
next_track_id = 1
def is_near_center(center, frame_size, ratio):
    """Проверяет находится ли центр объекта в центральной области кадра"""
    frame_w, frame_h = frame_size
    center_x, center_y = center
    region_w = frame_w * ratio
    region_h = frame_h * ratio
    
    # Границы центральной области
    x_start = (frame_w - region_w) // 2
    x_end = x_start + region_w
    y_start = (frame_h - region_h) // 2
    y_end = y_start + region_h
    
    return (x_start < center_x < x_end) and (y_start < center_y < y_end)

def calculate_distance(center1, center2):
    """Вычисляет евклидово расстояние между двумя точками"""
    return np.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)

def calculate_color_hist(image, box):
    x1, y1, x2, y2 = map(int, box)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0: return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def compare_hists(hist1, hist2):
    if hist1 is None or hist2 is None: return 0
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def predict_next_position(history):
    if len(history) < 2: return history[-1] if history else None
    dx = history[-1][0] - history[-2][0]
    dy = history[-1][1] - history[-2][1]
    return (history[-1][0] + dx, history[-1][1] + dy)

def object_similarity(new_obj, existing_obj, frame):
    # Сравнение размера
    new_area = (new_obj['box'][2]-new_obj['box'][0])*(new_obj['box'][3]-new_obj['box'][1])
    existing_area = (existing_obj['box'][2]-existing_obj['box'][0])*(existing_obj['box'][3]-existing_obj['box'][1])
    size_sim = 1 - abs(new_area - existing_area)/max(new_area, existing_area)
    
    # Сравнение позиции
    pred_pos = predict_next_position(existing_obj['history'])
    if pred_pos:
        curr_pos = np.mean([new_obj['box'][:2], new_obj['box'][2:]], axis=0)
        dist = np.linalg.norm(np.array(pred_pos) - np.array(curr_pos))
        pos_sim = max(0, 1 - dist/100)
    else:
        pos_sim = 0
    
    # Сравнение внешнего вида
    hist = calculate_color_hist(frame, new_obj['box'])
    app_sim = compare_hists(hist, existing_obj['color_hist'])
    
    return 0.4*size_sim + 0.3*pos_sim + 0.3*app_sim

def process_video(video_path):
    global active_objects, all_objects, next_track_id

    cap = cv2.VideoCapture(video_path)
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Создаем копию кадра для рисования
        display_frame = frame.copy()

        # Детекция и фильтрация объектов
        class_ids, scores, boxes = yolo_model.detect(frame, 0.6, 0.4)
        current_objects = []

        for box, cls in zip(boxes, class_ids):
            x, y, w, h = box
            area = w * h
            center = (x + w/2, y + h/2)
            
            if (area > MIN_ABSOLUTE_AREA 
                and area/(frame_size[0]*frame_size[1]) < MAX_RELATIVE_SIZE 
                and is_near_center(center, frame_size, CENTER_REGION_RATIO)):
                
                current_objects.append({
                    'box': [x, y, x+w, y+h],
                    'class': classes[cls],  # Гарантируем наличие класса
                    'center': center,
                    'area': area,
                    'hist': calculate_color_hist(frame, [x, y, x+w, y+h])
                })

        # Сопоставление объектов
        matched = set()
        for new_obj in current_objects:
            best_match = None
            max_sim = 0
            
            for track_id, existing_obj in active_objects.items():
                if track_id in matched: continue
                
                sim = object_similarity(new_obj, existing_obj, frame)
                if sim > max_sim and sim > APPEARANCE_THRESHOLD:
                    max_sim = sim
                    best_match = track_id
            
            if best_match:
                print(f"Обновлен объект: {new_obj['class']} (ID: {best_match})")
                # Обновление с сохранением класса
                active_objects[best_match].update({
                    'box': new_obj['box'],
                    'center': new_obj['center'],
                    'history': active_objects[best_match]['history'][-TRACK_HISTORY:] + [new_obj['center']],
                    'color_hist': new_obj['hist'],
                    'age': 0
                })
                matched.add(best_match)
            else:
                print(f"Обнаружен новый объект: {new_obj['class']} (ID: {next_track_id})")
                # Создание нового объекта с классом
                active_objects[next_track_id] = {
                    'box': new_obj['box'],
                    'class': new_obj['class'],
                    'history': [new_obj['center']],
                    'color_hist': new_obj['hist'],
                    'age': 0
                }
                all_objects[next_track_id] = {
                    'class': new_obj['class'],
                    'first_seen': len(all_objects)+1,
                    'last_seen': len(all_objects)+1
                }
                next_track_id += 1
        # to_delete = [tid for tid, obj in active_objects.items() if obj['age'] > MAX_AGE]
        # for tid in to_delete:
        #     del active_objects[tid]
        # Визуализация с проверкой
        for track_id, obj in active_objects.items():
                if obj['box'] is None:
                    continue
                
                # Получаем актуальные координаты
                x1, y1, x2, y2 = map(int, obj['box'])
                
                # Рисуем прямоугольник и текст
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(display_frame, f"{obj.get('class', 'unknown')} {track_id}",(x1, y1-10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2) 

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) == 27: break
    print("\nИтоговая статистика объектов:")
    print("----------------------------")
    class_stats = {}
    for track_id, data in all_objects.items():
        class_name = data['class']
        if class_name not in class_stats:
            class_stats[class_name] = []
        class_stats[class_name].append(track_id)
    
    # Выводим статистику
    for class_name, ids in class_stats.items():
        print(f"Класс: {class_name}")
        print(f"Количество: {len(ids)}")
        print(f"ID объектов: {', '.join(map(str, ids))}")
        print("----------------------------")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video("IMG_1811.MOV")