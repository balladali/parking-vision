"""
Утилиты для проекта ParkEye
"""

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Any

class ParkingSpotMarker:
    """
    Класс для интерактивной разметки парковочных мест на изображении
    """
    
    def __init__(self, image_path: str, output_path: str = None):
        """
        Инициализация
        
        :param image_path: путь к изображению для разметки
        :param output_path: путь для сохранения результатов (по умолчанию 'parking_spots.json')
        """
        self.image_path = image_path
        self.output_path = output_path or 'parking_spots.json'
        
        # Загрузка изображения
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
        # Копия исходного изображения
        self.original_image = self.image.copy()
        self.marking_image = self.image.copy()
        
        # Список парковочных мест
        self.spots = []
        self.current_spot_id = 1
        
        # Текущее состояние разметки
        self.drawing = False
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        
        # Режим: 'add' - добавление новых мест, 'edit' - редактирование
        self.mode = 'add'
        self.selected_spot_index = -1
        
    def mouse_callback(self, event, x, y, flags, param):
        """
        Обработчик событий мыши
        """
        if self.mode == 'add':
            self._handle_add_mode(event, x, y)
        elif self.mode == 'edit':
            self._handle_edit_mode(event, x, y)
            
    def _handle_add_mode(self, event, x, y):
        """
        Обработка событий мыши в режиме добавления
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Начало рисования
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Обновление временного прямоугольника
            self.marking_image = self.original_image.copy()
            self._draw_existing_spots()
            
            # Отрисовка текущего прямоугольника
            self.end_point = (x, y)
            cv2.rectangle(self.marking_image, self.start_point, self.end_point, (0, 255, 0), 2)
            
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            # Завершение рисования
            self.drawing = False
            self.end_point = (x, y)
            
            # Создание нового места
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            
            # Конвертация в формат (x, y, width, height)
            x = min(x1, x2)
            y = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            # Добавление места
            self._add_spot(x, y, width, height)
            
            # Обновление отображения
            self.marking_image = self.original_image.copy()
            self._draw_existing_spots()
    
    def _handle_edit_mode(self, event, x, y):
        """
        Обработка событий мыши в режиме редактирования
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Выбор места
            for i, spot in enumerate(self.spots):
                spot_x, spot_y, spot_width, spot_height = spot['coords']
                if (spot_x <= x <= spot_x + spot_width and 
                    spot_y <= y <= spot_y + spot_height):
                    self.selected_spot_index = i
                    self.drawing = True
                    self.start_point = (x, y)
                    break
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing and self.selected_spot_index >= 0:
            # Перемещение места
            dx = x - self.start_point[0]
            dy = y - self.start_point[1]
            
            spot = self.spots[self.selected_spot_index]
            spot_x, spot_y, spot_width, spot_height = spot['coords']
            
            # Обновление координат
            spot['coords'] = (spot_x + dx, spot_y + dy, spot_width, spot_height)
            
            # Обновление начальной точки
            self.start_point = (x, y)
            
            # Обновление отображения
            self.marking_image = self.original_image.copy()
            self._draw_existing_spots()
            
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.selected_spot_index = -1
    
    def _add_spot(self, x, y, width, height):
        """
        Добавление нового парковочного места
        """
        # Проверка минимального размера
        if width < 10 or height < 10:
            return
            
        spot = {
            'id': self.current_spot_id,
            'coords': (x, y, width, height)
        }
        
        self.spots.append(spot)
        self.current_spot_id += 1
        
    def _draw_existing_spots(self):
        """
        Отрисовка имеющихся парковочных мест
        """
        for i, spot in enumerate(self.spots):
            spot_id = spot['id']
            x, y, width, height = spot['coords']
            
            # Цвет зависит от выбранного места
            color = (0, 255, 255) if i == self.selected_spot_index else (0, 255, 0)
            
            # Рисуем прямоугольник
            cv2.rectangle(self.marking_image, (x, y), (x + width, y + height), color, 2)
            
            # Добавляем ID места
            cv2.putText(
                self.marking_image,
                f"ID: {spot_id}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )
    
    def run(self):
        """
        Запуск интерактивного режима разметки
        """
        window_name = "ParkEye - Разметка парковочных мест"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("=== Интерактивная разметка парковочных мест ===")
        print("Левая кнопка мыши - нарисовать место")
        print("'a' - режим добавления мест")
        print("'e' - режим редактирования мест")
        print("'d' - удалить последнее место")
        print("'c' - очистить все места")
        print("'s' - сохранить результаты")
        print("'q' или ESC - выход")
        
        while True:
            # Отображаем текущую информацию
            status_text = f"Режим: {'Добавление' if self.mode == 'add' else 'Редактирование'} | Мест: {len(self.spots)}"
            help_text = "a-добавить, e-редактировать, d-удалить, c-очистить, s-сохранить, q-выход"
            
            info_img = self.marking_image.copy()
            cv2.putText(info_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(info_img, help_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Показываем изображение
            cv2.imshow(window_name, info_img)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' или ESC
                break
                
            elif key == ord('a'):  # Режим добавления
                self.mode = 'add'
                print("Включен режим добавления мест")
                
            elif key == ord('e'):  # Режим редактирования
                self.mode = 'edit'
                print("Включен режим редактирования мест")
                
            elif key == ord('d'):  # Удаление последнего места
                if self.spots:
                    removed = self.spots.pop()
                    print(f"Удалено место с ID: {removed['id']}")
                    
                    # Обновление изображения
                    self.marking_image = self.original_image.copy()
                    self._draw_existing_spots()
                
            elif key == ord('c'):  # Очистка всех мест
                if self.spots:
                    conf = input("Вы уверены, что хотите удалить все места? (y/n): ")
                    if conf.lower() == 'y':
                        self.spots = []
                        self.current_spot_id = 1
                        print("Все места удалены")
                        
                        # Обновление изображения
                        self.marking_image = self.original_image.copy()
                
            elif key == ord('s'):  # Сохранение результатов
                self._save_results()
                
        cv2.destroyAllWindows()
        
    def _save_results(self):
        """
        Сохранение результатов разметки
        """
        if not self.spots:
            print("Нечего сохранять: нет размеченных мест")
            return
            
        # Сохранение в формате JSON
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(self.spots, f, indent=4)
                
            print(f"Разметка сохранена в {self.output_path}")
            
            # Также создаем код для config.py
            config_code = "PARKING_SPOTS = [\n"
            for spot in self.spots:
                x, y, width, height = spot['coords']
                config_code += f"    {{\"id\": {spot['id']}, \"coords\": ({x}, {y}, {width}, {height})}},\n"
            config_code += "]"
            
            # Сохраняем код в отдельный файл
            config_path = os.path.splitext(self.output_path)[0] + "_config.py"
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_code)
                
            print(f"Код для config.py сохранен в {config_path}")
            
            # Сохраняем изображение с разметкой
            image_path = os.path.splitext(self.output_path)[0] + "_marked.jpg"
            cv2.imwrite(image_path, self.marking_image)
            
            print(f"Изображение с разметкой сохранено в {image_path}")
            
            return True
        except Exception as e:
            print(f"Ошибка при сохранении: {e}")
            return False
            

def mark_parking_spots(image_path, output_path=None):
    """
    Запуск интерактивной разметки парковочных мест
    
    :param image_path: путь к изображению
    :param output_path: путь для сохранения результатов
    """
    try:
        marker = ParkingSpotMarker(image_path, output_path)
        marker.run()
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python utils.py <путь_к_изображению> [путь_для_сохранения]")
        sys.exit(1)
        
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    mark_parking_spots(image_path, output_path) 