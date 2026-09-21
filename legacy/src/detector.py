"""
Модуль для обнаружения и анализа парковочных мест
"""

import glob
import json
import os
import time
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src import config


def put_text_with_russian(img, text, position, font_size=22, color=(255, 255, 255), thickness=2, font_path=None):
    """
    Функция для отображения текста с поддержкой русского языка
    
    :param img: Изображение (numpy array)
    :param text: Текст для отображения
    :param position: Позиция (x, y)
    :param font_size: Размер шрифта
    :param color: Цвет текста (B, G, R)
    :param thickness: Толщина текста
    :param font_path: Путь к файлу шрифта (если None, используется шрифт по умолчанию)
    :return: Изображение с текстом
    """
    # Конвертируем OpenCV изображение (BGR) в формат PIL (RGB)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # Пытаемся загрузить шрифт для поддержки кириллицы
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Пытаемся найти стандартный шрифт в системе
            font_paths = [
                # Windows шрифты
                "C:/Windows/Fonts/Arial.ttf",
                "C:/Windows/Fonts/Calibri.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
                # Linux шрифты
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                # macOS шрифты
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc"
            ]
            
            font = None
            for path in font_paths:
                if os.path.exists(path):
                    font = ImageFont.truetype(path, font_size)
                    break
                    
            if font is None:
                # Если не нашли ни один шрифт, используем шрифт по умолчанию
                font = ImageFont.load_default()
    except Exception as e:
        print(f"Ошибка при загрузке шрифта: {e}")
        font = ImageFont.load_default()
    
    # Конвертируем BGR цвет (OpenCV) в RGB (PIL)
    rgb_color = (color[2], color[1], color[0])
    
    # Рисуем текст
    draw.text(position, text, font=font, fill=rgb_color)
    
    # Конвертируем обратно в формат OpenCV
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv2_img


class ParkingSpot:
    """
    Класс для представления парковочного места и его статуса
    """
    
    def __init__(self, spot_id: int, coords: Tuple[int, int, int, int]):
        """
        Инициализация парковочного места
        
        :param spot_id: идентификатор места
        :param coords: координаты (x, y, width, height)
        """
        self.id = spot_id
        self.x, self.y, self.width, self.height = coords
        self.occupied = False
        self.confidence = 0.0
        self.last_change_time = time.time()
        
    def get_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        Получить область интереса (ROI) для парковочного места
        
        :param frame: кадр изображения
        :return: вырезанная область парковочного места
        """
        return frame[self.y:self.y+self.height, self.x:self.x+self.width]
    

class ParkingDetector:
    """
    Класс для обнаружения свободных и занятых парковочных мест
    """
    
    def __init__(self, parking_spots=None, spots_config_file=None):
        """
        Инициализация детектора
        
        :param parking_spots: список парковочных мест
        :param spots_config_file: путь к файлу с конфигурацией мест (JSON)
        """
        self.spots = []
        
        # Если передан путь к файлу конфигурации, загружаем из него
        if spots_config_file and os.path.exists(spots_config_file):
            self.load_spots_from_file(spots_config_file)
        else:
            # Загрузка данных о парковочных местах из параметра или конфигурации
            spots_data = parking_spots if parking_spots is not None else config.PARKING_SPOTS
            
            for spot_data in spots_data:
                spot = ParkingSpot(spot_data["id"], spot_data["coords"])
                self.spots.append(spot)
            
        # Загрузка модели для детекции объектов (опционально)
        # Для простоты используем базовую обработку изображений
        self.use_ml_model = False
        self.model = None
        
        # Инициализация детектора заднего фона
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, 
            varThreshold=36, 
            detectShadows=True
        )
        
        # Для обработки первых кадров (адаптация детектора заднего фона)
        self.frames_processed = 0
        self.min_frames_for_bg = 10
        
    def load_spots_from_file(self, config_file: str) -> bool:
        """
        Загрузка данных о парковочных местах из файла JSON
        
        :param config_file: путь к файлу конфигурации
        :return: успешность загрузки
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                spots_data = json.load(f)
                
            # Очистка текущих мест
            self.spots = []
            
            # Загрузка новых мест
            for spot_data in spots_data:
                # Проверка формата данных
                if "id" in spot_data and "coords" in spot_data:
                    spot = ParkingSpot(spot_data["id"], spot_data["coords"])
                    self.spots.append(spot)
                else:
                    print(f"Предупреждение: некорректный формат данных для места: {spot_data}")
                    
            print(f"Загружено {len(self.spots)} парковочных мест из {config_file}")
            return len(self.spots) > 0
            
        except Exception as e:
            print(f"Ошибка при загрузке конфигурации парковочных мест: {e}")
            return False
        
    def load_ml_model(self) -> bool:
        """
        Загрузка моделей машинного обучения для улучшения детекции
        
        :return: успешность загрузки
        """
        # Здесь можно загрузить предобученную модель YOLOv5, EfficientDet и т.д.
        # Для простоты пока используем базовую обработку с OpenCV
        self.use_ml_model = False
        return False
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[int, bool]]:
        """
        Обработка кадра для определения статуса парковочных мест
        
        :param frame: кадр изображения
        :return: кортеж (обработанный кадр с разметкой, словарь статусов мест)
        """
        # Копия кадра для отрисовки результатов
        result_frame = frame.copy()
        
        # Обработка заднего фона
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Инкремент счетчика обработанных кадров
        self.frames_processed += 1
        
        # Карта статусов (id места -> занято)
        statuses = {}
        
        # Обработка каждого парковочного места
        for spot in self.spots:
            # Получение ROI для парковочного места
            roi = spot.get_roi(frame)
            roi_mask = spot.get_roi(fg_mask)
            
            # Подсчет количества пикселей переднего плана
            total_pixels = roi.shape[0] * roi.shape[1]
            fg_pixels = cv2.countNonZero(roi_mask)
            
            # Определение статуса места
            occupancy_ratio = fg_pixels / total_pixels if total_pixels > 0 else 0
            
            # Применение порога
            prev_status = spot.occupied
            spot.confidence = occupancy_ratio
            
            # Если не накопили достаточно кадров для BG Subtractor, считаем все места свободными
            if self.frames_processed < self.min_frames_for_bg:
                spot.occupied = False
            else:
                spot.occupied = occupancy_ratio > config.OCCUPANCY_THRESHOLD
                
            # Если статус изменился, обновляем время изменения
            if prev_status != spot.occupied:
                spot.last_change_time = time.time()
                
            # Сохранение статуса
            statuses[spot.id] = spot.occupied
            
            # Отрисовка места с цветом в зависимости от статуса
            color = (0, 0, 255) if spot.occupied else (0, 255, 0)  # Красный - занято, Зелёный - свободно
            cv2.rectangle(result_frame, (spot.x, spot.y), (spot.x + spot.width, spot.y + spot.height), color, 2)
            
            # Добавление текста с идентификатором и статусом (используем функцию для поддержки русского языка)
            status_text = f"ID: {spot.id}, {'Занято' if spot.occupied else 'Свободно'}"
            confidence_text = f"Conf: {spot.confidence:.2f}"
            
            # Используем нашу новую функцию для отображения русского текста
            result_frame = put_text_with_russian(
                result_frame,
                status_text,
                (spot.x, spot.y - 30),
                font_size=16,
                color=color
            )
            
            result_frame = put_text_with_russian(
                result_frame,
                confidence_text,
                (spot.x, spot.y - 60),
                font_size=16,
                color=color
            )
        
        # Добавляем общую информацию
        free_spots = sum(1 for occupied in statuses.values() if not occupied)
        total_spots = len(statuses)
        
        status_text = f"Свободно: {free_spots}/{total_spots}"
        result_frame = put_text_with_russian(
            result_frame,
            status_text,
            (10, 30),
            font_size=22,
            color=(255, 255, 255)
        )
            
        return result_frame, statuses
        
    def process_single_image(self, frame: np.ndarray, threshold=None, mode=None) -> Tuple[np.ndarray, Dict[int, bool]]:
        """
        Обработка одиночного изображения для определения статуса парковочных мест
        Использует другие методы вместо вычитания фона
        
        :param frame: изображение
        :param threshold: порог определения занятости (None - использовать значение из конфигурации)
        :param mode: режим обработки ('winter' - зимние условия, None - автоопределение)
        :return: кортеж (обработанный кадр с разметкой, словарь статусов мест)
        """
        # Копия кадра для отрисовки результатов
        result_frame = frame.copy()
        
        # Проверка на наличие снега
        is_snowy, snow_percent = self.is_snowy_image(frame)
        
        # Определение режима обработки
        winter_mode = mode == 'winter' or (mode is None and is_snowy)
        
        # Улучшаем контраст для заснеженных изображений
        if winter_mode:
            # Применяем улучшение контраста к копии изображения для обработки
            # Оригинальное изображение сохраняем для отображения
            frame_for_processing = self.enhance_snowy_image(frame)
        else:
            frame_for_processing = frame
        
        # Карта статусов (id места -> занято)
        statuses = {}
        
        # Порог определения занятости
        if threshold is not None:
            # Если пользователь задал порог явно, используем его
            occupancy_threshold = threshold
        else:
            # Используем соответствующий порог из конфигурации в зависимости от режима
            occupancy_threshold = config.WINTER_OCCUPANCY_THRESHOLD if winter_mode else config.SINGLE_IMAGE_THRESHOLD
        
        # Отладочная информация
        debug_info = {
            "mode": "winter" if winter_mode else "normal",
            "snow_percent": snow_percent,
            "threshold": occupancy_threshold
        }
        print(f"Режим обработки: {'Зимний' if winter_mode else 'Обычный'}, снега: {snow_percent:.1%}, порог: {occupancy_threshold:.2f}")
        
        # Обработка каждого парковочного места
        for spot in self.spots:
            # Получение ROI для парковочного места из обработанного изображения
            roi = spot.get_roi(frame_for_processing)
            
            # Преобразование в оттенки серого
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Применение размытия для уменьшения шума
            blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            
            # Извлечение признаков для определения наличия автомобиля
            
            # 1. Применение детекции краев Canny
            edges = cv2.Canny(blurred_roi, 50, 150)
            edge_density = cv2.countNonZero(edges) / (roi.shape[0] * roi.shape[1])
            
            # 2. Вычисление гистограммы цветов (для определения преобладающих цветов)
            hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            color_var = np.var(hist)  # Вариация цветов
            
            # 3. Адаптивная пороговая обработка для выделения объектов
            thresh = cv2.adaptiveThreshold(blurred_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
            thresh_density = cv2.countNonZero(thresh) / (roi.shape[0] * roi.shape[1])
            
            # 4. Вычисление стандартного отклонения как показателя текстуры
            std_dev = np.std(gray_roi) / 128.0  # Нормализация
            
            # 5. Оценка изменения градиента (используется для определения формы)
            sobelx = cv2.Sobel(blurred_roi, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred_roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = cv2.magnitude(sobelx, sobely)
            gradient_mean = np.mean(gradient_magnitude) / 255.0  # Нормализация
            
            # Комбинированный показатель занятости
            # Веса для каждого признака можно настроить
            occupancy_score = (
                0.35 * edge_density +       # Плотность краев
                0.20 * thresh_density +     # Плотность объектов после порога
                0.15 * std_dev +            # Вариация яркости
                0.20 * gradient_mean +      # Средний градиент
                0.10 * color_var            # Вариация цветов
            )
            
            # Определение статуса
            spot.confidence = occupancy_score
            
            # Для визуализации сохраним промежуточные результаты
            debug_images = {
                "edges": edges,
                "threshold": thresh,
                "gradient": cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            }
            
            # Финальное определение статуса
            spot.occupied = occupancy_score > occupancy_threshold
            
            # Сохранение статуса
            statuses[spot.id] = spot.occupied
            
            # Отрисовка места с цветом в зависимости от статуса
            color = (0, 0, 255) if spot.occupied else (0, 255, 0)  # Красный - занято, Зелёный - свободно
            cv2.rectangle(result_frame, (spot.x, spot.y), (spot.x + spot.width, spot.y + spot.height), color, 2)
            
            # Добавление текста с идентификатором и статусом
            status_text = f"ID: {spot.id}, {'Занято' if spot.occupied else 'Свободно'}"
            confidence_text = f"Conf: {spot.confidence:.2f}"
            
            # Используем функцию для отображения русского текста
            result_frame = put_text_with_russian(
                result_frame,
                status_text,
                (spot.x, spot.y - 30),
                font_size=16,
                color=color
            )
            
            result_frame = put_text_with_russian(
                result_frame,
                confidence_text,
                (spot.x, spot.y - 60),
                font_size=16,
                color=color
            )
        
        # Добавляем общую информацию
        free_spots = sum(1 for occupied in statuses.values() if not occupied)
        total_spots = len(statuses)
        
        status_text = f"Свободно: {free_spots}/{total_spots}"
        result_frame = put_text_with_russian(
            result_frame,
            status_text,
            (10, 30),
            font_size=22,
            color=(255, 255, 255)
        )
            
        return result_frame, statuses
        
    def reset(self) -> None:
        """
        Сброс состояния детектора
        """
        self.frames_processed = 0
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, 
            varThreshold=36, 
            detectShadows=True
        )
        
    def calibrate(self, frames: List[np.ndarray]) -> None:
        """
        Калибровка детектора на основе предоставленных кадров
        
        :param frames: список кадров для калибровки
        """
        for frame in frames:
            self.bg_subtractor.apply(frame)
            self.frames_processed += 1

    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """
        Загрузка изображения из файла
        
        :param image_path: путь к файлу изображения
        :return: изображение или None при ошибке
        """
        if not os.path.exists(image_path):
            print(f"Ошибка: Файл {image_path} не существует")
            return None
            
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Ошибка: Не удалось загрузить изображение {image_path}")
            return None
            
        return image
        
    @staticmethod
    def save_results(image_path: str, result_frame: np.ndarray, statuses: Dict[int, bool]) -> bool:
        """
        Сохранение результатов анализа
        
        :param image_path: путь для сохранения результата
        :param result_frame: обработанное изображение
        :param statuses: статусы парковочных мест
        :return: успешность сохранения
        """
        try:
            # Создание директории при необходимости
            directory = os.path.dirname(image_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            # Сохранение изображения
            cv2.imwrite(image_path, result_frame)
            
            # Сохранение текстового отчета
            txt_path = os.path.splitext(image_path)[0] + ".txt"
            
            total_spots = len(statuses)
            free_spots = sum(1 for occupied in statuses.values() if not occupied)
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Анализ парковки\n")
                f.write(f"Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Всего мест: {total_spots}\n")
                f.write(f"Свободных мест: {free_spots}\n")
                f.write(f"Занятых мест: {total_spots - free_spots}\n\n")
                
                f.write("Статус по местам:\n")
                for spot_id, is_occupied in statuses.items():
                    f.write(f"Место #{spot_id}: {'Занято' if is_occupied else 'Свободно'}\n")
                    
            return True
        except Exception as e:
            print(f"Ошибка при сохранении результатов: {e}")
            return False

    def batch_process_images(self, input_pattern: str, output_dir: str) -> int:
        """
        Пакетная обработка множества изображений
        
        :param input_pattern: шаблон пути к изображениям (например, "images/*.jpg")
        :param output_dir: директория для сохранения результатов
        :return: количество успешно обработанных изображений
        """
        # Создание выходной директории
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Поиск изображений по шаблону
        image_files = glob.glob(input_pattern)
        
        if not image_files:
            print(f"Предупреждение: Не найдено изображений по шаблону {input_pattern}")
            return 0
            
        print(f"Найдено {len(image_files)} изображений для обработки")
        
        success_count = 0
        
        for i, image_path in enumerate(image_files):
            print(f"Обработка изображения {i+1}/{len(image_files)}: {image_path}")
            
            # Загрузка изображения
            image = self.load_image(image_path)
            if image is None:
                continue
                
            # Обработка изображения (используем метод для одиночных изображений)
            result_frame, statuses = self.process_single_image(image)
            
            # Формирование имени выходного файла
            base_name = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"analyzed_{base_name}")
            
            # Сохранение результатов
            if self.save_results(output_path, result_frame, statuses):
                success_count += 1
                
        print(f"Обработка завершена. Успешно обработано {success_count}/{len(image_files)} изображений")
        return success_count

    def enhance_snowy_image(self, image: np.ndarray) -> np.ndarray:
        """
        Улучшение контраста на заснеженном изображении для лучшего обнаружения автомобилей
        
        :param image: исходное изображение
        :return: улучшенное изображение
        """
        # Преобразуем в LAB цветовое пространство (L - яркость, A,B - цветовые компоненты)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Разделяем на каналы
        l, a, b = cv2.split(lab)
        
        # Применяем CLAHE (адаптивное выравнивание гистограммы) к яркостному каналу
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Объединяем каналы обратно
        enhanced_lab = cv2.merge((cl, a, b))
        
        # Преобразуем обратно в BGR
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_image


# Пример использования
if __name__ == "__main__":
    from src.camera import Camera
    
    camera = Camera()
    detector = ParkingDetector()
    
    try:
        for frame in camera.get_video_stream(fps_limit=30):
            result_frame, statuses = detector.process_frame(frame)
            
            # Вывод общей информации (уже реализовано в process_frame)
            
            cv2.imshow('Parking Detector', result_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows() 