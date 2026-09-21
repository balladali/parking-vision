"""
Основной сервис для управления системой определения свободных парковочных мест
"""

import cv2
import numpy as np
import time
import os
import threading
import logging
from datetime import datetime
from typing import Dict, Optional, List, Tuple

from src import config
from src.camera import Camera
from src.detector import ParkingDetector


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("parkeye.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ParkEye")


class ParkEyeService:
    """
    Основной сервис для управления системой определения свободных парковочных мест
    """
    
    def __init__(self):
        """
        Инициализация сервиса
        """
        self.camera = Camera()
        self.detector = ParkingDetector()
        
        # Создаем результирующий каталог, если он не существует
        if config.SAVE_RESULTS and not os.path.exists(config.RESULTS_PATH):
            os.makedirs(config.RESULTS_PATH)
            
        # Состояние сервиса
        self.running = False
        self.processing_thread = None
        
        # Результат последней обработки
        self.last_frame = None
        self.last_statuses = {}
        self.last_update_time = time.time()
        
        # История изменений для анализа
        self.status_history = []
        
    def start(self) -> bool:
        """
        Запуск сервиса
        
        :return: успешность запуска
        """
        if self.running:
            logger.warning("Сервис уже запущен")
            return False
            
        logger.info("Запуск сервиса ParkEye")
        
        # Запуск обработки в отдельном потоке
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        return True
        
    def stop(self) -> bool:
        """
        Остановка сервиса
        
        :return: успешность остановки
        """
        if not self.running:
            logger.warning("Сервис не запущен")
            return False
            
        logger.info("Остановка сервиса ParkEye")
        
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
            
        return True
        
    def get_current_status(self) -> Dict:
        """
        Получить текущий статус парковки
        
        :return: словарь с текущей информацией
        """
        total_spots = len(self.last_statuses)
        free_spots = sum(1 for occupied in self.last_statuses.values() if not occupied)
        
        return {
            "timestamp": time.time(),
            "total_spots": total_spots,
            "free_spots": free_spots,
            "occupied_spots": total_spots - free_spots,
            "statuses": self.last_statuses,
            "last_update": self.last_update_time
        }
        
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Получить последний обработанный кадр
        
        :return: кадр с разметкой
        """
        return self.last_frame
        
    def _processing_loop(self) -> None:
        """
        Основной цикл обработки
        """
        logger.info("Запуск цикла обработки")
        
        try:
            for frame in self.camera.get_video_stream(fps_limit=30):
                if not self.running:
                    break
                    
                # Обработка кадра
                result_frame, statuses = self.detector.process_frame(frame)
                
                # Сохранение результатов
                self.last_frame = result_frame
                self.last_statuses = statuses
                self.last_update_time = time.time()
                
                # Добавление в историю
                self.status_history.append({
                    "timestamp": time.time(),
                    "statuses": statuses.copy()
                })
                
                # Ограничение размера истории
                if len(self.status_history) > 1000:
                    self.status_history = self.status_history[-1000:]
                    
                # Сохранение результатов при необходимости
                if config.SAVE_RESULTS:
                    self._save_results(result_frame, statuses)
                    
                # Проверка необходимости отправки уведомлений
                if config.ENABLE_NOTIFICATIONS:
                    self._check_notifications(statuses)
        except Exception as e:
            logger.error(f"Ошибка в цикле обработки: {e}")
        finally:
            logger.info("Завершение цикла обработки")
            
    def _save_results(self, frame: np.ndarray, statuses: Dict[int, bool]) -> None:
        """
        Сохранение результатов
        
        :param frame: кадр с разметкой
        :param statuses: статусы парковочных мест
        """
        try:
            # Сохранение кадра каждые 5 минут
            current_time = time.time()
            if current_time - self.last_update_time >= 300:  # 5 минут
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Сохранение изображения
                image_path = os.path.join(config.RESULTS_PATH, f"parkeye_{timestamp}.jpg")
                cv2.imwrite(image_path, frame)
                
                # Сохранение статусов
                total_spots = len(statuses)
                free_spots = sum(1 for occupied in statuses.values() if not occupied)
                
                status_summary = (
                    f"Timestamp: {timestamp}\n"
                    f"Total spots: {total_spots}\n"
                    f"Free spots: {free_spots}\n"
                    f"Occupied spots: {total_spots - free_spots}\n"
                    f"Statuses: {statuses}\n"
                )
                
                status_path = os.path.join(config.RESULTS_PATH, f"parkeye_{timestamp}.txt")
                with open(status_path, "w") as f:
                    f.write(status_summary)
        except Exception as e:
            logger.error(f"Ошибка при сохранении результатов: {e}")
            
    def _check_notifications(self, statuses: Dict[int, bool]) -> None:
        """
        Проверка необходимости отправки уведомлений
        
        :param statuses: статусы парковочных мест
        """
        # Здесь можно реализовать логику уведомлений
        # Например, если количество свободных мест изменилось или стало равно нулю
        pass

    def calibrate(self, num_frames: int = 50) -> bool:
        """
        Калибровка детектора
        
        :param num_frames: количество кадров для калибровки
        :return: успешность калибровки
        """
        logger.info(f"Запуск калибровки на {num_frames} кадрах")
        
        if self.running:
            logger.warning("Невозможно выполнить калибровку во время работы сервиса")
            return False
            
        try:
            calibration_frames = []
            
            # Сбор кадров для калибровки
            for i, frame in enumerate(self.camera.get_video_stream(fps_limit=10)):
                if i >= num_frames:
                    break
                calibration_frames.append(frame)
                
            # Калибровка детектора
            self.detector.calibrate(calibration_frames)
            
            logger.info(f"Калибровка завершена успешно на {len(calibration_frames)} кадрах")
            return True
        except Exception as e:
            logger.error(f"Ошибка при калибровке: {e}")
            return False
            
    def reset(self) -> None:
        """
        Сброс состояния сервиса
        """
        if self.running:
            self.stop()
            
        self.detector.reset()
        self.last_frame = None
        self.last_statuses = {}
        self.last_update_time = time.time()
        self.status_history = []


# Пример использования
if __name__ == "__main__":
    service = ParkEyeService()
    
    try:
        # Калибровка детектора
        service.calibrate(num_frames=30)
        
        # Запуск сервиса
        service.start()
        
        # Основной цикл для демонстрации
        while True:
            # Получение и отображение текущего кадра
            frame = service.get_current_frame()
            
            if frame is not None:
                # Получение текущего статуса
                status = service.get_current_status()
                
                # Вывод статистики
                status_text = f"Свободно: {status['free_spots']}/{status['total_spots']}"
                
                cv2.putText(
                    frame, 
                    status_text, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 255, 255), 
                    2, 
                    cv2.LINE_AA
                )
                
                cv2.imshow('ParkEye Service', frame)
                
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Прервано пользователем")
    finally:
        # Остановка сервиса
        service.stop()
        cv2.destroyAllWindows() 