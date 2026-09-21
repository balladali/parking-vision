"""
Модуль для работы с камерой или видеофайлом
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional, Generator
import os

from src import config


class Camera:
    """
    Класс для работы с видеопотоком (камера или файл)
    """
    
    def __init__(self, source=None, width=None, height=None):
        """
        Инициализация источника видео
        
        :param source: ID камеры (0, 1, ...) или путь к видеофайлу
        :param width: ширина кадра
        :param height: высота кадра
        """
        self.source = source if source is not None else config.VIDEO_SOURCE
        self.width = width if width is not None else config.FRAME_WIDTH
        self.height = height if height is not None else config.FRAME_HEIGHT
        self.cap = None
        
    def open(self) -> bool:
        """
        Открыть подключение к камере/видео
        
        :return: успешность подключения
        """
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            print(f"Ошибка: Не удалось открыть источник видео {self.source}")
            return False
            
        # Установка разрешения кадра
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        return True
        
    def close(self) -> None:
        """
        Закрыть подключение к камере/видео
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
            
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Получить один кадр из видеопотока
        
        :return: кортеж (успех, кадр)
        """
        if not self.cap or not self.cap.isOpened():
            if not self.open():
                return False, None
                
        ret, frame = self.cap.read()
        
        if not ret:
            print("Ошибка: Не удалось получить кадр")
            return False, None
            
        return True, frame
        
    def get_video_stream(self, fps_limit=None) -> Generator[np.ndarray, None, None]:
        """
        Генератор для получения последовательности кадров
        
        :param fps_limit: ограничение FPS (если None, то без ограничения)
        :yield: кадр
        """
        if not self.cap or not self.cap.isOpened():
            if not self.open():
                return
                
        prev_time = time.time()
        min_interval = 1.0 / fps_limit if fps_limit else 0
        
        try:
            while self.cap.isOpened():
                # Ограничение FPS при необходимости
                if fps_limit:
                    current_time = time.time()
                    elapsed = current_time - prev_time
                    
                    if elapsed < min_interval:
                        time.sleep(min_interval - elapsed)
                        
                    prev_time = time.time()
                
                ret, frame = self.cap.read()
                
                if not ret:
                    # Для видеофайла можно перезапустить при достижении конца
                    if isinstance(self.source, str) and os.path.exists(self.source):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                        
                yield frame
        finally:
            self.close()


# Пример использования
if __name__ == "__main__":
    camera = Camera()
    
    try:
        for frame in camera.get_video_stream(fps_limit=30):
            cv2.imshow('Camera Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows() 