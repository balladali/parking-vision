"""
ParkEye - Сервис определения свободных мест на парковке

Использование:
    python main.py - запуск в режиме реального времени с веб-камерой
    python main.py --image path/to/image.jpg - анализ одиночного изображения
    python main.py --source video.mp4 - анализ видеофайла
    python main.py --calibrate - запуск с калибровкой детектора
    python main.py --no-display - запуск без отображения интерфейса
    python main.py --mark-spots path/to/image.jpg - запуск утилиты для разметки мест
"""

import os
import sys
import time
import argparse
import cv2
import glob

from src import config
from src.service import ParkEyeService
from src.detector import ParkingDetector, put_text_with_russian
from src.utils import mark_parking_spots


def parse_arguments():
    """
    Обработка аргументов командной строки
    """
    parser = argparse.ArgumentParser(description="ParkEye - Сервис определения свободных мест на парковке")
    
    # Группа для режима работы
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--image', type=str, help='Путь к одиночному изображению для анализа')
    mode_group.add_argument('--source', type=str, help='Источник видео (0 для веб-камеры или путь к файлу)')
    mode_group.add_argument('--batch', type=str, help='Шаблон пути для пакетной обработки изображений (например, "images/*.jpg")')
    mode_group.add_argument('--mark-spots', type=str, help='Запуск утилиты для разметки парковочных мест на указанном изображении')
    
    parser.add_argument('--calibrate', action='store_true', help='Выполнить калибровку детектора перед запуском')
    parser.add_argument('--calibration-frames', type=int, default=50, help='Количество кадров для калибровки')
    parser.add_argument('--no-display', action='store_true', help='Запуск без отображения видео')
    parser.add_argument('--save', type=str, help='Сохранить результат анализа изображения в указанный файл')
    parser.add_argument('--output-dir', type=str, default='results', help='Директория для сохранения результатов пакетной обработки')
    parser.add_argument('--font', type=str, help='Путь к файлу шрифта для отображения текста')
    parser.add_argument('--threshold', type=float, help='Порог определения занятости (0.0-1.0)')
    parser.add_argument('--spots-config', type=str, help='Путь к файлу с конфигурацией мест (JSON)')
    parser.add_argument('--winter', action='store_true', help='Включить зимний режим для заснеженных парковок')
    
    return parser.parse_args()


def analyze_single_image(image_path, save_path=None, font_path=None, threshold=None, spots_config=None, winter_mode=False):
    """
    Анализ одиночного изображения
    
    :param image_path: путь к изображению
    :param save_path: путь для сохранения результата (опционально)
    :param font_path: путь к файлу шрифта (опционально)
    :param threshold: порог определения занятости (опционально)
    :param spots_config: путь к файлу с конфигурацией мест (опционально)
    :param winter_mode: активировать зимний режим для заснеженных парковок
    :return: статус выполнения
    """
    if not os.path.exists(image_path):
        print(f"Ошибка: Файл {image_path} не найден")
        return 1
        
    print(f"Анализ изображения: {image_path}")
    
    # Загрузка изображения
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return 1
        
    # Создание детектора с возможной пользовательской конфигурацией мест
    detector = ParkingDetector(spots_config_file=spots_config)
    
    # Если нет распознанных мест, выводим сообщение
    if not detector.spots:
        print("Ошибка: Не найдено настроенных парковочных мест")
        print("Сначала разметьте парковочные места с помощью --mark-spots")
        return 1
    
    # Обработка изображения
    # Используем метод для одиночных изображений вместо process_frame
    mode = 'winter' if winter_mode else None
    result_frame, statuses = detector.process_single_image(frame, threshold, mode)
    
    # Анализ результатов
    total_spots = len(statuses)
    free_spots = sum(1 for occupied in statuses.values() if not occupied)
    
    # Вывод результатов
    print(f"Результат анализа:")
    print(f"Всего мест: {total_spots}")
    print(f"Свободных мест: {free_spots}")
    print(f"Занятых мест: {total_spots - free_spots}")
    
    # Детализация по местам
    print("\nСтатус по местам:")
    for spot_id, is_occupied in statuses.items():
        print(f"Место #{spot_id}: {'Занято' if is_occupied else 'Свободно'}")
    
    # Отображение результата
    cv2.imshow('ParkEye - Анализ изображения', result_frame)
    print("Нажмите любую клавишу для продолжения...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Сохранение результата при необходимости
    if save_path:
        print(f"Сохранение результата в {save_path}")
        
        # Создание директории при необходимости
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        cv2.imwrite(save_path, result_frame)
        
        # Сохранение текстового отчета
        txt_path = os.path.splitext(save_path)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Анализ изображения: {image_path}\n")
            f.write(f"Режим: {'Зимний' if winter_mode else 'Автоопределение'}\n")
            f.write(f"Всего мест: {total_spots}\n")
            f.write(f"Свободных мест: {free_spots}\n")
            f.write(f"Занятых мест: {total_spots - free_spots}\n\n")
            f.write("Детали по местам:\n")
            
            for spot_id, is_occupied in statuses.items():
                f.write(f"Место #{spot_id}: {'Занято' if is_occupied else 'Свободно'}\n")
    
    return 0


def batch_process_images(input_pattern, output_dir, threshold=None, spots_config=None, winter_mode=False):
    """
    Пакетная обработка изображений
    
    :param input_pattern: шаблон пути к изображениям
    :param output_dir: директория для сохранения результатов
    :param threshold: порог определения занятости (опционально)
    :param spots_config: путь к файлу с конфигурацией мест (опционально)
    :param winter_mode: активировать зимний режим для заснеженных парковок
    :return: статус выполнения
    """
    print(f"Запуск пакетной обработки изображений по шаблону: {input_pattern}")
    print(f"Результаты будут сохранены в: {output_dir}")
    print(f"Режим обработки: {'Зимний' if winter_mode else 'Автоопределение'}")
    
    detector = ParkingDetector(spots_config_file=spots_config)
    
    # Если нет распознанных мест, выводим сообщение
    if not detector.spots:
        print("Ошибка: Не найдено настроенных парковочных мест")
        print("Сначала разметьте парковочные места с помощью --mark-spots")
        return 1
    
    # Обновляем метод для учета порога занятости
    success_count = 0
    
    # Поиск изображений по шаблону
    image_files = glob.glob(input_pattern)
    
    if not image_files:
        print(f"Предупреждение: Не найдено изображений по шаблону {input_pattern}")
        return 1
        
    print(f"Найдено {len(image_files)} изображений для обработки")
    
    # Создание директории для результатов
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, image_path in enumerate(image_files):
        print(f"Обработка изображения {i+1}/{len(image_files)}: {image_path}")
        
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка: Не удалось загрузить изображение {image_path}")
            continue
            
        # Обработка изображения
        mode = 'winter' if winter_mode else None
        result_frame, statuses = detector.process_single_image(image, threshold, mode)
        
        # Формирование имени выходного файла
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"analyzed_{base_name}")
        
        # Сохранение результатов
        if detector.save_results(output_path, result_frame, statuses):
            success_count += 1
            
    print(f"Обработка завершена. Успешно обработано {success_count}/{len(image_files)} изображений")
    
    if success_count > 0:
        return 0
    else:
        print("Не удалось обработать ни одного изображения")
        return 1


def main():
    """
    Основная функция
    """
    # Обработка аргументов командной строки
    args = parse_arguments()
    
    # Проверка режима разметки мест
    if args.mark_spots:
        print(f"Запуск утилиты разметки мест для изображения: {args.mark_spots}")
        output_path = args.spots_config if args.spots_config else None
        mark_parking_spots(args.mark_spots, output_path)
        return 0
        
    # Проверка режима одиночного изображения
    if args.image:
        return analyze_single_image(args.image, args.save, args.font, args.threshold, args.spots_config, args.winter)
        
    # Проверка режима пакетной обработки
    if args.batch:
        output_dir = args.output_dir if args.output_dir else "results"
        return batch_process_images(args.batch, output_dir, args.threshold, args.spots_config, args.winter)
    
    # Обновляем конфигурацию при необходимости
    if args.source is not None:
        try:
            # Если источник является числом, преобразуем его в int
            config.VIDEO_SOURCE = int(args.source)
        except ValueError:
            # Иначе считаем, что это путь к файлу
            config.VIDEO_SOURCE = args.source
    
    # Создаем сервис
    service = ParkEyeService()
    
    try:
        # Калибровка при необходимости
        if args.calibrate:
            print(f"Запуск калибровки на {args.calibration_frames} кадрах...")
            if service.calibrate(num_frames=args.calibration_frames):
                print("Калибровка завершена успешно")
            else:
                print("Ошибка при калибровке")
                return 1
        
        # Запуск сервиса
        print("Запуск сервиса ParkEye...")
        service.start()
        
        # Создаем папку для результатов, если необходимо
        if config.SAVE_RESULTS and not os.path.exists(config.RESULTS_PATH):
            os.makedirs(config.RESULTS_PATH)
        
        # Если нужно отображение
        if not args.no_display:
            print("Нажмите 'q' для выхода")
            
            try:
                while True:
                    # Получение и отображение текущего кадра
                    frame = service.get_current_frame()
                    
                    if frame is not None:
                        # Получение текущего статуса
                        status = service.get_current_status()
                        
                        # Добавление статистики на кадр
                        total_spots = status['total_spots']
                        free_spots = status['free_spots']
                        
                        # Вывод статистики
                        status_text = f"Свободно: {free_spots}/{total_spots}"
                        
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
                        
                        # Отображение кадра
                        cv2.imshow('ParkEye', frame)
                    
                    # Проверка нажатия клавиши 'q'
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
            except KeyboardInterrupt:
                print("\nПрервано пользователем")
            finally:
                cv2.destroyAllWindows()
        else:
            # Если режим без отображения, просто ждем прерывания
            print("Сервис запущен в фоновом режиме. Нажмите Ctrl+C для выхода.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nПрервано пользователем")
    finally:
        # Остановка сервиса
        print("Остановка сервиса...")
        service.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 