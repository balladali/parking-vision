"""
Конфигурационный файл для проекта ParkEye
"""

# Настройки камеры/видео
VIDEO_SOURCE = 0  # 0 - веб-камера, или путь к видеофайлу 'path/to/video.mp4'
FRAME_WIDTH = 640  # ширина кадра
FRAME_HEIGHT = 480  # высота кадра

# Настройки детектора парковочных мест
# Координаты парковочных мест (x, y, width, height)
# 
# ВАЖНО: Эти координаты необходимо настроить под конкретное изображение парковки!
# Для каждого нового изображения или угла камеры необходимо задать свои координаты.
# 
# Пример настройки:
# 1. Загрузите изображение парковки
# 2. Определите координаты каждого парковочного места в формате (x, y, width, height),
#    где x, y - координаты верхнего левого угла прямоугольника, а width, height - его размеры
#
# Вы можете использовать следующий код для определения координат:
"""
# import cv2
# 
# def get_coords(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Сохраняем начальные координаты
#         param[0] = (x, y)
#     elif event == cv2.EVENT_LBUTTONUP:
#         # Вычисляем ширину и высоту
#         param[1] = (x - param[0][0], y - param[0][1])
#         print(f"Координаты: ({param[0][0]}, {param[0][1]}, {param[1][0]}, {param[1][1]})")
# 
# # Загружаем изображение
# img = cv2.imread('path_to_your_image.jpg')
# cv2.namedWindow('Select Parking Spots')
# coord_data = [(0, 0), (0, 0)]
# cv2.setMouseCallback('Select Parking Spots', get_coords, coord_data)
# 
# while True:
#     # Отображаем изображение
#     img_show = img.copy()
#     if coord_data[0] != (0, 0):
#         # Рисуем прямоугольник если начальные координаты выбраны
#         cv2.rectangle(img_show, coord_data[0], 
#                     (coord_data[0][0] + coord_data[1][0], coord_data[0][1] + coord_data[1][1]), 
#                     (0, 255, 0), 2)
#     
#     cv2.imshow('Select Parking Spots', img_show)
#     if cv2.waitKey(20) & 0xFF == 27:  # Выход по ESC
#         break
# 
# cv2.destroyAllWindows()
"""
# После определения координат, внесите их в список PARKING_SPOTS ниже

PARKING_SPOTS = [
    {"id": 1, "coords": (100, 200, 80, 150)},
    {"id": 2, "coords": (200, 200, 80, 150)},
    {"id": 3, "coords": (300, 200, 80, 150)},
    {"id": 4, "coords": (400, 200, 80, 150)},
    # Для более точного определения добавьте больше мест с правильными координатами
]

# Порог для определения, занято ли место (значение от 0.0 до 1.0)
# - Для видеопотока (при использовании вычитания фона): ~0.6
# - Для одиночных изображений (при прямом анализе): ~0.3 
# Чем выше значение, тем больше изменений требуется для определения места как занятого
OCCUPANCY_THRESHOLD = 0.6  # Для видеопотока
SINGLE_IMAGE_THRESHOLD = 0.2  # Для одиночных изображений

# Настройки для зимнего режима (заснеженные парковки)
WINTER_OCCUPANCY_THRESHOLD = 0.15  # Порог для зимнего режима
SNOW_DETECTION_THRESHOLD = 0.3  # Доля белых пикселей для определения снега
# Веса признаков для зимнего режима
WINTER_WEIGHTS = {
    'edge_density': 0.20,     # Плотность краев
    'thresh_density': 0.15,   # Плотность объектов
    'std_dev': 0.20,          # Вариация яркости
    'gradient_mean': 0.15,    # Средний градиент
    'color_var': 0.10,        # Вариация цветов
    'color_density': 0.20     # Плотность нетипичных для снега цветов
}

# Настройки веб-интерфейса
WEB_PORT = 5000
DEBUG_MODE = True

# Настройки уведомлений
ENABLE_NOTIFICATIONS = False
NOTIFICATION_INTERVAL = 60  # секунды

# Настройки для сохранения результатов
SAVE_RESULTS = True
RESULTS_PATH = "results"

# Пути к шрифтам по умолчанию
DEFAULT_FONTS = [
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