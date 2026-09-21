# parking-vision

Локальный Python 3.12+ CV PoC определения занятости размеченных парковочных мест
по видео фиксированной камеры. OpenCV читает видео, Ultralytics YOLO обнаруживает
транспорт, геометрический модуль сравнивает bounding boxes с полигонами мест.
Основной интерфейс — CLI. Backend, frontend, authentication и облачная инфраструктура отсутствуют.

## Установка

```powershell
git clone https://github.com/balladali/parking-vision.git
cd parking-vision
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
parking-vision --help
pytest
```

Linux/macOS: `python3.12 -m venv .venv`, затем `source .venv/bin/activate`;
остальные команды те же. Для CPU-only окружения можно предварительно установить
`torch torchvision` через `--index-url https://download.pytorch.org/whl/cpu`.
Установка зависимостей требует интернета. Тесты и `--help` не скачивают веса,
не обращаются к камере и не требуют GPU. Доступен `python -m parking_vision`.

## Запуск

1. Поместите собственное видео в `datasets/parking.mp4`.
2. Извлеките кадр: `python tools/extract_frame.py datasets/parking.mp4 datasets/frame.jpg`.
3. Укажите полигоны каждого места в `configs/example.yaml` по координатам этого кадра.
   Демонстрационные координаты нужно заменить. Вершины выпуклых полигонов задаются
   по периметру, в пикселях исходного кадра, начало координат — левый верхний угол.
   Камера и разрешение должны оставаться неизменными.
4. Проверьте конфигурацию и запустите обработку:

```sh
parking-vision validate --config configs/example.yaml
parking-vision run --config configs/example.yaml --max-frames 100 --output results.jsonl
```

Без `--output` JSONL пишется в stdout, логи — в stderr. Файл результата создаётся
эксклюзивно: существующий файл не перезаписывается. Без `--max-frames` обрабатывается
всё видео; камера работает до Ctrl+C. Коды выхода: 0 — успех, 1 — ошибка выполнения,
2 — ошибка аргументов, 130 — прерывание пользователем.

`source: 0` выбирает локальную камеру; строка задаёт локальный видеофайл.
Относительные пути видео и явные пути весов разрешаются от каталога YAML.
При первом запуске `yolo11n.pt` Ultralytics скачает предобученные веса.
Для работы без сети заранее скачайте веса и укажите путь, например
`weights: ../datasets/weights/yolo11n.pt`. Пользовательские веса требуют своих class IDs.
`validate` проверяет схему и геометрию, но не открывает видео и не запускает YOLO.

Пример строки результата:

```json
{"frame_index": 0, "free": 1, "total": 2, "spots": [{"spot_id": "A1", "occupied": true, "coverage": 0.8}, {"spot_id": "A2", "occupied": false, "coverage": 0.0}]}
```

Занятость: максимальная площадь пересечения одного vehicle box с полигоном,
делённая на площадь места, достигает `occupancy_threshold`. Это базовый PoC:
нет временного сглаживания, обучения или гарантии точности при перекрытиях,
снеге и сильной перспективе. Качество нужно измерять на размеченных видео конкретной камеры.
Ошибка декодирования после прочитанных кадров файла трактуется OpenCV как конец видео.

## Структура

```text
src/                 # устанавливается как пакет parking_vision
  detection/         # адаптер Ultralytics YOLO
  occupancy/         # оценка занятости по геометрии
  video/             # чтение видео и освобождение ресурсов
  models/            # типизированные общие модели данных
tools/               # извлечение кадра для разметки
tests/               # автономные unit/integration tests
datasets/.gitkeep    # видео и веса не коммитятся
configs/             # YAML
docker/Dockerfile
pyproject.toml
legacy/              # исходный ParkEye до миграции, вне нового пакета и тестов
```

`legacy/` сохранён для справки; новый CLI не использует старые эвристики или зимний режим.

## Docker (CPU)

```sh
docker build -f docker/Dockerfile -t parking-vision .
docker run --rm parking-vision --help
docker run --rm -v "${PWD}/datasets:/app/datasets:ro" -v "${PWD}/configs:/app/configs:ro" parking-vision run --config configs/example.yaml --max-frames 100 > results.jsonl
```

Для запуска без сети смонтируйте заранее скачанные веса вместе с datasets и укажите
явный путь в YAML. Контейнер не требует GUI. Доступ к USB-камере зависит от ОС;
примеры Docker предназначены для видеофайлов.

API детектора: [Ultralytics Predict](https://docs.ultralytics.com/modes/predict/).
