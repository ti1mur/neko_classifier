# neko_classifier

Проект по классификации neko-girl с помощью pre-train efficientnetb0. Датасет собран из открытых источников, самостоятельно размечен и реализован на pytorch.
Задача: Бинарная классификация
  1. class neko - 1: neko-girl. 40% от всего датасета.
  2. class cringe - 0: персонаж не neko. 60%.

## baseline.py:
### def get_loaders: 
  Функция для создания dataloader'ов и предобработки изображений.
### def model_train:
  Функция для обучения и валидации модели. Реализованы train loop, сохранение best parametrs, вычисление train loss и validation loss по эпохам, а так же визуализация обучения с помощью tqdm.
### def evaluate:
  Функция для оценки модели на тестовой выборке по метрике accuracy.
### def imshow:
  Преобразование тензора изображения в формат numpy, обратная нормализация, визуализация изображения.
### def plot_training:
  Визуализация train, val loss curves и сохранения на диск.

## castomdataset.py:
### class NekoDataset:
  Кастомный датасет, наследуемый от pytorch dataset. Изменены методы __getitem__, __len__ для корректной работы под кастомные аннотации. Добавлен метод find_errors для поиска битых файлов в датасете.
## main.py:
  Обучения модели с помощью baseline.py. Применение transfer learning для обучения pre-train efficientnetb0. 
  Обучены 3 модельки:
    1. Full model - обучение полной модельки без fine tune.
    2. Features extractor - обучение лишь полносвязных слоев.
    3. Fine tune model - заморозка нескольких первых слоев сети.
## model_test.py:
  Реализована визуализация предсказаний нескольких тестовых изображений. А так же подсчет accuracy для 3х моделей с помощью def evaluate.
## rename_script.py:
  Простой скрипт для ренейма файлов. Нужен был для дальнейшей легкой разметки.
  


