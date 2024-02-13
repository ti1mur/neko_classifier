# neko_classifier

Проект по классификации neko-girl с помощью pre-train efficientnetb0. Датасет собран из открытых источников, самостоятельно размечен и реализован на pytorch.
Задача: Бинарная классификация
  1. class neko - 1: neko-girl.
  2. class cringe - 0: персонаж не neko.

## baseline.py:
### def get_loaders: 
  Функция для создания dataloader'ов и предобработки изображений.
### def model_train:
  Функция для обучения и валидации модели. Реализованы train loop, сохранение best parametrs, вычисление train loss и validation loss по эпохам, а так же визуализация обучения с помощью tqdm.
  


