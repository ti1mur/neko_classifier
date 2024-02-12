# Импорт необходимых библиотек
from base_line import get_loaders, model_train, plot_training
from torchvision import models
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    # загрузим нужные данные
    data_loaders, datasets_sizes, classes = get_loaders(train_batch_size=32, test_batch_size=10, train_size=0.8)

    # model 1
    # num_features -- это размерность вектора фич, поступающего на вход FC-слою в efficientnet
    num_features = 1280
    model = models.efficientnet_b0(weights="DEFAULT")
    for param in model.parameters():  # замораживаем все параметры и заменяем последний FC-слой
        param.requires_grad = False
    model.classifier[1] = nn.Linear(num_features, 2)
    param_path = "fine_tune.pth"

    # инициализируем все необходимое для обучения
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # обучение модели
    _, losses = model_train(data_loaders, datasets_sizes, model, criterion, optimizer, exp_lr_scheduler, param_path,
                            num_epochs=10)
    plot_training(losses["train"], losses["test"], "fine_tune_losses.png")  # визуализируем потери

    # model 2
    model = models.efficientnet_b0(weights="DEFAULT")
    layers_to_unfreeze = 5

    # Выключаем подсчет градиентов для слоев, которые не будем обучать
    for param in model.features[:-layers_to_unfreeze].parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(num_features, 2)
    param_path = "ftrs_extr_model.pth"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    _, losses = model_train(data_loaders, datasets_sizes, model, criterion, optimizer, exp_lr_scheduler, param_path,
                            num_epochs=10)

    plot_training(losses["train"], losses["test"], "ftrs_extr_model.png")

    # model 3
    model = models.efficientnet_b0(weights="DEFAULT")

    # Заменяем Fully-Connected слой на наш линейный классификатор и обучим полную модель
    model.classifier[1] = nn.Linear(num_features, 2)

    param_path = "full_model.pth"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    _, losses = model_train(data_loaders, datasets_sizes, model, criterion, optimizer, exp_lr_scheduler, param_path,
                            num_epochs=10)

    plot_training(losses["train"], losses["test"], "full_model.png")
