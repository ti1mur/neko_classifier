import torch
from torchvision import models
import torch.nn as nn
from base_line import imshow, evaluate, get_loaders

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = 1280

    data_loaders, datasets_sizes, classes = get_loaders(train_batch_size=100, test_batch_size=70, train_size=0.7)

    model = models.efficientnet_b0(weights='DEFAULT')  # инициализация модели
    model.classifier[1] = nn.Linear(num_features, 2)  # замена последнего слоя
    model.load_state_dict(torch.load("full_model.pth"))  # загрузка весов

    for batch in data_loaders["test"]:  # визуализация нескольких примеров из тестового множества
        inputs, labels = batch
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)

        for i in range(len(inputs)):
            image = inputs[i]
            label = labels[i]
            label_pred = pred[i]
            imshow(image, title=f"True_label: {classes[label]} pred_label: {classes[label_pred]}")
        break

    # подсчет точности для полной модели
    acc1 = evaluate(model, data_loaders["test"], datasets_sizes["test"])
    # подсчет точности для feature extractor модели
    model.load_state_dict(torch.load("ftrs_extr_model.pth"))
    acc2 = evaluate(model, data_loaders["test"], datasets_sizes["test"])
    # подсчет точности для fine tune модели
    model.load_state_dict(torch.load("fine_tune.pth"))
    acc3 = evaluate(model, data_loaders["test"], datasets_sizes["test"])
    print(f"Full_model acc: {acc1}, ftrs_extr_model acc: {acc2}, fine_tune_model acc {acc3}")
