import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import time
from tqdm import trange, tqdm
from customdataset import NekoDataset
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_loaders(train_batch_size, test_batch_size, train_size):  # функция для загрузки данных
    classes = ('Cringe', 'Neko')

    transform = transforms.Compose([  # список трансформаций для корректного взаимодействия с входами
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = NekoDataset(csv_file='annotation.csv', root_dir='sfw', transform=transform)
    len_train_set = int(len(dataset) * train_size)
    len_test_set = len(dataset) - len_train_set

    train_set, test_set = torch.utils.data.random_split(dataset, [len_train_set, len_test_set])
    train_loader = DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True, num_workers=2,
                              pin_memory=True)
    test_loader = DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=False, num_workers=2,
                             pin_memory=True)

    datasets_sizes = {"train": len(train_set), "test": len(test_set)}
    data_loaders = {"train": train_loader, "test": test_loader}

    return data_loaders, datasets_sizes, classes


# обучение модели
def model_train(data_loaders, dataset_sizes, model, criterion, optimizer, scheduler, param_path, num_epochs=25):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.cuda.empty_cache()  # очистим неиспользованную видеопамять
    device = torch.device("cuda") if use_gpu else torch.device("cpu")
    model = model.to(device)  # переводим модель на gpu если оно доступно

    start = time.time()
    best_model_param_path = param_path
    best_acc = 0.0
    losses = {"train": [], "test": []}
    pbar = trange(num_epochs, position=0, desc="Epoch: ")

    for epoch in pbar:
        # основной цикл обучения
        for phase in ["train", "test"]:  # в зависимости от фазы обучения по разному взаимодействуем с моделью
            if phase == "train":  # переводим модель в режим обучения или предсказания
                model.train(True)
            else:
                model.eval()
            running_loss = 0.0
            running_correct = 0

            for inputs, labels in tqdm(data_loaders[phase], position=1, leave=False, desc=f"{phase} iter:"):
                if use_gpu:
                    inputs, labels = inputs.to(device), labels.to(device)

                if phase == "train":
                    optimizer.zero_grad()  # обнуляем градиенты чтобы они не накапливались

                if phase == "test":
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)  # подсчитываем ошибку

                if phase == "train":  # вычисление градиентов и шаг оптимизатора
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels.data)

            if phase == "train":  # шаг оптимизатора learning rate
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct / dataset_sizes[phase]

            losses[phase].append(epoch_loss)

            pbar.set_description(f"{phase} Loss: {epoch_loss}, Acc: {epoch_acc}")
            if phase == "test" and epoch_acc > best_acc:  # сохраняем параметры если достигли лучшего показателя точности
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_param_path)

            if use_gpu:
                torch.cuda.empty_cache()  # очищаем кэш видеопамяти в конце эпохи

    time_elapsed = time.time() - start
    print(f"Well Done Solder time: {time_elapsed // 60}m {time_elapsed % 60}s")
    print(f"Best acc: {best_acc}, path_model: {param_path}")

    model.load_state_dict(torch.load(best_model_param_path))  # загружаем лучшие параметры

    return model, losses


def evaluate(model, test_loader, dataset_size):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()  # переводим модель в режим предсказания, не обновляем градиенты
    correct = 0

    for data in tqdm(test_loader, desc="Iter:"):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)

        correct += int(torch.sum(pred == labels))  # вычисляем кол-во совпадений между предсказанными лейблами и реальными

    acc = 100 * correct / dataset_size  # делим все правильные предсказания на общее кол-во лейблов
    print(f"test_acc: {acc}")

    return 100 * correct / dataset_size


def imshow(inp, title=None):
    # преобразование и визуализация тензора в картинку
    inp = inp.numpy().transpose((1, 2, 0))  # меняет порядок осей с C,H,W на H,W,C для matplotlib
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # обратная нормализация
    inp = np.clip(inp, 0, 1)  # "зажимает" значения пикселей в диапазоне 0-1 для предотвращения выхода значений пикселей за пределы после нормализации
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)


def plot_training(train_losses, valid_losses, path):  # функция для визуализации потерь
    sns.set(font_scale=1.4, style="whitegrid")
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 1, 1)
    plt.xlabel("epoch")
    plt.plot(train_losses, label="train_losses")
    plt.plot(valid_losses, label="valid_losses")
    plt.legend()
    plt.pause(10)
    plt.savefig(path)
