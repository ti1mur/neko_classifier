import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import trange

# кастомный класс датасета наследуемый от Pytorch dataset
class NekoDataset(Dataset):
    def __init__(self, root_dir, transform=None, csv_file=None):
        self.csv_file = csv_file  # путь к файлу с лейблами и именами картинок
        self.root_dir = root_dir  # путь к картинкам
        self.transform = transform  # необходимые преобразования для корректного взаимодействия с моделью
        self.annotations = self.load_annotations()  # загрузка лейблов

    def load_annotations(self):  # чтение данных из файла с лейблами
        if self.csv_file != None:
            return pd.read_csv(self.csv_file, sep=';')
        else:
            return None

    def __len__(self):  # определение длинны
        return len(self.annotations)

    def __getitem__(self, idx):  # определение метода с помощью которого получают доступ к данным в датасете
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        try:  # обработка исключения если файл некорректный
            image = Image.open(img_path).convert("RGB")
            label = self.annotations.iloc[idx, 1]
        except Exception:
            print(img_path)  # вывод пути "битого" файла
        else:
            if self.transform:
                image = self.transform(image)

            return image, label

    def find_error(self):  # функция для поиска некорректных изображений
        error_files = []
        for idx in trange(NekoDataset.__len__(self)):
            img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                error_files.append(img_path)
                print(img_path)
            finally:
                continue
        return error_files  # возвращает список некорректных файлов


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Добавьте другие трансформации по необходимости
    ])
    dataset = NekoDataset(csv_file='annotation.csv', root_dir='sfw', transform=transform)

    errors = dataset.find_error()

    print(errors)
