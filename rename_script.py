import os


def rename_files(folder_path):  # переименовывает файлы по нужному формату
    files = os.listdir(folder_path)
    files.sort()
    for i, file_name in enumerate(files):
        file_path = os.path.join(folder_path, file_name)
        new_name = f"cringe{i}.png"  # Вы можете изменить формат имени по вашему усмотрению
        new_path = os.path.join(folder_path, new_name)

        os.rename(file_path, new_path)
        print(f"Переименован файл: {file_name} -> {new_name}")


# Укажите путь к вашей папке с изображениями
folder_path = r'C:\Users\timgo\PycharmProjects\india_style\neko_class\neko\not_neko'
rename_files(folder_path)
