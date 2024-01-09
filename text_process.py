import os


def merge_files(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    # 创建新文件的文件名
    new_file_name = os.path.basename(folder_path) + '.txt'
    new_file_path = os.path.join(folder_path, new_file_name)
    # 以写入模式打开新文件
    with open(new_file_path, 'w', encoding='utf-8') as nf:
        # 以读取模式打开要合并的文件
        for file in files:
            file = os.path.join(folder_path, file)
            with open(file, 'r', encoding='utf-8') as f:
                # 读取文件的所有行并写入新文件
                lines_in_file = f.readlines()
                nf.writelines(lines_in_file)
                # 读取每个文件后插入一个换行符
                nf.write("\n")


def split_file(file_path, chunk_size):
    # 获取文件名和扩展名
    file_name, file_ext = os.path.splitext(file_path)

    folder_path = '{}/'.format(file_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 创建新文件的文件名
    new_file_name = folder_path + 'part'
    # 打开要拆分的文件
    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取文件的所有行
        lines_in_file = f.readlines()
        # 将文件拆分为多个文件
        num = 0
        i = 0
        while i < len(lines_in_file):
            # 创建新文件的文件名
            new_file_path = new_file_name + str(num) + file_ext

            num += 1
            count = 0
            content = []
            # 以写入模式打开新文件
            with open(new_file_path, 'w', encoding='utf-8') as nf:
                # 写入文件的所有行
                while count < chunk_size and i < len(lines_in_file):
                    count = count + len(lines_in_file[i])
                    content.append(lines_in_file[i])
                    i = i + 1
                nf.writelines(content)


# 将函数应用于指定的文件
folder_path = './dataset/raw_data/THUCNews/'
files = os.listdir(folder_path)
chunk_size = 100 * 1024

for file in files:
    file_path = os.path.join(folder_path, file)
    split_file(file_path, chunk_size)

