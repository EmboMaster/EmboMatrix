import os

def replace_in_file(file_path):
    """读取文件并进行替换操作"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 替换字符串
        content = content.replace("grasped", "attached")
        content = content.replace("attched", "attached")
        content = content.replace("isgrasping", "attached")
        content = content.replace("inroom", "inside")

        # 将修改后的内容写回文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Processed: {file_path}")
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def process_directory(directory):
    """递归处理目录及其子目录中的所有文件"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # 只处理文本文件，如果需要处理其他文件类型，添加条件
            if file_path.endswith('.bddl'):  # 这里假设只处理 .txt 文件
                replace_in_file(file_path)

if __name__ == "__main__":
    directory = input("请输入要处理的目录路径: ")  # 输入需要处理的目录路径
    process_directory(directory)
