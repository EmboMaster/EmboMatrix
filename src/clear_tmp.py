import os
import time
import shutil
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clean_large_tmp_dirs.log'),
        logging.StreamHandler()
    ]
)

# 常量
CHECK_INTERVAL = 600  # 10 分钟（秒）
SIZE_THRESHOLD = 20 * 1024 * 1024  # 100MB（字节）
TIME_THRESHOLD = 50 * 60  # 50 分钟（秒）
MAX_AGE_THRESHOLD = 3 * 24 * 60 * 60  # 3 天（秒）

def get_dir_size(dir_path):
    """计算文件夹总大小（字节）"""
    total_size = 0
    try:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                except (OSError, PermissionError):
                    continue
    except (OSError, PermissionError) as e:
        logging.error(f"Failed to calculate size of {dir_path}: {e}")
        return 0
    return total_size

def get_dir_ctime(dir_path):
    """获取文件夹的创建时间（秒）"""
    try:
        return os.path.getctime(dir_path)
    except (OSError, PermissionError) as e:
        logging.error(f"Failed to get ctime of {dir_path}: {e}")
        return float('inf')  # 防止误删

def get_dir_mtime(dir_path):
    """获取文件夹的修改时间（秒）"""
    try:
        return os.path.getmtime(dir_path)
    except (OSError, PermissionError) as e:
        logging.error(f"Failed to get mtime of {dir_path}: {e}")
        return float('inf')  # 防止误删

def clean_large_tmp_dirs():
    """检查 /tmp 下的文件夹，删除大小 > 100MB、创建时间 > 50 分钟且不超过 3 天的文件夹"""
    tmp_dir = "/tmp"
    current_time = time.time()

    if not os.path.exists(tmp_dir):
        logging.error(f"Directory {tmp_dir} does not exist")
        return

    # 遍历 /tmp 下的直接子目录
    for entry in os.listdir(tmp_dir):
        dir_path = os.path.join(tmp_dir, entry)
        
        # 只处理文件夹
        if not os.path.isdir(dir_path):
            continue

        # 计算文件夹大小
        dir_size = get_dir_size(dir_path)
        if dir_size < SIZE_THRESHOLD:
            continue

        # 获取创建时间和修改时间
        dir_ctime = get_dir_ctime(dir_path)
        dir_mtime = get_dir_mtime(dir_path)
        time_elapsed = current_time - dir_ctime

        # 检查是否超过 3 天
        ctime_age = current_time - dir_ctime
        mtime_age = current_time - dir_mtime
        if ctime_age > MAX_AGE_THRESHOLD or mtime_age > MAX_AGE_THRESHOLD:
            logging.info(
                f"Skipped directory: {dir_path}, "
                f"Size: {dir_size / (1024 * 1024):.2f} MB, "
                f"Created: {datetime.fromtimestamp(dir_ctime).strftime('%Y-%m-%d %H:%M:%S')}, "
                f"Modified: {datetime.fromtimestamp(dir_mtime).strftime('%Y-%m-%d %H:%M:%S')}, "
                f"Reason: Older than 3 days (ctime_age: {ctime_age / (24 * 3600):.2f} days, mtime_age: {mtime_age / (24 * 3600):.2f} days)"
            )
            continue

        # 检查创建时间是否超过 50 分钟
        if time_elapsed > TIME_THRESHOLD:
            try:
                shutil.rmtree(dir_path, ignore_errors=True)
                logging.info(
                    f"Deleted directory: {dir_path}, "
                    f"Size: {dir_size / (1024 * 1024):.2f} MB, "
                    f"Created: {datetime.fromtimestamp(dir_ctime).strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"Modified: {datetime.fromtimestamp(dir_mtime).strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"Age: {time_elapsed / 60:.2f} minutes"
                )
            except Exception as e:
                logging.error(f"Failed to delete directory {dir_path}: {e}")
        else:
            logging.info(
                f"Skipped directory: {dir_path}, "
                f"Size: {dir_size / (1024 * 1024):.2f} MB, "
                f"Created: {datetime.fromtimestamp(dir_ctime).strftime('%Y-%m-%d %H:%M:%S')}, "
                f"Modified: {datetime.fromtimestamp(dir_mtime).strftime('%Y-%m-%d %H:%M:%S')}, "
                f"Age: {time_elapsed / 60:.2f} minutes (too recent)"
            )

def main():
    """主循环，每 10 分钟检查一次 /tmp"""
    while True:
        logging.info("Starting check of /tmp directories")
        clean_large_tmp_dirs()
        logging.info("Check completed. Waiting for 10 minutes...")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Script terminated by user")