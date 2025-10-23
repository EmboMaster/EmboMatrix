import glob
import os
import time
import schedule
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 目标目录
TARGET_DIR = "/data/embodied/planner_bench/"

def clean_core_files():
    """检查并删除创建超过2分钟的 core* 文件"""
    try:
        # 获取当前时间
        current_time = time.time()
        # 查找 core* 文件
        core_files = glob.glob(os.path.join(TARGET_DIR, "core*"))
        
        if not core_files:
            logging.info("No core* files found.")
            return

        for file_path in core_files:
            # 确保是文件而非目录
            if not os.path.isfile(file_path):
                continue
            
            # 获取文件创建时间（Linux 上可能是 mtime）
            file_stat = os.stat(file_path)
            file_creation_time = file_stat.st_mtime  # 使用修改时间作为近似创建时间
            file_age_minutes = (current_time - file_creation_time) / 60

            if file_age_minutes > 2:
                try:
                    os.remove(file_path)
                    logging.info(f"Deleted {file_path} (age: {file_age_minutes:.2f} minutes)")
                except Exception as e:
                    logging.error(f"Failed to delete {file_path}: {e}")
            else:
                logging.info(f"Skipped {file_path} (age: {file_age_minutes:.2f} minutes, too new)")

    except Exception as e:
        logging.error(f"Error during core file cleanup: {e}")

def main():
    """主函数，设置定时任务"""
    # 每 3 分钟运行一次
    schedule.every(1).minutes.do(clean_core_files)
    
    logging.info("Starting core file cleanup scheduler...")
    
    # 无限循环运行调度器
    while True:
        try:
            schedule.run_pending()
            time.sleep(30)  # 每分钟检查一次调度
        except KeyboardInterrupt:
            logging.info("Scheduler stopped by user.")
            break
        except Exception as e:
            logging.error(f"Scheduler error: {e}")

if __name__ == "__main__":
    main()