#python bddl_gen_test.py
import bddl_generation
import bddl_modify_object
import command_gen
import read_files
from datetime import datetime
import argparse
import time
import traceback
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import openai

import bddl_modify_object  


import yaml
import logging
import os
from src.utils.config_loader import config
def process_scene_task(scene_index,save_path,command_num):

    
    command_difficulty = 1
    
    max_retries = 2
    retry_delay = 30
    backoff_factor = 2.0
    
    try:
        import bddl_modify_object
    except ImportError:
        logging.error(f"[Scene {scene_index}] 无法导入 'bddl_modify_object' 模块。任务失败。")
        return (scene_index, "Failed: ModuleImportError")

    for attempt in range(max_retries + 1):
        try:
            logging.info(f"[Scene {scene_index}] 开始处理 (尝试 {attempt + 1}/{max_retries + 1})...")
            
            bddl_modify_object.generate_bddl_api(
                scene_index=scene_index,
                command_num=command_num,
                command_difficulty=command_difficulty,
                save_path=save_path
            )
            
            logging.info(f"[Scene {scene_index}] 成功完成。")
            return (scene_index, "Success") # 成功，立即返回
            
        except Exception as e:
            logging.warning(f"[Scene {scene_index}] 尝试 {attempt + 1} 失败: {e}")
            
            if attempt < max_retries:
                wait_time = retry_delay * (backoff_factor ** attempt)
                logging.info(f"[Scene {scene_index}] 在 {wait_time:.1f} 秒后重试...")
                time.sleep(wait_time)
            else:
                logging.error(f"[Scene {scene_index}] 所有 {max_retries + 1} 次尝试均失败。放弃。")
                return (scene_index, f"Failed: {e}") 

    return (scene_index, "Failed: UnknownError")

def main():
    """
    主函数：加载配置，设置并行池，并分发任务。
    """
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("generator.log"), 
            logging.StreamHandler()              
        ]
    )
    logging.info("--- 任务开始 ---")
    save_path = config['task_generation']['output_dir']
    command_num = config['task_generation']['command_num']
    start_index = config['task_generation']['range'][0]
    end_index = config['task_generation']['range'][1]
    
    max_workers = config['task_generation']['max_workers']
    
    if max_workers == 0 or max_workers is None:
        max_workers = None 
        logging.info("max_workers 设置为 'None' (将使用所有可用的 CPU 核心)。")
    elif max_workers == 1:
        logging.info("max_workers 设置为 1 (将顺序执行)。")
    else:
        logging.info(f"并行工作进程数设置为: {max_workers}")

    # 创建任务列表
    scene_indices = list(range(start_index, end_index))
    total_tasks = len(scene_indices)
    logging.info(f"准备处理 {total_tasks} 个场景 (索引从 {start_index} 到 {end_index - 1})...")

    # 确保保存路径存在
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        logging.info(f"确保保存路径存在: {save_path}")

    # --- 4. 执行并行任务 (要求 2) ---
    futures = []
    results = {"success": [], "failed": []}

    # ProcessPoolExecutor 是处理 CPU 密集型任务的首选
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        for index in scene_indices:
            future = executor.submit(process_scene_task, index,save_path,command_num)
            futures.append(future)
        
        logging.info(f"已提交 {total_tasks} 个任务到进程池。等待任务完成...")

        for i, future in enumerate(as_completed(futures)):
            try:
                scene_index, status = future.result()
                
                if status == "Success":
                    results["success"].append(scene_index)
                else:
                    results["failed"].append((scene_index, status))
                    
            except Exception as e:
                logging.error(f"一个工作进程意外崩溃: {e}")
                results["failed"].append(("Unknown", f"WorkerCrash: {e}"))
            
            logging.info(f"--- 进度: {i + 1}/{total_tasks} (已完成) ---")

    logging.info("--- 任务全部完成 ---")
    logging.info(f"总计成功: {len(results['success'])}")
    logging.info(f"总计失败: {len(results['failed'])}")
    
    if results['failed']:
        logging.warning("失败的场景列表:")
        for index, reason in results['failed']:
            logging.warning(f"  - 场景 {index}: {reason}")
    logging.info("--- 任务结束 ---")

if __name__ == "__main__":
    main()