import os
import subprocess
import time
import json
from multiprocessing import Pool
from src.utils.config_loader import config
from bddl_gen.read_files import process_all_bddl_files
def cac_bddl_num(root_dir):
    bddl_files = []
    for subdir, dirs, files in os.walk(root_dir):
        
        for file in files:
            if file.endswith(".bddl"):
                bddl_file = os.path.join(subdir, file)
                bddl_files.append(bddl_file)
    return bddl_files ,len(bddl_files)
def cac_scene_num(root_dir):
    total_files = 0
    success_files = []
    for subdir, dirs, files in os.walk(root_dir):
        total_files += len(files)  # 累加所有文件的数量
        for file in files:
            if file.endswith(".json"):
                success_files.append(file)
    return success_files,total_files if total_files > 0 else 0  
def main():
    room_list = ['Beechwood_0_garden', 'Beechwood_0_int', 'Beechwood_1_int', 'Benevolence_0_int', 'Benevolence_1_int', 'Benevolence_2_int', 'Ihlen_0_int', 'Ihlen_1_int', 'Merom_0_garden', 'Merom_0_int', 'Merom_1_int', 'Pomaria_0_garden', 'Pomaria_0_int', 'Pomaria_1_int', 'Pomaria_2_int', 'Rs_garden', 'Rs_int', 'Wainscott_0_garden', 'Wainscott_0_int', 'Wainscott_1_int', 'gates_bedroom', 'grocery_store_asian', 'grocery_store_cafe', 'grocery_store_convenience', 'grocery_store_half_stocked', 'hall_arch_wood', 'hall_conference_large', 'hall_glass_ceiling', 'hall_train_station', 'hotel_gym_spa', 'hotel_suite_large', 'hotel_suite_small', 'house_double_floor_lower', 'house_double_floor_upper', 'house_single_floor', 'office_bike', 'office_cubicles_left', 'office_cubicles_right', 'office_large', 'office_vendor_machine', 'restaurant_asian', 'restaurant_brunch', 'restaurant_cafeteria', 'restaurant_diner', 'restaurant_hotel', 'restaurant_urban', 'school_biology', 'school_chemistry', 'school_computer_lab_and_infirmary', 'school_geography', 'school_gym']
    result = {}
    preprocessed_task_list_path = config['scene_generation']['preprocessed_task_list_path']
    bddl_dir_base = config['scene_generation']['task_definitions_dir']
    scene_dir_base = config['scene_generation']['output_dir']
    with open(preprocessed_task_list_path, "w") as file:
        for id,room in enumerate(room_list):
            bddl_dir = os.path.join(bddl_dir_base, room)
            scene_dir = os.path.join(scene_dir_base, room)
            
            bddl_list, bddl_num = cac_bddl_num(bddl_dir)
            scene_list, scene_num = cac_scene_num(scene_dir)
            if bddl_num > 0:

                file.write(f"Room: {room}\n Room id: {id}\n")
                for bddl_file in bddl_list:
                    file.write(f"'{bddl_file}'\n")  # 用单引号框住每个文件名
                file.write("\n")  # 每个房间的信息之间加一个空行
    process_all_bddl_files(bddl_dir_base)
if __name__ == "__main__":
    main()