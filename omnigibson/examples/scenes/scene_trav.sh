#!/bin/bash

# 定义 Python 脚本路径
PYTHON_SCRIPT="omnigibson/examples/scenes/scene_trav.py"

# 定义场景列表
scenes_array=('Beechwood_0_garden' 'Beechwood_0_int' 'Beechwood_1_int' 'Benevolence_0_int' 'Benevolence_1_int' 'Benevolence_2_int' 'Ihlen_0_int' 'Ihlen_1_int' 'Merom_0_garden' 'Merom_0_int' 'Merom_1_int' 'Pomaria_0_garden' 'Pomaria_0_int' 'Pomaria_1_int' 'Pomaria_2_int' 'Rs_garden' 'Rs_int' 'Wainscott_0_garden' 'Wainscott_0_int' 'Wainscott_1_int' 'grocery_store_asian' 'grocery_store_cafe' 'grocery_store_convenience' 'grocery_store_half_stocked' 'hall_arch_wood' 'hall_conference_large' 'hall_glass_ceiling' 'hall_train_station' 'hotel_gym_spa' 'hotel_suite_large' 'hotel_suite_small' 'house_double_floor_lower' 'house_double_floor_upper' 'house_single_floor' 'office_bike' 'office_cubicles_left' 'office_cubicles_right' 'office_large' 'office_vendor_machine' 'restaurant_asian' 'restaurant_brunch' 'restaurant_cafeteria' 'restaurant_diner' 'restaurant_hotel' 'restaurant_urban' 'school_biology' 'school_chemistry' 'school_computer_lab_and_infirmary' 'school_geography' 'school_gym')

# 获取开始遍历的下标
start_index=${1:-0}

# 循环执行 Python 脚本，每次使用不同的 scene_model
for ((i=start_index; i<${#scenes_array[@]}; i++))
do
    scene_model=${scenes_array[$i]}
    echo "Running scene_trav.py with scene_model: $scene_model"
    python3 $PYTHON_SCRIPT --scene_model $scene_model
done