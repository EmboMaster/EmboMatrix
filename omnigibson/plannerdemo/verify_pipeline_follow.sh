#!/bin/bash

# 定义基础路径
BASE_PATH="omnigibson"
PYTHON_SCRIPT="${BASE_PATH}/plannerdemo/verify_pipeline.py"
COMPUTE_SCRIPT="${BASE_PATH}/plannerdemo/verify_pipeline_compute.py"
JSON_FILE="${BASE_PATH}/feasible_scene/feasible_scene-eai-0821.json"
regenerate_flag=false

# 检查必要的文件是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

if [ ! -f "$COMPUTE_SCRIPT" ]; then
    echo "Error: Compute script not found at $COMPUTE_SCRIPT"
    exit 1
fi

if [ ! -f "$JSON_FILE" ]; then
    echo "Error: JSON file not found at $JSON_FILE"
    exit 1
fi


if [ "$regenerate_flag" = "true" ]; then
  echo "regenerate_flag is true, 提取feasible == true的场景..."
  # 初始化scene_files数组（任务队列）
  SCENE_FILES=($(jq -r 'to_entries | map(select(.value.feasible == true )) | .[].key' "$JSON_FILE"))
else
  echo "regenerate_flag is false, 提取feasible == false的场景..."
  # 初始化scene_files数组（任务队列）
  SCENE_FILES=($(jq -r 'to_entries | map(select(.value.feasible == false )) | .[].key' "$JSON_FILE"))
fi

# 初始化所有已处理的scene_file集合
declare -A ALL_SCENE_FILES

# 将初始的scene_files加入ALL_SCENE_FILES
for scene in "${SCENE_FILES[@]}"; do
    ALL_SCENE_FILES["$scene"]=1
done

# 检查是否成功获取初始scene files
if [ ${#SCENE_FILES[@]} -eq 0 ]; then
    echo "No initial scene files found in $JSON_FILE"
fi

# 定义mode和gpu_id的选择
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)
MAX_JOBS=8
COUNT_TASK=0

# 定义一个函数来运行任务
run_task() {
    local scene="$1"
    local gpu_id="$2"
    
    #echo "Running scene: $scene on GPU: $gpu_id"

    export PYTHONPATH="bddl:$PYTHONPATH"
    
    python3 "$PYTHON_SCRIPT" \
        --scene_file "$scene" \
        --gpu_id "$gpu_id" \
        --feasible_scene_file "$JSON_FILE" \
        --regenerate_flag "$regenerate_flag"
    
    if [ $? -eq 0 ]; then
        echo "Successfully completed: $scene on GPU: $gpu_id"
    else
        echo "Error occurred while running: $scene on GPU: $gpu_id"
    fi
}

check_new_json() {
    echo "Running compute script..."
    python3 "$COMPUTE_SCRIPT"
    if [ $? -eq 0 ]; then
        echo "Compute script completed successfully"
    else
        echo "Error in compute script"
        return 1
    fi

    new_scenes=($(jq -r 'to_entries | map(select(.value.feasible == false)) | .[].key' "$JSON_FILE"))

    local added_count=0
    for scene in "${new_scenes[@]}"; do
        if [[ -z "${ALL_SCENE_FILES[$scene]}" ]]; then
            echo "Added $scene!"
            SCENE_FILES+=("$scene")
            ALL_SCENE_FILES["$scene"]=1
            ((added_count++))
        fi
    done

    echo "Added $added_count new scenes from $JSON_FILE"
}

# 主处理循环
gpu_counter=0
job_count=0

# 原来的主循环修改如下
known_json_files=()
while true; do
    # 尝试每分钟检查一次新 JSON
    if (( $(date +%s) % 60 == 0 )); then
        check_new_json
    fi

    if [ ${#SCENE_FILES[@]} -gt 0 ]; then
        scene="${SCENE_FILES[0]}"
        SCENE_FILES=("${SCENE_FILES[@]:1}")

        gpu_id=${AVAILABLE_GPUS[$((gpu_counter % ${#AVAILABLE_GPUS[@]}))]}
        ((gpu_counter++))

        run_task "$scene" "$gpu_id" &
        ((COUNT_TASK++))
        echo "Running $COUNT_TASK"

        ((job_count++))
        if [ $job_count -ge $MAX_JOBS ]; then
            wait -n
            ((job_count--))
        fi
    else
        sleep 1
    fi
done

# 保存主处理循环的PID
MAIN_PID=$!

# 等待中断信号
trap 'kill $CHECK_PID $MAIN_PID; echo "Program terminated"; exit 0' SIGINT SIGTERM

# 等待所有进程完成
wait $MAIN_PID

echo "All scenes have been processed!"