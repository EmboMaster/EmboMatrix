#!/bin/bash

CONFIG_FILE="config/config0919.json"

# -------------------------
# 函数：统计文件数量
# 参数: 要处理的索引列表（空则处理全部）
# -------------------------
count_files() {
    local indices=("$@")
    echo "==== 开始统计文件数量 ===="

    SAVE_PATHS=($(jq -r '.[].save_path' "$CONFIG_FILE"))
    local paths_to_process=()

    if [ ${#indices[@]} -eq 0 ]; then
        paths_to_process=("${SAVE_PATHS[@]}")
    else
        for i in "${indices[@]}"; do
            paths_to_process+=("${SAVE_PATHS[i]}")
        done
    fi

    for path in "${paths_to_process[@]}"; do
        if [ -d "$path" ]; then
            count=$(find "$path" -type f | wc -l)
            echo "$path: $count 文件"
        else
            echo "$path: 目录不存在"
        fi
    done
    echo "==== 文件数量统计完成 ===="
}

# -------------------------
# 函数：合并文件夹
# 参数: 要处理的索引列表（空则处理全部）
# -------------------------
merge_folders() {
    local indices=("$@")
    echo "==== 开始合并文件夹 ===="

    SAVE_PATHS=($(jq -r '.[].save_path' "$CONFIG_FILE"))
    local paths_to_process=()

    if [ ${#indices[@]} -eq 0 ]; then
        paths_to_process=("${SAVE_PATHS[@]}")
    else
        for i in "${indices[@]}"; do
            paths_to_process+=("${SAVE_PATHS[i]}")
        done
    fi

    # 用第一个路径生成目标目录名
    first_path="${paths_to_process[0]}"
    base_name=$(basename "$first_path")
    new_name=$(echo "$base_name" | sed -E 's/[-_][0-9]+$//')
    parent_dir=$(dirname "$first_path")
    target_dir="$parent_dir/$new_name"

    mkdir -p "$target_dir"
    echo "合并到目录：$target_dir"

    for path in "${paths_to_process[@]}"; do
        if [ -d "$path" ]; then
            echo "复制 $path ..."
            cp -r "$path/"* "$target_dir/"
        else
            echo "目录不存在：$path"
        fi
    done
    echo "==== 文件夹合并完成 ===="
}

# -------------------------
# 主函数
# 参数: run_mode [count|merge|all] indices...
# -------------------------
main() {
    local run_mode=${1:-all}
    shift
    local indices=("$@")  # 索引列表，从 0 开始

    case "$run_mode" in
        count)
            count_files "${indices[@]}"
            ;;
        merge)
            merge_folders "${indices[@]}"
            ;;
        all)
            count_files "${indices[@]}"
            merge_folders "${indices[@]}"
            ;;
        *)
            echo "未知模式: $run_mode"
            echo "用法: $0 [count|merge|all] [indices...]"
            exit 1
            ;;
    esac
}

# -------------------------
# 调用主函数
# 默认先处理第 0 个，再第 1 个
# -------------------------
if [ $# -eq 0 ]; then
    main all 0 1 2
else
    main "$@"
fi