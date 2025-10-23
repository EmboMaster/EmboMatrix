import os
import re
import json
from datetime import datetime

def calculate_time_diff(log_dir):
    # Dictionary to store filename and time difference
    result_dict = {}
    
    # Regex patterns
    task_path_pattern = re.compile(r'\[(.*?)\] The task path is (.*?)\.')
    post_process_pattern = re.compile(r'\[(.*?)\] Post process done!')
    
    # Walk through all txt files in the directory
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                
                # Read the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                task_time = None
                task_filename = None
                post_time = None
                
                # Search for patterns in the file
                for line in lines:
                    # Match task path line
                    task_match = task_path_pattern.search(line)
                    if task_match and not task_time:  # Only take first occurrence
                        task_time = datetime.strptime(task_match.group(1), '%Y-%m-%d %H:%M:%S')
                        task_filename = task_match.group(2)
                    
                    # Match post process line
                    post_match = post_process_pattern.search(line)
                    if post_match:
                        post_time = datetime.strptime(post_match.group(1), '%Y-%m-%d %H:%M:%S')
                        break  # Only need first occurrence
                
                # Calculate time difference if both times are found
                if task_time and post_time and task_filename:
                    time_diff = (post_time - task_time).total_seconds()
                    result_dict[task_filename] = time_diff
    
    return result_dict

# Directory path
log_dir = '/GPFS/rhome/yuanzhuoding/planner-bench/planner_bench/omnigibson/shengyin/results0506-layoutgpt/logs'
output_file = '/GPFS/rhome/yuanzhuoding/planner-bench/planner_bench/omnigibson/shengyin/results0506-layoutgpt/time.json'

# Get the dictionary of filename and time difference
time_diff_dict = calculate_time_diff(log_dir)

# Calculate average time difference
if time_diff_dict:
    avg_time_diff = sum(time_diff_dict.values()) / len(time_diff_dict)
else:
    avg_time_diff = 0

# Prepare output dictionary
output_dict = {
    "file_time_differences": time_diff_dict,
    "average_time_difference": avg_time_diff
}

# Save to JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_dict, f, indent=4)

# Print results
print(f"Average Time Difference: {avg_time_diff} seconds")
for filename, time_diff in time_diff_dict.items():
    print(f"Filename: {filename}, Time Difference: {time_diff} seconds")