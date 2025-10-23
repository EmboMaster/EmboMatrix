import os
import json

def search_and_process_txt_files(log_dir, output_file):
    # Initialize list to store result paths
    result_paths = []
    
    # Walk through the directory
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                
                # Check if "Success For Starting!" exists in the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "Success For Starting!" not in content:
                        # Extract scene and task from file path
                        # Assuming path format: .../scene/task/...
                        parts = root.split(os.sep)
                        if len(parts) >= 2:
                            scene = parts[-2]  # Second to last part is scene
                            task = parts[-1]   # Last part is task
                            # Construct result path
                            result_path = f"omnigibson/shengyin/results-online/{scene}/{task}.json"
                            result_paths.append(result_path)
    
    # Write results to output JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_paths, f, indent=4)

if __name__ == "__main__":
    log_directory = "omnigibson/plannerdemo/logs/step2"
    output_json = "omnigibson/feasible_scene/middle_task.json"
    search_and_process_txt_files(log_directory, output_json)