import os
import argparse
import re
import json
from pathlib import Path
import logging
from openai import OpenAI
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from omnigibson.shengyin.scene_level.without_distributeagent import DistributeAgent
import base64
from PIL import Image
import io
from statistics import mean
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key='sk-AjgPUQzxcuKCscN3R0IPEru7G4hsAku16srLfzinmmn2AZKE', base_url="https://ai.sorasora.top/v1")

# Define output JSON path


def encode_image(image_path):
    """Encode image to base64 string."""
    try:
        with Image.open(image_path) as img:
            # Convert to PNG if not already
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None

def extract_room_name(image_name):
    """Extract room name from image file name (e.g., empty_room_0_topview.png -> empty_room_0)."""
    match = re.match(r"(.+?)_topview\.png", image_name)
    if match:
        return match.group(1)
    return None

def get_gpt4_score(image_path, new_added_tree):
    """Send image and new_added_tree to GPT-4 for scoring."""
    try:
        # Encode image
        base64_image = encode_image(image_path)
        if not base64_image:
            return None

        # Prepare prompt
        prompt = (
            f"""You are a professional scene arrangement evaluator, capable of providing objective assessments of the reasonableness and aesthetics of each scene. Now, to complete a task, the user needs to place some new objects in an initial room, and these objects must satisfy certain spatial relationships, such as "A inside B" meaning B must be placed inside A, and so on. In the context of this task, please act as an evaluator to assess how well the user has arranged these new objects.

            We will provide you with an image, which is a top-down view of a room. The image will label the names of some objects, which are either newly added objects or initial objects that have spatial relationships with the newly added ones. Unlabeled objects are part of the room's original arrangement. Additionally, we will provide a JSON description of the new added objects that must be placed in this room, along with their spatial relationships that must be satisfied.

            Here are some rules you must follow:

            
Step 1: Start with a full score of 10.
then:

🔹 1. Check Label Correspondence (Deduct 0–2 points)
- Verify whether the bounding boxes in the image match the objects specified in the JSON file.
- If there are mismatches, omissions, or incorrect names, deduct points accordingly:
  - Minor mismatches (1–2 objects incorrect): Deduct 1 point
  - Major mismatches (multiple objects incorrect or serious relational errors): Deduct 2 points

🔹 2. Assess Room Clutter (Deduct 0–4 points)
- Observe whether the room looks cluttered, whether objects are overlapping, crowded, or arranged chaotically.
- Deduct points as follows:
  - Generally tidy, only slightly crowded: Deduct 1 point
  - Noticeable crowding or some overlap: Deduct 2 points
  - Multiple overlaps or moderate chaos but still recognizable: Deduct 3 points
  - Extremely cluttered or unrecognizable: Deduct 4 points

🔹 3. Evaluate Aesthetics and Placement Reasonableness (Deduct 0–4 points)
- Consider whether objects are oriented naturally, placed reasonably, and visually harmonious.
- Deduct points as follows:
  - Mostly reasonable with minor visual inconsistencies: Deduct 1 point
  - Some objects have unnatural orientation or awkward positions: Deduct 2 points
  - Several unreasonable placements or orientations: Deduct 3 points
  - Most objects poorly placed or visually chaotic: Deduct 4 points

            **Object Placements and Relationships (new_added_tree):**\n
            {json.dumps(new_added_tree, indent=2)}\n\n
            Please provide the score and a short explanation."""
        )

        # Call GPT-4 with image and prompt
        response = client.chat.completions.create(
            model="gpt-4o",  # Use gpt-4o or similar model that supports vision
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=300
        )

        # Extract score and explanation
        response_text = response.choices[0].message.content
        score_match = re.search(r"(\d+)/10", response_text)
        score = int(score_match.group(1)) if score_match else None
        return {"score": score, "explanation": response_text}

    except Exception as e:
        logger.error(f"Failed to get GPT-4 score for {image_path}: {e}")
        return None

def process_folder(input_path, official_api=True):
    """Process a single folder and return scores for its images."""
    input_path = Path(input_path)
    if not input_path.exists():
        logger.error(f"Input path {input_path} does not exist")
        return None

    # Extract scene and task names
    scene_name = input_path.parts[-2]  # e.g., Beechwood_0_garden
    task_name = input_path.parts[-1]   # e.g., empty_room_ham_table

    # Find all PNG files
    png_files = list(input_path.glob("*.png"))
    if not png_files:
        logger.warning(f"No PNG files found in {input_path}")
        return None

    # Get BDDL file path
    bddl_file = input_path.with_suffix('.bddl')
    if not bddl_file.exists():
        logger.error(f"BDDL file {bddl_file} does not exist")
        return None
    bddl_directory = str(bddl_file)

    # Initialize DistributeAgent
    try:
        agent = DistributeAgent(bddl_directory, scene_name, official_api)
        result, inst_to_name, agent_tuples = agent.DistributeObj()
    except Exception as e:
        logger.error(f"Failed to run DistributeAgent for {input_path}: {e}")
        return None

    # Process each PNG file
    folder_results = []
    for png_file in png_files:
        room_name = extract_room_name(png_file.name)
        if not room_name:
            logger.warning(f"Could not extract room name from {png_file.name}")
            continue

        # Find matching room in result
        for room in result['rooms']:
            if result['rooms'][room]["room_name"] == room_name:
                new_added_tree = result['rooms'][room]["new_added_tree"]
                logger.info(f"Processing {png_file.name} for room {room_name}")

                # Get GPT-4 score
                update_added_tree = {}
                for key in new_added_tree.keys():
                    if "floors_" in key and new_added_tree[key] == {}:
                        continue
                    else:
                        update_added_tree[key] = new_added_tree[key]

                score_result = get_gpt4_score(png_file, update_added_tree)
                if score_result and score_result['score'] is not None:
                    folder_results.append({
                        "image": str(png_file),
                        "room_name": room_name,
                        "score": score_result['score'],
                        "explanation": score_result['explanation']
                    })
                    print({
                        "image": str(png_file),
                        "room_name": room_name,
                        "score": score_result['score'],
                        "explanation": score_result['explanation']
                    })
                else:
                    logger.warning(f"Failed to get score for {png_file.name}")
                break
        else:
            logger.warning(f"No matching room found for {room_name} in DistributeAgent results")

    return {"folder": str(input_path), "results": folder_results}

def main(base_path, OUTPUT_JSON, official_api=True):
    """Main function to process all subfolders and compute average score."""
    base_path = Path(base_path)
    if not base_path.exists():
        logger.error(f"Base path {base_path} does not exist")
        return

    # Collect results from all subfolders
    all_results = []
    scores = []
    
    for subfolder in tqdm(base_path.iterdir()):
        if subfolder.is_dir():
            for taskfolder in tqdm(subfolder.iterdir()):
                if taskfolder.is_dir():
                    folder_result = process_folder(taskfolder, official_api)
                    if folder_result and folder_result['results']:
                        all_results.append(folder_result)
                        for result in folder_result['results']:
                            scores.append(result['score'])

        # Save results to JSON
        try:
            os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(all_results, f, indent=4)
            logger.info(f"Saved results to {OUTPUT_JSON}")
        except Exception as e:
            logger.error(f"Failed to write {OUTPUT_JSON}: {e}")

    # Calculate and print average score
    if scores:
        average_score = mean(scores)
        print(f"Average score across all rooms: {average_score:.2f}/10")
    else:
        print("No valid scores were obtained.")



if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Run the script with a custom results folder.")
    # 添加命令行参数，用于指定最后一个文件夹
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        help="Name of the results folder (e.g., results0506-our)"
    )
    # 解析参数
    args = parser.parse_args()
    # results_folder = "results0506-" + args.mode

    # OUTPUT_JSON = f"omnigibson/shengyin/results0506-{args.mode}/Beechwood_0_garden_scores_dyz_addpenalty.json"
    results_folder = "results-ablation"
    OUTPUT_JSON = f"omnigibson/shengyin/results-ablation/Beechwood_0_garden_scores_dyz.json" 

    # 构造 base_path
    base_dir = "omnigibson/shengyin"
    base_path = os.path.join(base_dir, results_folder)

    main(base_path, OUTPUT_JSON, official_api=True)