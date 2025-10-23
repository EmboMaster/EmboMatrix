import json
from openai import OpenAI
import sys
import re
import torch as th
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.constants import PrimType
from src.llm_selection import get_gpt_response_by_request
from src.utils.config_loader import config
relation_words = ['attachedto', 'covered', 'draped', 'filled', 'inside', 'ontop', 'overlaid', 'saturated', 'under', 'unchanged']
relation_words_large = ['AttachedTo', 'Covered', 'Draped', 'Filled', 'Inside', 'OnTop', 'Overlaid', 'Saturated', 'Under', 'Unchanged']

def layoutgpt_main(init_keys, unchanged_old_objs, new_added_tree, new_added_objects, room_name, room_img_list, official_api=False, additional_conditions=[], error_tuple=[], env=None):
    changed_old_objs = [item for item in init_keys if item not in unchanged_old_objs]

    moved_objs = []
    moved_old_objects = moved_objs

    init_keys, init_keys_instances = [], []

    for obj in env.scene.objects:
        if obj is not None:
            if "in_rooms" in dir(obj) and room_name in obj.in_rooms and obj.name not in new_added_objects:
                init_keys.append(obj.name)
                init_keys_instances.append(obj)

    for key in new_added_tree.keys():
        # 如果物体场景本来就有，那么就不用放置
        if key not in init_keys:
            if key not in moved_objs and "floors" not in key:
                moved_objs.append(key)

    if len(moved_objs) == 0:
        print("There is no new added objects in the room, please add some new objects")
        return {"inroom": new_added_tree}, {}

    # Get extents for moved objects
    move_objs_instances = [env.scene.object_registry("name", obj) for obj in moved_objs]
    move_objs_extents = [get_relative_aabb_information(obj)[2].tolist() for obj in move_objs_instances]

    # Get position and orientation for initial objects
    init_keys_positions = []
    init_keys_orientations = []
    for obj in init_keys_instances:
        pos, orn = obj.get_position_orientation()
        init_keys_positions.append(pos.tolist())  # Convert numpy array to list
        init_keys_orientations.append(orn.tolist())  # Convert numpy array to list

    # Prepare prompt for GPT to generate positions and orientations
    prompt = f"""
    You are a helpful assistant as a home decorator, tasked with generating precise positions and orientations for new objects to be placed in a room. The inputs include:
    (1) A list of existing objects in the room, along with their positions (x, y, z) and orientations (quaternion: w, x, y, z).
    (2) A list of new objects to be placed, along with their extents (width, depth, height).

    You must generate a reasonable and realistic placement for each new object, considering:
    - Common-sense placement (e.g., a notebook is usually placed on a table, not floating in mid-air).
    - The extents of the new objects to ensure they fit in the space without collisions.
    - The positions and orientations of existing objects to avoid overlaps and ensure harmony.

    The output should be a JSON dictionary where each key is a new object name, and the value is a dictionary containing:
    - "position": [x, y, z] (coordinates in meters).
    - "orientation": [w, x, y, z] (quaternion for rotation).

    The inputs are as follows:
    The room is a: {room_name},
    The existing objects: {init_keys},
    The existing objects' positions: {init_keys_positions},
    The existing objects' orientations: {init_keys_orientations},
    The new objects: {moved_objs},
    The new objects' extents (width, depth, height): {move_objs_extents},

    The output format should be:
    <result>
    {{
        "[obj1]": {{"position": [x1, y1, z1], "orientation": [w1, x1, y1, z1]}},
        "[obj2]": {{"position": [x2, y2, z2], "orientation": [w2, x2, y2, z2]}},
        ...
    }}
    </result>

    Ensure all object names are correct and included in the inputs. The response must be wrapped in <result></result> tags.
    """

    tree_flag = True
    while tree_flag:
        response = generate_new_json(prompt, image_paths=room_img_list, official_api=official_api)
        print("Position and orientation response:", response)
        pattern = re.compile(r'<result>(.*?)</result>', re.DOTALL)
        match = pattern.search(response)
        if match:
            try:
                json_content = match.group(1).strip()
                extracted_dict = json.loads(json_content)
            except json.JSONDecodeError as e:
                print("JSON decode error:", e)
                continue
        else:
            try:
                extracted_dict = json.loads(response)
            except json.JSONDecodeError as e:
                print("JSON decode error:", e)
                continue

        # Validate the output
        try:
            valid = True
            for obj in moved_objs:
                if obj not in extracted_dict:
                    print(f"Missing object {obj} in generated output")
                    valid = False
                    break
                if not isinstance(extracted_dict[obj], dict):
                    print(f"Invalid format for {obj}: expected dict")
                    valid = False
                    break
                if "position" not in extracted_dict[obj] or "orientation" not in extracted_dict[obj]:
                    print(f"Missing position or orientation for {obj}")
                    valid = False
                    break
                if len(extracted_dict[obj]["position"]) != 3 or len(extracted_dict[obj]["orientation"]) != 4:
                    print(f"Invalid position or orientation length for {obj}")
                    valid = False
                    break
            if valid:
                tree_flag = False
            else:
                continue
        except Exception as e:
            print("Generated JSON format error:", e)
            continue

    # Combine the new added tree with the generated positions
    new_added_triples = tree_to_triples(new_added_tree)
    combined_tree = triples_to_tree(new_added_triples, moved_objs)

    return {"inroom": combined_tree}, extracted_dict

def tree_to_triples(tree):
    """
    Convert a nested tree structure into a list of triples.

    Args:
        tree (dict): The nested tree structure.

    Returns:
        list: A list of triples (obj1, relation, obj2).
    """
    def traverse(subtree, triples):
        for obj1, relations in subtree.items():
            for relation, sub_relations in relations.items():
                for obj2, nested in sub_relations.items():
                    triples.append((obj1, relation, obj2))
                    # Recursively process deeper levels
                    traverse({obj2: nested}, triples)

    triples = []
    traverse(tree, triples)
    return triples

def get_relative_aabb_information(objA):
    if isinstance(objA, DatasetObject) and objA.prim_type == PrimType.RIGID:
        parallel_bbox_center, parallel_bbox_orn, parallel_bbox_extents, _ = objA.get_base_aligned_bbox(
            xy_aligned=True
        )
    else:
        aabb_lower, aabb_upper = objA.states[AABB].get_value()
        parallel_bbox_center = (aabb_lower + aabb_upper) / 2.0
        parallel_bbox_orn = th.tensor([0.0, 0.0, 0.0, 1.0])
        parallel_bbox_extents = aabb_upper - aabb_lower
    
    return parallel_bbox_center.numpy(), parallel_bbox_orn, parallel_bbox_extents.numpy()

def find_valid_placement(move_objs_extents, move_objs_name, init_keys_extents, init_keys_positions, init_keys):
    result_dict = {}
    for obj_name, (obj_w, obj_d, obj_h) in zip(move_objs_name, move_objs_extents):
        valid_platforms = []
        for i, ((plat_w, plat_d, plat_h), (plat_x, plat_y, plat_z), obj) in enumerate(zip(init_keys_extents, init_keys_positions, init_keys)):
            if plat_w >= obj_w and plat_d >= obj_d:
                if (plat_z + plat_h/2 + obj_h/2) <= 1.5:
                    valid_platforms.append(obj)
        result_dict[obj_name] = valid_platforms
    return result_dict

def triples_to_tree(triples, objects):
    new_triples = []
    for triple in triples:
        new_triples.append((triple[0], triple[1].replace('ontop', "OnTop").replace('under', "Under").replace('inside', "Inside"), triple[2]))
    triples = new_triples

    tree = {}
    obj1_list = [triple[0] for triple in triples]
    obj2_list = [triple[2] for triple in triples]
    need_to_add = set()
    for obj in obj1_list:
        if obj not in obj2_list:
            tree[obj] = {}
            need_to_add.add(obj)

    def add_to_tree(root, current_tree, triples, need_to_add):
        def find_subtree(tree, target):
            if target in tree:
                return tree[target]
            for key, value in tree.items():
                if isinstance(value, dict):
                    result = find_subtree(value, target)
                    if result is not None:
                        return result
            return None

        subtree = find_subtree(current_tree, root)
        if subtree is None:
            raise ValueError(f"Root '{root}' not found in the current tree.")

        for obj1, relation, obj2 in triples:
            if obj1 == root:
                if relation not in subtree:
                    subtree[relation] = {}
                if obj2 not in subtree[relation]:
                    subtree[relation][obj2] = {}
                    need_to_add.add(obj2)

        need_to_add.remove(root)

    while len(list(need_to_add)) != 0:
        for root in list(need_to_add):
            add_to_tree(root, tree, triples, need_to_add)

    for obj in objects:
        flag = False
        for triple in triples:
            if obj == triple[0] or obj == triple[2]:
                flag = True
                break
        if not flag:
            tree[obj] = {}

    return tree

def generate_new_json(prompt, image_paths, official_api):
    from openai import OpenAI
    import base64
    use_official_api = official_api
    model = config['scene_generation']['model']
    api_key = config['scene_generation']['api_key']
    base_url = config['scene_generation']['base_url']

    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": prompt
        }]
    },
    {
        "role": "user",
        "content": "please generate the positions and orientations of the new objects within <result></result> tags"
    }]

    response = get_gpt_response_by_request(model=model, api_key=api_key, base_url=base_url, messages=messages, image_paths=image_paths, max_try=3, use_official_api=use_official_api, if_json_output=False)
    return response

if __name__ == "__main__":
    init_keys = ["floors_yrwwan_0", "trash_can_vasiit_86", "ottoman_ahgkci_0", "sofa_ahgkci_1", "shelf_shgad_2"]
    unchanged_old_objs = ["floors_yrwwan_0", "trash_can_vasiit_86", "ottoman_ahgkci_0", "sofa_ahgkci_1", "shelf_shgad_2"]
    new_added_tree = {
        "sofa_ahgkci_2": {
            "ontop": {
                "notebook_aanuhi_1": {},
                "notebook_aanuhi_2": {},
                "notebook_aanuhi_3": {},
            }
        },
        "floors_yrwwan_0": {"ontop": {"notebook_aanuhi_4": {}}}
    }
    new_added_objects = ["trash_can_vasiit_87", "sticky_note_tghqep_88", "notepad_njeocp_89"]
    room_name = "kitchen_0"
    room_img_list = ["omnigibson/examples/scenes/out_with_bbox_objectname/Beechwood_0_garden_Full/kitchen_0_7_bbox_2d_tight.jpg"]

    print(step2_main(init_keys, unchanged_old_objs, new_added_tree, new_added_objects, room_name, room_img_list, official_api=True))