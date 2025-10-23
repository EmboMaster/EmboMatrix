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

def step2_main(init_keys, unchanged_old_objs, new_added_tree, new_added_objects, room_name, room_img_list, official_api=False,additional_conditions=[], error_tuple=[],env=None):

    changed_old_objs = [item for item in init_keys if item not in unchanged_old_objs]

    moved_objs = []
    moved_old_objects = moved_objs

    init_keys,init_keys_instances = [],[]

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
        return {"inroom": new_added_tree}, []
    
    move_objs_instances = [env.scene.object_registry("name", obj) for obj in moved_objs]
    move_objs_extents = [get_relative_aabb_information(obj)[2] for obj in move_objs_instances]
    if init_keys_instances == []:
        init_keys_extents, init_keys_positions = [], []
    else:
        init_keys_extents = [get_relative_aabb_information(obj)[2] for obj in init_keys_instances]
        init_keys_positions = [get_relative_aabb_information(obj)[0] for obj in init_keys_instances]
    ontop_dict = find_valid_placement(move_objs_extents, moved_objs, init_keys_extents, init_keys_positions,init_keys)

    floor_obj = [obj for obj in init_keys if 'floors' in obj][0]
    for key in ontop_dict:
        if floor_obj not in ontop_dict[key]:
            ontop_dict[key].append(floor_obj)
    print("OnTop Dict:", ontop_dict)

    relation_rules = """

    1. Inside:
    Defines whether this object is considered inside another object. This does raycasting in all axes (x, y, z), and checks to make sure that rays shot in at least two of these axes hit the other object.

    2. OnTop:
    Defines whether this object is considered on top of another object. This checks to make sure that this object is touching the other and that the other is in this object"s VerticalAdjacency negative_neighbors list.

    3. Under:
    Defines whether this object is considered under another object. This checks to make sure that this object is touching the other and that the other is in this object"s VerticalAdjacency positive_neighbors list.
    """


    prompt = f"""
    You are a helpful assistant as a home decorator and are now being asked to generate the relation between the new added objects and the existed objects in the room. 
    The inputs include (1) a list of existed objects in the room, (2) a list of new added objects, (3) a list of generation suggestions that must be followed, (4) a list of past errors that must be avoided, (5) a dict of containing all the objects on which each new added object can be placed. (3) and (4) are sometimes not given.
    
    There are only three types of relations and the detailed explanation is as follows:
    {relation_rules}

    There are rules you should obey:
    1. You should consider the common-sense relationships between objects, resulting in a layout that is reasonable and correct. For example, a notebook is usually placed on a table, not on the floor.
    2. If you want to put a new added object OnTop another object in the room, you must choose the object in the given dict. We compute the height and the size of the new added object and other objects to make sure that the put operation is normal. Do not put big objects like shelf or tall objects like lamp on other high objects like countertop or other cabinet! These objects should be put on the floor!
    3. Please refer to the provided image and choose the most appropriate placement relationships. For example, if an object has a small cross-sectional area, it may not be suitable for placing large objects on top of it. 
	4. If generation suggestions are provided, the generation must satisfy these suggestions.
	5. If error information is given, the final outputs must avoid these errors.
	6. Ensure that the names of objects in the outputs are correct and they must be included in the inputs.
    7. For each new added object, you should only output one relation between a certain existed object and the new added object. The relation will guide the simulator to place the new added object in the right position.
    8. The relation will be outputed as a tuple, which contains three elements: the existed object, the relation, and the new added object.

    The inputs are as follows:
    The room is a: {room_name},
    The existed objects: {init_keys},
    The new added objects: {moved_objs}
    The generation suggestions are: {additional_conditions}
    The error information is: {error_tuple}
    The dict containing all available objects on which each new added object can be placed: {ontop_dict}

    The output format should comply with the rules as follows:
    <result>
    {{
    "tuples":[
    ([objA],[relation1],[obj1]),
    ([objB],[relation2],[obj2]),
    ...
    ]
    }}
    </result>

    The content inside [] is what you need to generate. objA and objB are existed objects. obj1 and obj2 are new added objects. You must contain <result></result> tags in your response.

    """

    tree_flag = True
    while tree_flag:
        # response = generate_new_json(prompt, [])
        # generated_json_str = response
        response = generate_new_json(prompt, image_paths = room_img_list,official_api=official_api)
        print("multi-level tree response:", response)
        pattern = re.compile(r'<result>(.*?)</result>', re.DOTALL)
        match = pattern.search(response)
        #import pdb;pdb.set_trace()
        if match:
            try:
                json_content = match.group(1).strip()
                json_content = json_content.replace("(", "[").replace(")", "]")
                json_content = json_content.replace("'", "\"")  # 替换单引号为双引号
                extracted_dict = json.loads(json_content)
            except json.JSONDecodeError as e:
                print("JSON decode error:", e)
                continue
        else:
            try:
                json_content = response.replace("'", "\"")  # 替换单引号为双引号
                extracted_dict = json.loads(response) 
            except json.JSONDecodeError as e:
                continue

        try:
            extracted_dict = extracted_dict["tuples"]
            for i in range(len(extracted_dict)):
                if not isinstance(extracted_dict[i], tuple):
                    extracted_dict[i] = tuple(extracted_dict[i])
            triples = extracted_dict

            for triple in triples:
                triple_flag = True
                if triple[1] in relation_words_large or triple[1] in relation_words:
                    if triple[0] not in relation_words_large and triple[0] not in relation_words:
                        if triple[2] not in relation_words_large and triple[2] not in relation_words:
                            triple_flag = False
                if triple_flag:
                    print(f"wrong tuples for generated tree: {triple}")
                    error_tuple.append(f"there is an error for {triple}")  
                    continue
                if triple[1] == "OnTop" or triple[1] == "ontop":
                    if triple[0] not in ontop_dict[triple[2]]:
                        print(f"wrong tuples for generated tree: {triple}")
                        error_tuple.append(f"there is an error for {triple}! You must choose the object from {ontop_dict[triple[2]]} if you want to put {triple[2]} ontop of an object.")  
                        continue
            
            # import pdb; pdb.set_trace()
            new_added_triples = tree_to_triples(new_added_tree) 
            combined_triples = triples + new_added_triples
            extracted_dict = triples_to_tree(combined_triples, moved_objs)
            if extracted_dict == {}:
                print("The generated tree is empty, please regenerate the tree")
                continue
            tree_flag = False
        except Exception as e:
            print("生成的 JSON 格式错误:", e)
            continue
    #print("extracted_dict:", extracted_dict)
    return {"inroom": extracted_dict}, triples


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
        # Retrieve base CoM frame-aligned bounding box parallel to the XY plane
        parallel_bbox_center, parallel_bbox_orn, parallel_bbox_extents, _ = objA.get_base_aligned_bbox(
            xy_aligned=True
        )
        #print(parallel_bbox_center, parallel_bbox_orn, parallel_bbox_extents)
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
            # 1. 平台的横截面 (w, d) 是否足够大
            if plat_w >= obj_w and plat_d >= obj_d:
                # 2. 确保放置后总高度不超过 1.5m
                if (plat_z + plat_h/2 + obj_h/2) <= 1.5:
                    valid_platforms.append(obj)
        
        result_dict[obj_name] = valid_platforms

    return result_dict

def triples_to_tree(triples, objects):
        """
        Convert a list of triples into a nested tree dictionary.
        Starts with root categories (e.g., floors, ceiling, wall) and iteratively adds connected objects.

        Args:
            triples (list of tuples): List of triples (obj1, relation, obj2).

        Returns:
            dict: A nested tree structure representing the triples.
        """

        # new_triples = []
        # for triple in triples:
        #     if 'floor' in triple[0]:
        #         new_triples.append(("floors", triple[1], triple[2]))
        #     else:
        #         new_triples.append(triple)
        # triples = new_triples

        new_triples = []
        for triple in triples:
            new_triples.append((triple[0], triple[1].replace('ontop',"OnTop").replace('under',"Under").replace('inside',"Inside"), triple[2]))
        triples = new_triples

        tree = {}

        obj1_list = [triple[0] for triple in triples]
        obj2_list = [triple[2] for triple in triples]
        need_to_add = set()
        for obj in obj1_list:
            if obj not in obj2_list:
                tree[obj] = {}
                need_to_add.add(obj)
        # for obj in obj1_list:
        #     if ('floors' in obj or 'ceilings' in obj or 'wall' in obj) and obj not in tree.keys():
        #         tree[obj] = {}
        #         need_to_add.add(obj)

        def add_to_tree(root, current_tree, triples, need_to_add):
            """
            Add triples to the tree under the given root object.

            Args:
                root (str): The root object to which the relationships should be added.
                current_tree (dict): The current tree structure.
                triples (list of tuples): List of triples (obj1, relation, obj2).

            Returns:
                None: The current_tree is modified in place.
            """
            def find_subtree(tree, target):
                """Recursively find the subtree where the target root exists."""
                if target in tree:
                    return tree[target]
                for key, value in tree.items():
                    if isinstance(value, dict):
                        result = find_subtree(value, target)
                        if result is not None:
                            return result
                return None

            # Locate the subtree where the root exists
            subtree = find_subtree(current_tree, root)

            if subtree is None:
                raise ValueError(f"Root '{root}' not found in the current tree.")

            # Add triples to the located subtree
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
def generate_new_json(prompt, image_paths,official_api):
    """
    生成新的 JSON,其中可以包含多个图片
    Args:
        prompt (str): 提示文本。
        image_paths (list): 包含图片路径的列表。
    Returns:
        dict: GPT-4 API 的响应。
    """
    from openai import OpenAI
    import base64

    use_official_api = official_api
    model = config['scene_generation']['model']
    api_key = config['scene_generation']['api_key']
    base_url = config['scene_generation']['base_url']


    # 准备消息内容
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": prompt
        }]
    },
    {
        "role": "user",
        "content": "please generate the relation triples of the new room within <result></result> tags"
    }]

    response = get_gpt_response_by_request(model=model, api_key=api_key,base_url = base_url, messages=messages, image_paths = image_paths, max_try=3, use_official_api=use_official_api, if_json_output=False)
    return response


if __name__ == "__main__":
    # inputs
    init_keys = ["floors_yrwwan_0", "trash_can_vasiit_86","ottoman_ahgkci_0", "sofa_ahgkci_1","shelf_shgad_2"]
    unchanged_old_objs = ["floors_yrwwan_0", "trash_can_vasiit_86","ottoman_ahgkci_0", "sofa_ahgkci_1","shelf_shgad_2"]
    new_added_tree = {
                
                "sofa_ahgkci_2": {
                    "ontop": {
                        "notebook_aanuhi_1": {},
                        "notebook_aanuhi_2": {},
                        "notebook_aanuhi_3": {},
                    }
                },
                "floors_yrwwan_0": {"ontop": { "notebook_aanuhi_4": {}} }

    }

    new_added_objects = ["trash_can_vasiit_87", "sticky_note_tghqep_88", "notepad_njeocp_89"]
    room_name = "kitchen_0"
    room_img_list = ["omnigibson/examples/scenes/out_with_bbox_objectname/Beechwood_0_garden_Full/kitchen_0_7_bbox_2d_tight.jpg"]

    print(step2_main(init_keys, unchanged_old_objs, new_added_tree, new_added_objects, room_name, room_img_list, official_api=True))