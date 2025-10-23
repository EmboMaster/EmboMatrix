import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.object_states import AttachedTo, Covered, Draped, Filled, Inside, OnTop, Overlaid, Saturated, Under
from src.llm_selection import convert_images_to_base64
from omnigibson.utils.camera_utils import camera_for_scene_room 

from PIL import Image
import torch as th
import random

# # Configure macros for maximum performance
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = False
# gm.ENABLE_OBJECT_STATES = False
# gm.ENABLE_TRANSITION_RULES = False

import json, os

def step3_main(scene_model, room_name, excluded_obj_list, new_added_object_list, tree_data, image_height, image_width, if_reset=False, reset_env=None, already_modify_objs={}, if_scene_create=False, room_already_config={}, if_continue_create=True,use_official_api=False, rule_path=None, debug_path_image=None,sample_pose_final_path=None, use_holodeck=True, use_tree=True, scene_file = None):

    init_obj_pos_rot = get_init_obj_pos_rot(scene_model, room_name)

    cfg = {
        "scene": {
            "type": 'InteractiveTraversableScene',
            "scene_model": scene_model,
            "not_load_object_categories": ["roof", "downlight", "room_ventilator", "square_light", "fire_alarm", "range_hood", "room_light", "fire_sprinkler", "signpost"],
            "load_room_instances": [room_name]
        },
    }

    # modified_obj_info = load_init_object_from_scene(scene_model=cfg["scene"]["scene_model"], excluded_obj_list=excluded_obj_list, room_name=room_name)
    modified_obj_info = {}
    
    cfg['objects'] = [
        {
            "type": "LightObject",
            "name": "Light_test",
            "light_type": "Distant",
            "intensity": 514,
            "radius": 10,
            "position": [0, 0, 30],
        }
    ]
    cfg['objects'].extend(new_added_object_list)

    # add modified obj
    for obj in modified_obj_info.keys():
        tmp = {
            "type": "DatasetObject",
            "name": obj,
            "category": modified_obj_info[obj]["args"]["category"],
            "model": modified_obj_info[obj]["args"]["model"],
            "scale": modified_obj_info[obj]["args"]["scale"],
            "in_rooms": modified_obj_info[obj]["args"]["in_rooms"],
        }
        cfg['objects'].append(tmp)

    cfg["render"] = {
        "viewer_width": image_width,
        "viewer_height": image_height,
    }

    import torch
    colide_flag, trial_times = True, 5
    while colide_flag and trial_times > 0:
    
        if not if_scene_create:

            # Load the environment
            if not if_reset:
                env = og.Environment(configs=cfg)
            else:
                env = reset_env
                env.reset()
                for tmp_obj_name in already_modify_objs.keys():
                    tmp_obj = env.scene.object_registry("name", tmp_obj_name)
                    if tmp_obj != None:
                        tmp_obj.set_position_orientation(already_modify_objs[tmp_obj_name]["pos"], already_modify_objs[tmp_obj_name]["ori"])
                        tmp_obj.set_angular_velocity(torch.tensor([0., 0., 0.]))
                        tmp_obj.set_linear_velocity(torch.tensor([0., 0., 0.]))
                if not if_continue_create:
                    return [], True, env, [], "", {}
        else:
            env = reset_env
            if if_reset:
                env.reset()
                for tmp_obj_name in already_modify_objs.keys():
                    tmp_obj = env.scene.object_registry("name", tmp_obj_name)
                    if tmp_obj != None:
                        tmp_obj.set_position_orientation(already_modify_objs[tmp_obj_name]["pos"], already_modify_objs[tmp_obj_name]["ori"])
                        tmp_obj.set_angular_velocity(torch.tensor([0., 0., 0.]))
                        tmp_obj.set_linear_velocity(torch.tensor([0., 0., 0.]))
                if not if_continue_create:
                    return [], True, env, [], "", {}
                
        for obj in room_already_config.keys():
            tmp_obj_instance = env.scene.object_registry("name", obj)
            if tmp_obj_instance is not None:
                tmp_obj_instance.set_position(room_already_config[obj]["pos"])
                tmp_obj_instance.set_orientation(room_already_config[obj]["ori"])
                # tmp_obj_instance.set_angular_velocity(torch.tensor([0., 0., 0.]))
                # tmp_obj_instance.set_linear_velocity(torch.tensor([0., 0., 0.]))

        new_objects_name_list = []
        for new_obj in new_added_object_list:
            if new_obj["in_rooms"][0] == room_name:
                new_objects_name_list.append(new_obj["name"])
                new_obj_instance = env.scene.object_registry("name", new_obj["name"])
                new_obj_instance.set_angular_velocity(torch.tensor([0., 0., 0.]))
                new_obj_instance.set_linear_velocity(torch.tensor([0., 0., 0.]))

        colide_list = velocity_collide(env, new_objects_name_list, norm_threshold=0.1, room_name=room_name)
        if colide_list == []:
            colide_flag = False
        else:
            trial_times -= 1
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(colide_list)
            if debug_path_image is not None:
                img_path = f"{debug_path_image}/{scene_model}_{room_name}_initial_collide_{timestamp}_{trial_times}.png"
            else:
                img_path = f"src/room_creater/ontop_check_images/{scene_model}_{room_name}_initial_collide_{timestamp}_{trial_times}.png"

            get_image_now(scene_model, room_name, env, img_path)

    # print(og.log)
    # import pdb; pdb.set_trace()

    excluded_obj_list = set(excluded_obj_list)
    must_exclude = ['light', 'Light', 'floor', 'wall', 'door', 'ceiling', 'window']
    for obj in env.scene.objects:
        if obj.name not in excluded_obj_list and any([keyword in obj.name for keyword in must_exclude]):
            excluded_obj_list.add(obj.name)
    excluded_obj_list = list(excluded_obj_list)

    tree_structure = tree_data.get("inroom", {})

    global add_objects_relative
    add_objects_relative = set()
                                            
    # set the room
    tree_flag = True
    holodeck_fail_dict = {}
    if use_tree:
        for obj in tree_structure.keys():

            already_obj = [tmp.name for tmp in env.scene.objects]
            
            if obj not in already_obj:
                print("The first level object needs to exist in this scene.")
                tree_flag = False
                # og.sim.stop()
                return None, tree_flag, env, add_objects_relative, "The first level object needs to exist in this scene.", holodeck_fail_dict
            
            print(f"Already install {obj}!!")
            if tree_structure[obj] != {}:
                tree_flag, error_tuple, holodeck_fail_dict = add_leaf_object(obj, tree_structure[obj], env,excluded_obj_list, init_obj_pos_rot, scene_model=scene_model, room_name=room_name, use_official_api=use_official_api, holodeck_fail_dict=holodeck_fail_dict,debug_path_image=debug_path_image, new_objects_name_list=new_objects_name_list, sample_pose_final_path=sample_pose_final_path, use_holodeck = use_holodeck, scene_file=scene_file)
            else:
                if obj not in excluded_obj_list:
                    add_objects_relative.add(obj)
                    tree_flag = True
            if tree_flag == False:
                return None, tree_flag, env, add_objects_relative, error_tuple, holodeck_fail_dict
    else:
        # Convert tree structure to triples
        triples = tree_to_triples(tree_structure)
        if not triples:
            print("No valid triples generated from tree structure.")
            return None, False, env, add_objects_relative, "No valid triples generated from tree structure.", holodeck_fail_dict

        # Get list of existing objects in the scene
        already_obj = [tmp.name for tmp in env.scene.objects]

        random.shuffle(triples)

        # Check if all parent objects exist in the scene
        for parent, relation, child in triples:
            if parent not in already_obj:
                print(f"Parent object {parent} does not exist in the scene.")
                return None, False, env, add_objects_relative, f"Parent object {parent} does not exist in the scene.", holodeck_fail_dict

        # Process each triple
        for parent, relation, child in triples:
            if child in excluded_obj_list:
                continue

            print(f"Setting relation: {child} {relation} {parent}")
            
            # Call set_relation to place the child object relative to the parent
            tree_flag, holodeck_fail_list, robot_list = set_relation(
                child, parent, relation, env, init_obj_pos_rot, 
                scene_model=scene_model, room_name=room_name, 
                use_official_api=use_official_api, debug_path_image=debug_path_image, 
                new_objects_name_list=new_objects_name_list, use_holodeck=use_holodeck
            )

            # Handle holodeck failures for OnTop relations
            if (relation.lower() in ["ontop", "on top"]) and tree_flag:
                holodeck_fail_dict[child] = holodeck_fail_list

            # Save robot poses if applicable
            if tree_flag and robot_list and sample_pose_final_path:
                with open(os.path.join(sample_pose_final_path, "sample_pose.txt"), 'a') as f:
                    f.write(f"{child}: {robot_list}\n")
                update_sample_pose_file(sample_pose_final_path, parent, robot_list)

            # Add objects to relative set
            if tree_flag:
                print(f"Already installed {child}!!")
                tmp_child_obj = env.scene.object_registry("name", child)
                tmp_parent_obj = env.scene.object_registry("name", parent)
                if tmp_child_obj:
                    add_objects_relative.add(child)
                if tmp_parent_obj:
                    add_objects_relative.add(parent)

            # If placement fails, return error
            if not tree_flag:
                error_tuple = (parent, relation, child)
                return None, False, env, add_objects_relative, error_tuple, holodeck_fail_dict
        # return [], True, env, add_objects_relative, ""
    from omnigibson.utils.camera_utils import camera_for_scene_room
    print(list(add_objects_relative))
    all_objects = [obj.name for obj in env.scene.objects]
    final_uninclude_obj_list = [obj for obj in all_objects if obj not in list(add_objects_relative)]
    top_view_img = camera_for_scene_room(scene_model, room_name, 'top_view', 'bbox_2d_tight', final_uninclude_obj_list, env, image_height=image_height, image_width=image_width)
    if top_view_img is None:
        print("Can not get the top view image.")
    else:
        top_view_img = top_view_img[0][0]
    
    front_view_img_list = camera_for_scene_room(scene_model, room_name, 'front_view', 'bbox_2d_tight', final_uninclude_obj_list, env, image_height=image_height, image_width=image_width)
    if front_view_img_list is None:
        print("Can not get the front view image.")
    else:
        front_view_img_list, front_view_obj_list = front_view_img_list
        front_view_best_idx = find_max_gt_overlap_idx(front_view_obj_list, add_objects_relative)
    
    img_path_list = []
    from PIL import Image
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if top_view_img is not None:
        image = Image.fromarray(top_view_img.astype('uint8'))
        # 将图像保存到内存中的字节流
        if rule_path == None:
            save_path = f'src/images_new/{timestamp}_{scene_model}_{room_name}_top_view.png'
        else:
            save_path = os.path.join(rule_path, f'{timestamp}_{scene_model}_{room_name}_top_view.png')
        image.save(save_path)
        img_path_list.append(save_path)
    
    if front_view_img_list is not None:
        img = front_view_img_list[front_view_best_idx]
        image = Image.fromarray(img.astype('uint8'))
        # 将图像保存到内存中的字节流
        if rule_path == None:
            save_path = f'src/images_new/{timestamp}_{scene_model}_{room_name}_front_view.png'
        else:
            save_path = os.path.join(rule_path, f'{timestamp}_{scene_model}_{room_name}_front_view.png')
        image.save(save_path)
        img_path_list.append(save_path)
    
    img_path_list = convert_images_to_base64(img_path_list, target_width=800)

    return img_path_list, True, env, add_objects_relative, ("","",""), holodeck_fail_dict

def find_max_gt_overlap_idx(front_view_obj_list, add_objects_relative):
    """
    Finds the index of the list in front_view_obj_list that contains the most ground truth (GT) object names.

    Parameters:
        front_view_obj_list (list of lists): A list containing 8 sub-lists, each sub-list containing object names.
        add_objects_relative (list): A list containing the GT object names.

    Returns:
        int: The index of the list in front_view_obj_list with the most GT object names.
    """
    max_overlap_count = 0
    max_overlap_idx = -1

    for idx, obj_list in enumerate(front_view_obj_list):
        overlap_count = len(set(obj_list) & set(add_objects_relative))
        if overlap_count > max_overlap_count:
            max_overlap_count = overlap_count
            max_overlap_idx = idx

    return max_overlap_idx

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


def get_init_obj_pos_rot(scene_model, room_name):

    json_path = f"omnigibson/data/og_dataset/scenes/{scene_model}/json/{scene_model}_best.json"
    json_origin_path = f"omnigibson/data/og_dataset/scenes/{scene_model}/json/{scene_model}_best_origin.json"

    if not os.path.exists(json_origin_path):
        with open(json_path, 'r') as file:
            data = json.load(file)
    else:
        with open(json_origin_path, 'r') as file:
            data = json.load(file)
    
    object_registry = data['state']['object_registry']
    init_info = data['objects_info']['init_info']

    room_objects = {}
    for obj_name, obj_info in init_info.items():
        try:
            if 'in_rooms' in obj_info['args'].keys():
                if obj_info['args']['in_rooms'] != '' and room_name in obj_info['args']['in_rooms'][0]:
                    room_objects[obj_name] = obj_info
        except:
            #import pdb; pdb.set_trace()
            continue
    # 提取 pos 和 ori 信息
    room_objects_pos_ori = {}
    for obj_name in room_objects:
        link_info = object_registry.get(obj_name, {}).get('root_link', {})
        if link_info:
            pos = link_info.get('pos')
            ori = link_info.get('ori')
            room_objects_pos_ori[obj_name] = {'pos': pos, 'ori': ori}

    return room_objects_pos_ori

    
# filter the object need to be moved
def load_init_object_from_scene(scene_model, excluded_obj_list, room_name):

    scene_best_json_path = f"omnigibson/data/og_dataset/scenes/{scene_model}/json/{scene_model}_best.json"
    scene_best_origin_json_path = f"omnigibson/data/og_dataset/scenes/{scene_model}/json/{scene_model}_best_origin.json"
    if not os.path.exists(scene_best_origin_json_path):
        best_path = scene_best_json_path
        with open(best_path, "r") as file:
            data = json.load(file)
        with open(scene_best_origin_json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"origin best json created")
    else:
        best_path = scene_best_origin_json_path
        with open(best_path, "r") as file:
            data = json.load(file)

    new_init_info = {}
    modified_obj_info = {}
    for obj in data['objects_info']['init_info']:
        if obj in excluded_obj_list or data['objects_info']['init_info'][obj]["args"]["in_rooms"] != [room_name]:
            new_init_info[obj] = data['objects_info']['init_info'][obj]
        else:
            modified_obj_info[obj] = data['objects_info']['init_info'][obj]
    data['objects_info']['init_info'] = new_init_info

    new_object_registry = {}
    for obj in data['state']['object_registry']:

        if obj in excluded_obj_list or obj not in modified_obj_info.keys():
            new_object_registry[obj] = data['state']['object_registry'][obj]
    
    data['state']['object_registry'] = new_object_registry

    # 定义文件路径
    old_file = scene_best_json_path
    with open(old_file, 'w') as f:
        json.dump(data, f, indent=4)
        #print(f"数据已保存到: {old_file}")
    
    return modified_obj_info

def update_sample_pose_file(sample_pose_final_path, obj, robot_list):
    """
    读取 sample_pose.txt 文件，检查是否包含 obj，若不存在则写入 obj: robot_list。
    
    参数：
        sample_pose_final_path (str): sample_pose.txt 所在的目录路径
        obj (str): 要检查的对象名称
        robot_list (list): 机器人列表，将与 obj 一起写入
    """
    # 构造文件路径
    file_path = os.path.join(sample_pose_final_path, "sample_pose.txt")
    
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 读取文件内容
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 检查是否包含 obj
        if f"{obj}:" not in content:
            # 如果不存在，追加写入
            with open(file_path, 'a') as f:
                f.write(f"{obj}: {robot_list}\n")
    else:
        # 如果文件不存在，直接创建并写入
        with open(file_path, 'w') as f:
            f.write(f"{obj}: {robot_list}\n")


def add_leaf_object(obj, obj_leaf_dict, env, excluded_obj_list, init_obj_pos_rot, scene_model=None, room_name=None, use_official_api=False, holodeck_fail_dict = {},debug_path_image=None,new_objects_name_list=None,sample_pose_final_path=None, use_holodeck = True, scene_file = None):

    tree_flag = True
    from omnigibson.objects.dataset_object import DatasetObject

    #print(f"Already install {obj}!!")
    for leaf_rel in obj_leaf_dict.keys():
        for leaf_leaf_obj in obj_leaf_dict[leaf_rel].keys():

            if leaf_leaf_obj not in excluded_obj_list:
                # set relation to obj
                if scene_file == None:
                    tree_flag, holodeck_fail_list, robot_list = set_relation(leaf_leaf_obj, obj, leaf_rel, env, init_obj_pos_rot, scene_model=scene_model, room_name=room_name,use_official_api=use_official_api,debug_path_image=debug_path_image,new_objects_name_list=new_objects_name_list, use_holodeck = use_holodeck)
                    if (leaf_rel == "OnTop" or leaf_rel == "ontop") and tree_flag == True:
                        holodeck_fail_dict[leaf_leaf_obj] = holodeck_fail_list

                    if tree_flag == True:
                        if robot_list != [] and sample_pose_final_path != None:
                            with open(os.path.join(sample_pose_final_path, "sample_pose.txt"), 'a') as f:
                                f.write(f"{leaf_leaf_obj}: {robot_list}\n")
                            update_sample_pose_file(sample_pose_final_path, obj, robot_list)

                    if tree_flag == True:
                        print(f"Already install {leaf_leaf_obj}!!")
                else:
                    tree_flag = True
                    holodeck_fail_dict = {}

                tmp_leaf_obj = env.scene.object_registry("name", leaf_leaf_obj)
                tmp_obj  = env.scene.object_registry("name", obj)
                if tmp_leaf_obj != None:
                    add_objects_relative.add(leaf_leaf_obj)
                if tmp_obj != None:
                    add_objects_relative.add(obj)

            if tree_flag == False:
                return False, (obj,leaf_rel,leaf_leaf_obj), holodeck_fail_dict

            # add leaf
            if obj_leaf_dict[leaf_rel][leaf_leaf_obj] != {}:
                tree_flag, error_tuple, holodeck_fail_dict = add_leaf_object(leaf_leaf_obj, obj_leaf_dict[leaf_rel][leaf_leaf_obj], env, excluded_obj_list, init_obj_pos_rot,scene_model=scene_model, room_name=room_name,use_official_api=use_official_api, holodeck_fail_dict = holodeck_fail_dict,debug_path_image=debug_path_image,new_objects_name_list=new_objects_name_list, sample_pose_final_path=sample_pose_final_path, use_holodeck=use_holodeck, scene_file=scene_file)

            if tree_flag == False:
                return False, error_tuple, holodeck_fail_dict
    
    return True, "", holodeck_fail_dict

def set_relation(leaf_obj, obj, rel, env, init_obj_pos_rot, scene_model=None, room_name=None,use_official_api=False,debug_path_image=None,new_objects_name_list=None, use_holodeck = True):

    tree_flag = True
    rel_list = ["AttachedTo", "Covered", "Draped", "Filled", "Inside", "OnTop", "Overlaid", "Saturated", "Under", "Unchanged"]
    lowercase_list = [item.lower() for item in rel_list]
    obj_name_all = [tmp_obj.name for tmp_obj in env.scene.objects]

    if obj in rel_list or obj in lowercase_list:
        print("Relation cannot be an object!")
        return False, [], []
    if leaf_obj in rel_list or leaf_obj in lowercase_list:
        print("Relation cannot be an object!")
        return False, [], []

    try:
        if leaf_obj not in obj_name_all:
            if '.' in leaf_obj:
                particle_type = leaf_obj.split('.')[0]
            else:
                particle_type = leaf_obj
            leaf_object = env.scene.get_system(particle_type)
            parent_object = env.scene.object_registry("name", obj)
        else:
            leaf_object, parent_object = env.scene.object_registry("name", leaf_obj), env.scene.object_registry("name", obj)
    except:
        return False, [], []
    
    if leaf_obj in obj_name_all and rel not in rel_list and rel not in lowercase_list:
        print("Wrong relation!")
        return False, [], []
    
    from src.room_creater.ontop_holodeck import OnTopHolodeck
    robot_list = []
    effect_flag = True
    # try:
    if rel == "AttachedTo" or rel == "attachedto" or rel == "attached" or rel == "Attached":
        leaf_object.states[AttachedTo].set_value(parent_object, True)
        effect_flag = leaf_object.states[AttachedTo].get_value(parent_object)
        ontop_flag, fail_cons_list, robot_list = OnTopHolodeck(parent_object, None, env, scene_model=scene_model, room_name=room_name, use_official_api=use_official_api, debug_path_image=debug_path_image,new_added_objects_list=new_objects_name_list, just_for_robot_pose=True)
    elif rel == "Covered" or rel == "covered":
        parent_object.states[Covered].set_value(leaf_object, True)
        effect_flag = leaf_object.states[Covered].get_value(parent_object)
        ontop_flag, fail_cons_list, robot_list = OnTopHolodeck(parent_object, None, env, scene_model=scene_model, room_name=room_name, use_official_api=use_official_api, debug_path_image=debug_path_image,new_added_objects_list=new_objects_name_list, just_for_robot_pose=True)
    elif rel == "Draped" or rel == "draped":
        leaf_object.states[Draped].set_value(parent_object, True)
        effect_flag = leaf_object.states[Draped].get_value(parent_object)
        ontop_flag, fail_cons_list, robot_list = OnTopHolodeck(parent_object, None, env, scene_model=scene_model, room_name=room_name, use_official_api=use_official_api, debug_path_image=debug_path_image,new_added_objects_list=new_objects_name_list, just_for_robot_pose=True)
    elif rel == "Filled" or rel == "filled":
        #import pdb; pdb.set_trace()
        parent_object.states[Filled].set_value(leaf_object, True)
        effect_flag = leaf_object.states[Filled].get_value(parent_object)
        ontop_flag, fail_cons_list, robot_list = OnTopHolodeck(parent_object, None, env, scene_model=scene_model, room_name=room_name, use_official_api=use_official_api, debug_path_image=debug_path_image,new_added_objects_list=new_objects_name_list, just_for_robot_pose=True)
    elif rel == "Inside" or rel == "inside":
        leaf_object.states[Inside].set_value(parent_object, True)
        effect_flag = leaf_object.states[Inside].get_value(parent_object)
        ontop_flag, fail_cons_list, robot_list = OnTopHolodeck(parent_object, None, env, scene_model=scene_model, room_name=room_name, use_official_api=use_official_api, debug_path_image=debug_path_image,new_added_objects_list=new_objects_name_list, just_for_robot_pose=True)
    elif rel == "OnTop" or rel == "ontop":
        if use_holodeck:
            ontop_flag, fail_cons_list, robot_list = OnTopHolodeck(leaf_object, parent_object, env, scene_model=scene_model, room_name=room_name, use_official_api=use_official_api, debug_path_image=debug_path_image,new_added_objects_list=new_objects_name_list)
        else:
            effect_flag = leaf_object.states[OnTop].set_value(parent_object, True)
            ontop_flag, fail_cons_list, robot_list = OnTopHolodeck(parent_object, None, env, scene_model=scene_model, room_name=room_name, use_official_api=use_official_api, debug_path_image=debug_path_image,new_added_objects_list=new_objects_name_list, just_for_robot_pose=True)
        return effect_flag, fail_cons_list, robot_list
        # leaf_object.states[OnTop].set_value(parent_object, True)
    elif rel == "Overlaid" or rel == "overlaid":
        effect_flag = leaf_object.states[Overlaid].set_value(parent_object, True)
        ontop_flag, fail_cons_list, robot_list = OnTopHolodeck(parent_object, None, env, scene_model=scene_model, room_name=room_name, use_official_api=use_official_api, debug_path_image=debug_path_image,new_added_objects_list=new_objects_name_list, just_for_robot_pose=True)
    elif rel == "Saturated" or rel == "saturated":
        effect_flag = parent_object.states[Saturated].set_value(leaf_object, True)
        ontop_flag, fail_cons_list, robot_list = OnTopHolodeck(parent_object, None, env, scene_model=scene_model, room_name=room_name, use_official_api=use_official_api, debug_path_image=debug_path_image,new_added_objects_list=new_objects_name_list, just_for_robot_pose=True)
    elif rel == "Under" or rel == "under":
        effect_flag = leaf_object.states[Under].set_value(parent_object, True)
        ontop_flag, fail_cons_list, robot_list = OnTopHolodeck(parent_object, None, env, scene_model=scene_model, room_name=room_name, use_official_api=use_official_api, debug_path_image=debug_path_image,new_added_objects_list=new_objects_name_list, just_for_robot_pose=True)
    elif rel == "Unchanged" or rel == "unchanged":
        try:
            original_pos, original_rot, origin_parent_pos, origin_parent_rot = get_original_pos_rot(leaf_obj, obj, init_obj_pos_rot)
            current_parent_pos, current_parent_rot = parent_object.get_position_orientation()
            leaf_object.set_position_orientation(
                original_pos + current_parent_pos - origin_parent_pos,
                original_rot + current_parent_rot - origin_parent_rot
            )
        except:
            print(f"The object {leaf_obj} or {obj} maybe not in the original scene.")
            return False, [], []
    else:
        if leaf_obj not in obj_name_all:
            return True, [], []
        else:
            print("Wrong relation!")
            return False, [], []

    return effect_flag, [], robot_list
    
def get_original_pos_rot(leaf_obj, obj, init_obj_pos_rot):

    if leaf_obj not in init_obj_pos_rot.keys():
        print(f"The object {leaf_obj} is not in the original scene.")
        return
    elif obj not in init_obj_pos_rot.keys():
        print(f"The object {obj} is not in the original scene.")
        return
    else:
        return init_obj_pos_rot[leaf_obj]["pos"], init_obj_pos_rot[leaf_obj]["ori"], init_obj_pos_rot[obj]["pos"], init_obj_pos_rot[obj]["ori"]
    
def velocity_collide(env,target_name, norm_threshold=0.1, room_name = None):
    collide_object_list = []
    for obj in env.scene.objects:
        if obj.name in target_name:
            continue

        if obj.get_linear_velocity().norm() > norm_threshold or obj.get_angular_velocity().norm() > norm_threshold:
            if "in_rooms" in dir(obj) and obj.in_rooms[0] != room_name:
                obj.set_angular_velocity(th.tensor([0., 0., 0.]))
                obj.set_linear_velocity(th.tensor([0., 0., 0.]))
                continue
            collide_object_list.append(obj.name)
    return collide_object_list

def get_image_now(scene_model, room_name, env, full_path, top_view_obj_name = None, uninclude_list = ['floor']):

    
    bbox = camera_for_scene_room(scene_model, room_name, "top_view", "bbox_2d_tight", uninclude_list=uninclude_list, env=env, image_height=1080, image_width=1440, focal_length=14, top_view_obj_name=top_view_obj_name)

    if bbox is None:
        return None
    else:
        bbox, obj_name_list = bbox

    # print(bbox)
    # import pdb; pdb.set_trace()
    image = Image.fromarray(bbox[0].astype('uint8'))

    # 保存图像
    image.save(full_path)
    return obj_name_list[0]
    
if __name__ == "__main__":
    main()