import sys
import os

# 添加项目路径到 sys.path
project_path = "omnigibson"
sys.path.append(project_path)
# 导入函数
from src.tree_generator.multi_level_tree_generator import step2_main
from src.tree_generator.layoutgpt_generator import layoutgpt_main
from src.room_creater.room_creater import step3_main
from omnigibson.utils.camera_utils import camera_for_scene_room
from src.llm_selection import convert_images_to_base64
from PIL import Image
from src.scene_level.distributeagent import DistributeAgent
import omnigibson as og
from omnigibson.object_states import Cooked, Frozen, Open, Folded, Unfolded, ToggledOn, Heated, OnFire, Burnt
import sys
import datetime
import os
import json, re

class PrintToFile:
    def __init__(self, folder_path="src/logs_new",scene_name=None, task_name=None, gpu_id = 0):
        # 创建存放日志文件的目录（如果不存在）
        os.makedirs(folder_path, exist_ok=True)
         
        # 根据当前日期时间生成唯一的文件名
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if scene_name is not None and task_name is not None:
            bddl_name = task_name.split('/')[-2]
            bddl_name = bddl_name[:min(15,len(bddl_name))]
            self.file_path = os.path.join(folder_path, f"gpu{gpu_id}_{timestamp}_{scene_name}_{bddl_name}.txt")
        else:
            self.file_path = os.path.join(folder_path, f"log_{timestamp}.txt")
        
        self.original_stdout = sys.stdout  # 保存原始标准输出

    def write(self, message):
        # 如果是字典类型，将其转化为 JSON 格式
        if message[0] == "{" and message[-1] == "}":
            try:
                tmp_message = eval(message)
            except:
                tmp_message = message
            if isinstance(tmp_message, dict):
                message = tmp_message
                with open(self.file_path, 'a', encoding='utf-8') as file:
                    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
                    file.write(timestamp +"\n")
                    json.dump(message, file, indent=4, ensure_ascii=False)
            else:
                message = message
                # 将消息写入文件
                with open(self.file_path, 'a') as file:
                    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
                    file.write(timestamp + message + "\n")
        else:
            message = message
            # 将消息写入文件
            with open(self.file_path, 'a') as file:
                timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
                file.write(timestamp + message + "\n")
        
        # 同时打印到控制台
        self.original_stdout.write(str(message) + "\n")

    def flush(self):
        """保持与标准输出行为一致，避免出错"""
        self.original_stdout.flush()


# args input
import argparse

# 创建命令行解析器
parser = argparse.ArgumentParser(description="Process image and task parameters.")

# 添加参数
parser.add_argument('--image_height', type=int, help="Height of the image", default=1080)
parser.add_argument('--image_width', type=int, help="Width of the image", default=1440)
parser.add_argument('--scene_name', type=str, help="Name of the scene", default="Rs_garden")
parser.add_argument('--task_name', type=str, help="Name of the task", default="open_backpack_bedroom_close")
parser.add_argument("--gpu_id",type=int,default=6,help="GPU ID for the simulator")
parser.add_argument("--official_api",type=bool,default=True,help="whether use official_api")
parser.add_argument("--bddl_directory",type=str,default="data-0506-our/Rs_garden/open_backpack_bedroom_close/problem0.bddl",help="bddl path")
parser.add_argument("--save_final_path",type=str,default="src/data/tasks/generated_scenes_0520",help="save_final_path")
parser.add_argument("--debug",type=bool,default=True,help="whether debug")
parser.add_argument("--using_holodeck",type=bool,default=True,help="whether use holodeck")
parser.add_argument("--using_tree",type=bool,default=True,help="whether use tree")
parser.add_argument("--using_layoutgpt",type=bool,default=False,help="whether use layoutgpt")
parser.add_argument("--scene_file",type=str,default=None,help="whether use scene_file")


# 解析命令行参数
args = parser.parse_args()
debug_path_image = f"{args.save_final_path}/holodeck_image"
if args.debug:
    image_path_rule = f"{args.save_final_path}/image"
    log_path_rule = f"{args.save_final_path}/logs"
    sample_pose_rule = f"{args.save_final_path}/sample_pose"
else:
    image_path_rule = None
    log_path_rule = "src/logs_new"

os.makedirs(image_path_rule, exist_ok=True)
os.makedirs(debug_path_image, exist_ok=True)
os.makedirs(sample_pose_rule, exist_ok=True)
os.makedirs(log_path_rule, exist_ok=True)

# 使用传递的参数
image_height = args.image_height
image_width = args.image_width
scene_name = args.scene_name
official_api=args.official_api
task_name = args.task_name
from omnigibson.macros import gm
gm.GPU_ID = args.gpu_id
bddl_directory = args.bddl_directory
save_final_path = args.save_final_path
use_holodeck = args.using_holodeck
use_tree = args.using_tree
use_layoutgpt = args.using_layoutgpt
scene_file = args.scene_file

def redirect_print_to_file():
    sys.stdout = PrintToFile(folder_path=log_path_rule, scene_name=scene_name, task_name=bddl_directory, gpu_id=args.gpu_id)

redirect_print_to_file()

def get_image_now(scene_model, room_name, env, full_path, if_front_view=False, unincluded_obj_list=None):

    if unincluded_obj_list is None:
        unincluded_obj_list = ["floor","wall","window","door"]

    if not if_front_view:
        bbox = camera_for_scene_room(scene_model, room_name, "top_view", "bbox_2d_tight", uninclude_list=unincluded_obj_list, env=env, image_height=1440, image_width=2560, focal_length=14)

        if bbox is None:
            print("Failed to get top view image!")
            return
        
        bbox =  bbox[0][0]

    else:
        front_view_img_list = camera_for_scene_room(scene_model, room_name, 'front_view', 'bbox_2d_tight', unincluded_obj_list, env, image_height=image_height, image_width=image_width)
        max_idx, max_value = -1, -1
        if front_view_img_list is None:
            print("Failed to get front view image!")
            return
        front_view_img_list, front_view_obj_list = front_view_img_list[0], front_view_img_list[1]
        for i in range(len(front_view_obj_list)):
            if len(front_view_obj_list[i]) > max_value:
                max_idx, max_value = i, len(front_view_obj_list[i])
        front_view_best_idx = max_idx
        bbox = front_view_img_list[front_view_best_idx]

    image = Image.fromarray(bbox.astype('uint8'))

    # 保存图像
    image.save(full_path)


import logging
import sys
import traceback

problems = bddl_directory.split('/')[-2]

error_path = f"{save_final_path}/error_log/{scene_name}/{problems}"
if not os.path.exists(error_path):
    os.makedirs(error_path)

# 日志配置
log_file = f'{error_path}/problem0.log'
logging.basicConfig(
    filename=log_file,
    level=logging.ERROR,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)

# 全局未捕获异常的处理函数
def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"Uncaught exception: {error_message}")
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    logging.shutdown()

# 设置全局异常捕获
sys.excepthook = log_uncaught_exceptions

def main():

    # a = 2 / 0
    print(f"The task path is {bddl_directory}.")

    agent = DistributeAgent(bddl_directory, scene_name,official_api)
    print("Start to distribute objects...")
    result, inst_to_name, agent_tuples = agent.DistributeObj()

    from src.room_creater.create_scene import scene_creater
    # create a scen
    print("Start to create a scene...")
    env,room_all_obj_name_list_dict, unincluded_obj_list_dict = scene_creater(official_api,scene_name, result, image_height, image_width, skip_step_1 = True, scene_file = scene_file)

    already_modify_objs = {}
    not_tree_objects_list = {}

    for room in result['rooms']:

        room_name = result['rooms'][room]["room_name"]

        # if room_name == 'living_room_1':
        #     continue

        print(f"Start to create {room_name}...")

        img_tmp_path = f"{save_final_path}/{scene_name}/{problems}"
        if not os.path.exists(img_tmp_path):
            os.makedirs(img_tmp_path)

        room_init_image_path_top_view, room_init_image_path_front_view = f"{save_final_path}/{scene_name}/{problems}/{room_name}_topview.png", f"{save_final_path}/{problems}/{room_name}_frontview.png"
        # initial_room_image_path_list = [room_init_image_path_top_view, room_init_image_path_front_view]
        # initial_room_image_path_list = convert_images_to_base64(initial_room_image_path_list, target_width=800)
        initial_room_image_path_list = []
        new_added_tree = result['rooms'][room]["new_added_tree"]
        new_added_objects = result['rooms'][room]["new_added_objects"]
        new_added_objects_list = result['rooms'][room]["new_added_objects_list"]
        single_states = result['rooms'][room]["single_obj_state"]

        # step 1: identity which objects don't move
        room_all_obj_name_list, unincluded_obj_list = room_all_obj_name_list_dict[room_name], unincluded_obj_list_dict[room_name]
        unincluded_obj_list = list(unincluded_obj_list)

        tree_error_list = []
        tree_suggestion_list = []

        # step 3: create a new room based on the new tree
        def create_new_room_based_tree(scene_name, room_all_obj_name_list, unincluded_obj_list, new_added_tree, new_added_objects, room_name, initial_room_image_path_list, tree_suggestion_list, tree_error_list, image_height, image_width, scene_env=None, already_modify_objs={}, activity_name = "", scene_file = None):
            tree_flag = False
            reset_flag = False
            env = scene_env
            print(f"Start to create tree...")
            while not tree_flag:
                # step 2: generate new tree structure
                for obj in not_tree_objects_list.keys():
                    if not_tree_objects_list[obj] >= 3:
                        if obj not in new_added_objects:
                            unincluded_obj_list.append(obj)
                        else:
                            print(f"Already Try {not_tree_objects_list[obj]} times to add {obj} to the room!")
                            print(f"Can not add {obj} to the room!")
                            import sys
                            print("Sys Exit!")
                            sys.exit(1)

                if not use_layoutgpt:
                    generated_tree, new_added_triples = step2_main(room_all_obj_name_list, unincluded_obj_list, new_added_tree, new_added_objects, room_name, initial_room_image_path_list, official_api=official_api,additional_conditions=tree_suggestion_list, error_tuple=tree_error_list, env=env)  
                else:
                    generated_tree, layout_dict = layoutgpt_main(room_all_obj_name_list, unincluded_obj_list, new_added_tree, new_added_objects, room_name, initial_room_image_path_list, official_api=official_api,additional_conditions=tree_suggestion_list, error_tuple=tree_error_list, env=env)
                    print("Perfect Layout:", layout_dict)
                    
                    for obj_tmp_name in layout_dict.keys():
                        already_modify_objs[obj_tmp_name] = {"pos": layout_dict[obj_tmp_name]["position"], "ori": layout_dict[obj_tmp_name]["orientation"]}

                print("Perfect trees:", generated_tree)  

                sample_pose_final_path = os.path.join(sample_pose_rule, scene_name, activity_name)
                if not os.path.exists(sample_pose_final_path):
                    os.makedirs(sample_pose_final_path)
        
                new_room_img_path_list, tree_flag, env, include_obj_list, error_tuple, holodeck_fail_dict = step3_main(scene_name, room_name, unincluded_obj_list, new_added_objects_list, generated_tree, image_height=image_height, image_width=image_width, if_reset=reset_flag, reset_env=env, if_scene_create=True, already_modify_objs=already_modify_objs, use_official_api=official_api, rule_path=image_path_rule,debug_path_image=debug_path_image, sample_pose_final_path=sample_pose_final_path,use_holodeck = use_holodeck, use_tree = use_tree, scene_file = scene_file)
                if not tree_flag:
                    print("Fail to create room by tree, retrying...")
                    if isinstance(error_tuple, str):
                        print(error_tuple)
                    else:
                        obj1, rel, obj2 = error_tuple
                        if obj1 in not_tree_objects_list.keys():
                            not_tree_objects_list[obj1] += 1
                        if obj2 not in not_tree_objects_list.keys():
                            not_tree_objects_list[obj2] = 1
                        if obj2 in not_tree_objects_list.keys():
                            not_tree_objects_list[obj2] += 1
                        error_tuple = f"there is an error for {error_tuple}."

                        print(error_tuple)
                    reset_flag = True
                    tree_error_list.append(error_tuple)
                    continue

                # eval tree
                print("Start to evaluate the tree...")
                tree_suggestion = []
                # tree_suggestion = tree_eval(generated_tree, new_added_tree, new_room_img_path_list, official_api,uninclude_obj_list = unincluded_obj_list, new_added_obj_list=new_added_objects, new_added_triples=new_added_triples)
                if len(tree_suggestion) == 0:
                    print("Tree successfully generated!")
                    tree_flag = True
                else:
                    print("Retry to create tree...")
                    for suggestion in tree_suggestion:
                        tree_error_list.append(f"there is an error for {suggestion}.")
                    tree_flag = False
                    reset_flag = True

            room_modify_objects_list = []
            print("include_obj_list: ", include_obj_list)
            for obj_name in include_obj_list:
                if "floor" in obj_name:
                    continue
                obj = env.scene.object_registry("name", obj_name)
                if obj != None:
                    pos, ori = obj.get_position_orientation()
                already_modify_objs[obj_name] = {"pos": pos, "ori": ori}
                room_modify_objects_list.append(obj_name)
            return generated_tree, new_room_img_path_list, env, include_obj_list,tree_suggestion_list, tree_error_list, already_modify_objs, room_modify_objects_list, holodeck_fail_dict

        generated_tree, new_room_img_path_list, env, include_obj_list, tree_suggestion_list, tree_error_list, already_modify_objs, room_modify_objects_list, holodeck_fail_dict = create_new_room_based_tree(scene_name, room_all_obj_name_list, unincluded_obj_list, new_added_tree, new_added_objects, room_name, initial_room_image_path_list, tree_suggestion_list, tree_error_list, image_height, image_width, scene_env=env, already_modify_objs=already_modify_objs, activity_name=problems, scene_file=scene_file)
        
        # import pdb; pdb.set_trace()
        # step 4: fine-tune the objects in room
        semantic_suggestion, semantic_modify_times = {}, 0
        print("Start to evaluate the room...")


        if scene_file == None:
            for obj in single_states.keys():
                real_obj = env.scene.object_registry("name", obj)
                for state in single_states[obj]:
                    try:
                        if state == "Cooked" or state == "cooked":
                            real_obj.states[Cooked].set_value(single_states[obj][state])
                        elif state == "Frozen" or state == "frozen":
                            real_obj.states[Frozen].set_value(single_states[obj][state])
                        elif state == "Open" or state == "open":
                            real_obj.states[Open].set_value(single_states[obj][state])
                        elif state == "Folded" or state == "folded":
                            if single_states[obj][state] == False:
                                real_obj.states[Unfolded].set_value(single_states[obj][state])
                            else:
                                print("Not supported yet")
                        elif state == "Unfolded" or state == "unfolded":
                            if single_states[obj][state] == True:
                                real_obj.states[Folded].set_value(single_states[obj][state])
                            else:
                                print("Not supported yet")
                        elif state == "ToggledOn" or state == "toggledon":
                            real_obj.states[ToggledOn].set_value(single_states[obj][state])
                        elif state == "Heated" or state == "heated":
                            real_obj.states[Heated].set_value(single_states[obj][state])
                        elif state == "OnFire" or state == "onfire":
                            real_obj.states[OnFire].set_value(single_states[obj][state])
                        elif state == "Burnt" or state == "burnt":
                            real_obj.states[Burnt].set_value(single_states[obj][state])
                        else:
                            print(f"Unknown state: {state}")
                    except:
                        print(f"Error when setting state {state} for object {obj}")
        
        print(f"{room_name} created successfully!")

        all_objects = [obj.name for obj in env.scene.objects]
        final_uninclude_obj_list = [obj for obj in all_objects if obj not in list(include_obj_list)]

        try:
            get_image_now(scene_name, room_name, env, room_init_image_path_top_view, if_front_view=False, unincluded_obj_list=final_uninclude_obj_list)
            get_image_now(scene_name, room_name, env, room_init_image_path_front_view, if_front_view=True,unincluded_obj_list=final_uninclude_obj_list)
        except:
            continue

    # step6: save the final room
    task = problems
    save_scene_path = f"{save_final_path}/{scene_name}/{task}.json"
    og.sim.save([save_scene_path])
    print("The final room has been saved to: ", save_scene_path)
    save_scene_path = f"{save_final_path}/{scene_name}/{task}.json"
    new_bddl_path = f"{save_final_path}/{scene_name}/{task}.bddl"

    # step7: post process
    with open(save_scene_path, 'r') as f:
        data = json.load(f)
    inst_to_name["agent.n.01_1"] = "robot0"
    data["metadata"] = {"task": {"inst_to_name": inst_to_name}}
    with open(save_scene_path, 'w') as f:
        json.dump(data, f, indent=4)

    def load_and_copy_data(save_scene_path, scene):
        # 加载save_scene_path JSON文件
        with open(save_scene_path, 'r') as f:
            save_scene_data = json.load(f)
        
        # 加载omnigibson中的含有'template'的JSON文件
        scene_path = f"omnigibson/data/og_dataset/scenes/{scene}/json/"
        template_files = [f for f in os.listdir(scene_path) if 'template' in f and f.endswith('.json')]
        
        if not template_files:
            raise FileNotFoundError(f"No template JSON file found in {scene_path}")
        
        # 选择一个template文件
        template_file_path = os.path.join(scene_path, template_files[0])
        with open(template_file_path, 'r') as f:
            template_data = json.load(f)
        
        # 从template文件中提取"robot0"的"object_registry"和"init_info"
        object_registry_robot0 = template_data.get('state', {}).get('object_registry', {}).get('robot0', {})
        init_info_robot0 = template_data.get('objects_info', {}).get('init_info', {}).get('robot0', {})
        
        # 将这些数据复制到save_scene_data中的相应位置
        if 'state' not in save_scene_data:
            save_scene_data['state'] = {}
        if 'object_registry' not in save_scene_data['state']:
            save_scene_data['state']['object_registry'] = {}
        save_scene_data['state']['object_registry']['robot0'] = object_registry_robot0
        
        if 'objects_info' not in save_scene_data:
            save_scene_data['objects_info'] = {}
        if 'init_info' not in save_scene_data['objects_info']:
            save_scene_data['objects_info']['init_info'] = {}
        save_scene_data['objects_info']['init_info']['robot0'] = init_info_robot0
        
        # 返回修改后的数据
        return save_scene_data

    # step8: remove inroom tuples from bddl
    def remove_inroom_tuples_from_bddl(file_path, inst_to_name, env, new_bddl_path):
        # 读取文件内容
        with open(file_path, 'r') as file:
            content = file.read()

        # 使用正则表达式匹配以(inroom开头的元组
        pattern = r'\(inroom\s+[^\)]+\)'
        inroom_tuples = re.findall(pattern, content)

        room_replace_key = {}
        for tuple in inroom_tuples:
            if "?" in tuple:
                continue
            object_inst, room_name = tuple[1:-1].split(" ")[1], tuple[1:-1].split(" ")[2]
            object_name = inst_to_name[object_inst]
            
            try:
                new_room_name = env.scene.object_registry("name", object_name).in_rooms[0]
            except:
                print(f"{object_name} is not in any room!")
                return
            if room_name in room_replace_key.keys() and new_room_name != room_replace_key[room_name]:
                print(f"Error: object {object_name} is in room {new_room_name} and {room_name} at the same time!")
                room_replace_key[room_name] = new_room_name
            else:
                room_replace_key[room_name] = new_room_name
        
        def replace_words_in_bddl(bddl_content, replacement_dict):
            # 遍历字典中的每一对键值
            for old_word, new_word in replacement_dict.items():
                # 使用正则表达式进行替换，确保只替换整词而不是部分匹配
                pattern = r'\b' + re.escape(old_word) + r'\b'  # \b 是单词边界，避免部分匹配
                bddl_content = re.sub(pattern, new_word, bddl_content)
            return bddl_content
        
        new_content = replace_words_in_bddl(content, room_replace_key)

        # 保存更新后的内容回原文件
        with open(new_bddl_path, 'w') as file:
            file.write(new_content)

        print(f"文件 {new_bddl_path} 中的(:init部分已更新，删除了所有(inroom开头的tuple。")

    remove_inroom_tuples_from_bddl(bddl_directory, inst_to_name, env, new_bddl_path)

    print("Post process done!")

    # 示例调用
    try:
        contain_robot0_data = load_and_copy_data(save_scene_path, scene_name)
        with open(save_scene_path, 'w') as f:
            json.dump(contain_robot0_data, f, indent=4)
    except:
        print("Error when copying robot0 data!")
    

def run_pipeline():
    try:
        # 运行你的主程序逻辑，比如调用 run_single_simulation
        main()
    except Exception as e:
        # 捕获局部异常，记录详细信息
        logging.error("Caught exception during pipeline execution", exc_info=True)
        # 打印到控制台以便实时查看
        print(f"[ERROR] {e}")
        print(traceback.format_exc())
        logging.error(traceback.format_exc())

run_pipeline()

