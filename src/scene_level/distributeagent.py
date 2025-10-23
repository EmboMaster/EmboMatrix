import re
import json
import random
import sys
from typing import Union
import os
import datetime
from src.scene_level.SynsetTree import Tree
from openai import OpenAI
import copy
from src.llm_selection import get_gpt_response_by_request
from src.utils.config_loader import config
class DistributeAgent:
    def __init__(self,  bddl_input: Union[str, dict], scene_model: str,official_api: bool,csv_path="src/scene_level/BEHAVIOR-1KSynsets.csv",synset2obj_path="src/scene_level/synset_object_category.json",scene_info_path="src/scene_level/scenes_with_descriptions.json",scene_room_size_path="omnigibson/examples/scenes/overhead_view/area.json"):
        """
        初始化 DistributeAgent 类，设置 BDDL 文件路径、场景模型字符串和结果字典。

        :param bddl_path: BDDL 文件路径
        :param scene_model: 场景模型字符串
        """
        if isinstance(bddl_input, str):
            if os.path.isfile(bddl_input):
                with open(bddl_input, 'r') as file:
                    self.bddl_content = file.read()

            else:
                self.bddl_content = bddl_input
        elif isinstance(bddl_input, dict):
            self.bddl_content = json.dumps(bddl_input)
        else:
            raise ValueError("bddl_input 必须是文件路径或 BDDL 文件内容")
        self.scene_model = scene_model  # 场景模型字符串

        self.scene_room_size_path = scene_room_size_path
        self.result_dict = {}  #target_format
        self.synset2instance_dict = {} # synset:instance_name:instance
        self.sceneData = {} #包括obj rejes and init state
        self.room_info = set() #scene里包括的所有房间
        self.scene_info_path = scene_info_path  #提取房间信息的文件路径
        self.synset_mapping_tree = Tree(csv_path)   #同义词树
        with open(synset2obj_path,"r") as f:
            self.synset2obj_dict = json.load(f) #用同义词的叶对应到物体模型的字典
        self.obj_for_each_room=dict()
        self.use_official_api = official_api
        self.model =config['scene_generation']['model']
        self.api_key =config['scene_generation']['api_key']
        self.base_url =config['scene_generation']['base_url']


    def extract_objects_section(self):
        """
        提取 BDDL 文件中的 :objects 部分内容。

        :return: :objects 部分的内容，如果没有找到返回空字符串。
        """
        try:
            objects_match = re.search(r"\(:objects(.*?)\)", self.bddl_content, re.DOTALL)
            if objects_match:
                objects_section = objects_match.group(1)
                return objects_section.strip()
            else:
                return ""
        except Exception as e:
            return ""

    def extract_init_section(self):
        """
        提取 BDDL 文件中的 :init 部分内容，直到 :goal 之前。

        :return: :init 部分的内容，如果没有找到返回空字符串。
        """
        try:
            init_match = re.search(r"\(:init(.*?)\(:goal", self.bddl_content, re.DOTALL)
            if init_match:
                init_section = init_match.group(1)
                return init_section.strip()
            else:
                return ""
        except Exception as e:
            return ""
    def get_bddl_sections(self):
        """
        获取 BDDL 文件中的 :objects 和 :init 部分，并返回一个字典。

        :return: 返回包含 'objects' 和 'init' 部分的字典
        """
        objects_section = self.extract_objects_section().strip().split('\n')
        objects_section = [objects.strip() for objects in objects_section]
        init_section = self.extract_init_section().strip()
        init_section = re.findall(r'\((.*?)\)', init_section)
        excluded_prefixes = {"dusty", "stained", "soaked", "sliced"}

        filtered_inits = [
            init for init in init_section 
            if not init.strip().startswith(tuple(excluded_prefixes))
        ]
        return {"objects": objects_section, "init": filtered_inits}
    
    def get_scene_json(self):
        origin_scene_path = "omnigibson/data/og_dataset/scenes/"+ self.scene_model+"/json/"+self.scene_model+"_best_origin.json"
        scene_path = "omnigibson/data/og_dataset/scenes/"+ self.scene_model+"/json/"+self.scene_model+"_best.json"

        import os
        if os.path.exists(origin_scene_path):  # 如果存在原始场景文件，则使用原始场景文件
            with open(origin_scene_path) as f:
                self.sceneData = json.load(f)
        else:
            with open(scene_path) as f:
                self.sceneData = json.load(f)

    def SynsetMapping_version2(self):
        bddl_info = self.get_bddl_sections()
        #import pdb; pdb.set_trace()
        #print(f"bddl_info{bddl_info}")
        for objects in bddl_info["objects"]:
            try:
                synset=objects.split('-')[1].strip()
            except:
                print(f"wrong Objects! {objects}")
                print("Sys Exit!")
                sys.exit(1)
            if "agent" in synset:
                pass
            elif "floor.n" in synset:
                pass
            else:
                candidates = self.synset_mapping_tree.get_terminal_leaves_by_name(synset)
                try:
                    non_empty_candidates = [candidate for candidate in candidates if candidate.strip()]
                except:
                    print(f"Not have synset {synset}!")
                    print("wrong bddl!")
                    print("Sys Exit!")
                    sys.exit(1)
                    import pdb; pdb.set_trace()
                self.synset2instance_dict[synset] =non_empty_candidates 
                
    def SynsetMapping_foreachroom(self,exist_obj):
        for room in self.obj_for_each_room:
            object_list = self.obj_for_each_room[room]
            #print("self.obj_for_each_room[room]",self.obj_for_each_room[room])
            object_list = list(set(item.rsplit("_",1)[0] for item in object_list))
            #print("objlist",object_list)
            all_obj_in_room = {}
            for k, v in self.sceneData["objects_info"]["init_info"].items():
                if room in v["args"]["in_rooms"]:
                    elem={}
                    elem["Object"] = k
                    elem["Category"] = v["args"]["category"]
                    all_obj_in_room[k] = elem
            
            for synset in object_list:
                flag = False
                candidates = self.synset_mapping_tree.get_terminal_leaves_by_name(synset)
                #print("synset",synset)
                #print("candidate",candidates)
                non_empty_candidates = [candidate for candidate in candidates if candidate.strip()]
                for candidate in non_empty_candidates:                  #candidate: "towel_rack.n.01"
                    if flag == True:
                        break
                    if candidate in self.synset2obj_dict: 
                        for choice in self.synset2obj_dict[candidate]:  
                            if flag == True:
                                break
                            for obj,value in all_obj_in_room.items():
                                if choice["Object"] in obj:
                                    exist_obj[synset] = value["Object"]
                                    self.synset2instance_dict[synset] = value
                                    flag = True 
                                    break
                if flag == False:
                    for choice in non_empty_candidates:
                        if choice in self.synset2obj_dict:
                            self.synset2instance_dict[synset] = self.synset2obj_dict[choice][0]
                            break
                        else: 
                            continue 
        
            
    def getsingleobjstate(self,room, singleobjlist, replace_dict):

        state_mapping = {
            # 原始映射
            "cooked": "Cooked",
            "frozen": "Frozen",
            "open": "Open",
            "folded": "Folded",
            "unfolded": "Unfolded",
            "toggled_on": "ToggledOn",
            "hot": "Heated",
            "on_fire": "OnFire",
            "broken": "Burnt",

            # 包含下划线和大小写变体
            "toggled-on": "ToggledOn",
            "toggled_on": "ToggledOn",
            "toggledOn": "ToggledOn",
            "Toggled_on": "ToggledOn",

            "on_fire": "OnFire",
            "on-fire": "OnFire",
            "onFire": "OnFire",
            "On_fire": "OnFire",

            # 其他大小写变体
            "Cooked": "Cooked",
            "Frozen": "Frozen",
            "Open": "Open",
            "Folded": "Folded",
            "Unfolded": "Unfolded",
            "Hot": "Heated",
            "Broken": "Burnt",
        }

        result = {}
        # #print("singleobjlist\n",singleobjlist)
        # #print("replace_dict\n",replace_dict)
        
        for init_info in singleobjlist:
            elem = init_info.split(" ")  # 分割字符串为列表
            synset, suffix = elem[-1].rsplit("_", 1)
            if elem[-1] in replace_dict:
                #print(f"synset {synset} is in replace dict")
                synset = replace_dict[elem[-1]]
                if len(elem) == 3:
                    status = elem[1].replace("(","",1)
                elif len(elem) ==2:
                    status = elem[0]
                else:
                    pass
                if status in ["future", "real"]:
                    continue
                is_negative = "not" in elem[0]
                if synset not in result:
                    result[synset] = {}
                if status in state_mapping:
                    mapped_status = state_mapping[status]
                    if status == "folded" or status == "unfolded":
                        result[synset][mapped_status] = True
                    else:
                        result[synset][mapped_status] = not is_negative
            else:
                continue

        return result
    def extract_objects(self,room_dict):
        """
        从嵌套的房间字典结构中提取每个房间包含的物体。
        """
        def recursive_extract(sub_dict, objects):
            """
            递归解析嵌套字典，收集物体。
            """
            for key, value in sub_dict.items():
                # 如果值是一个嵌套字典，继续递归
                if isinstance(value, dict):
                    if "_" in key:
                        objects.append(key)  # 当前键是物体，加入列表
                    recursive_extract(value, objects)

        room_objects = {}
        for room, room_content in room_dict.items():
            objects = []
            # 提取每个房间的物体
            recursive_extract(room_content, objects)
            room_objects[room] = objects

        return room_objects
    
    def parse_relations(self,input_string):
        """
        Parse the input string into a list of tuples (obj1, rel, obj2) with obj1 always being the larger object.

        Args:
            input_string (str): Input string containing relationships.

        Returns:
            list: A list of tuples in the format (obj1, rel, obj2).
        """
        relation_order = {
            "ontop": lambda obj1, obj2: (obj2, obj1),  # Ontop: larger object is obj1
            "filled": lambda obj1, obj2: (obj1, obj2),  # Filled: container is obj1
            "insource": lambda obj1, obj2: (obj1, obj2),  # Insource: container is obj1
            "inside": lambda obj1, obj2: (obj2, obj1),  # Inside: larger object is obj1
            "covered": lambda obj1, obj2: (obj1, obj2),  # Covered: larger object is obj1
            "under": lambda obj1, obj2: (obj2, obj1),
            "attached": lambda obj1, obj2: (obj2, obj1),
            "saturated": lambda obj1, obj2: (obj1, obj2),
            "overlaid": lambda obj1, obj2: (obj2, obj1),
            "draped": lambda obj1, obj2: (obj2, obj1),
            "inroom": lambda obj1, obj2: (obj1, obj2),
        }

        tuples = []
        for line in input_string.strip().split("\n"):
            parts = line.split()
            if len(parts) == 3:
                rel, obj1, obj2 = parts
                if rel in relation_order:
                    obj1, obj2 = relation_order[rel](obj1, obj2)
                    tuples.append((obj1, rel, obj2))

        return tuples

    def check_floors_in_single_room(self,objects_list, room_data):
        """
        检查 list 中的 floors 是否仅出现在 dict 中的某一个房间中。

        Args:
            objects_list (list): 包含 floor 信息的列表。
            room_data (dict): 包含房间信息的字典。

        Returns:
            dict: 每个 floor 所在房间的映射。
            list: 同时出现在多个房间的 floors。
        """
        # 提取所有 floor 的标识符
        floor_set = set()
        for obj in objects_list:
            if "floor" in obj:
                # 提取出类似 floor.n.01_1 的信息
                floor_entries = [item for item in obj.split() if item.startswith("floor")]
                floor_set.update(floor_entries)

        # 创建一个映射来跟踪每个 floor 出现的房间
        floor_room_map = {}

        for room, data in room_data.items():
            for tuple in data['tuples']:
                obj = tuple[0]
                if obj in floor_set:
                    if obj not in floor_room_map:
                        floor_room_map[obj] = set()
                    floor_room_map[obj].add(room)

        # 找出出现在多个房间的 floors
        conflicting_floors = [floor for floor, rooms in floor_room_map.items() if len(rooms) > 1]

        return floor_room_map, conflicting_floors

    def DistributeObj(self):
        self.get_scene_json()
        self.SynsetMapping_version2() 
        
        with open(self.scene_info_path,"r") as f:
            scene_info = json.load(f)
        room_info = [room for room ,_ in scene_info[self.scene_model]["Rooms"].items()]
        room_info = "\n".join(room_info)
        bddl_info = self.get_bddl_sections()
        print("bddl_info",bddl_info)
        object_info = bddl_info["objects"]
        print("object_info",object_info)
        agent_tuples = {}
        agent_obj_info = [obj for obj in object_info if "agent" in obj]
        agent_tuples["agent_name"] = agent_obj_info
        object_info = [obj for obj in object_info if "agent" not in obj] 
        old_object_info = copy.deepcopy(object_info)
        object_info = []
        for info in old_object_info:
            tmp_obj = info.split("-")[0].strip().split(" ")
            for obj in tmp_obj:
                object_info.append(obj)
        overall_object_number = len(object_info)
        object_info = "\n".join(object_info)
        init_info = bddl_info["init"]
        print("init_info",init_info)

        old_init_info = copy.deepcopy(init_info)
        room_num = sum([1 for info in old_init_info if "inroom" in info])
        init_info = [info for info in init_info if "agent" not in info]
        agent_init_info = [info for info in old_init_info if "agent" in info]
        agent_tuples["agent_init"] = agent_init_info
        
        SingleObjState_list = [] 
        for info in init_info[:]:
            if "not" in info.split(" ")[0] or len(info.split(" ")) <= 2:
                SingleObjState_list.append(info)
                init_info.remove(info)

        init_info = "\n".join(init_info) 
        init_info = self.parse_relations(init_info)
        floor_hint = ""
        error_hint = ""
        prompt = [
                    {
                        "role": "user",
                        "content": f"""You are an object distribution expert, and your task is to assign objects for robot training to different rooms according to the input information.
I will provide you with all objects' information and the relation information among these objects which are shown in several tuples. 

There are some rules you need to follow:
1. For each room mentioned in the input, at least one relation will clarify that a certain object is in that room.
2. If two objects have a relation between them, they must be in one room.
3. You can not make the same object name appear in multiple rooms. 
4. All tuples containing "inroom" should not be included in the final result since they are used to specify the room of the objects.
5. You should not modify object names or relations between objects. Just allocate!

{floor_hint}
{error_hint}

Objects information: {object_info}
Relation information: {init_info}

In the end, please provide a result dict, with the format as follows, where the content inside [] is what you need to generate, and the rest of the content should not be generated. The output format should be like this:
{{
    [room name1]: {{
        "objects": [object1, object2, ...],
        "tuples": [(object1, relation, object2), ...]
    }},
    [room name2]: {{
        "objects": [object3, object4, ...],
        "tuples": [(object3, relation, object4), ...]
    }},
    ......
}}
"""
                    }
        ]

        floor_flag = True
        max_trial = 5
        while floor_flag and max_trial > 0:

            max_trial -= 1
            text = get_gpt_response_by_request(model=self.model, api_key=self.api_key ,base_url = self.base_url, messages=prompt, max_try=3, use_official_api=self.use_official_api)
            def extract_json_from_string(input_string):
                """
                从字符串中提取 JSON 字典
                """
                import re

                for i in range(len(input_string)):
                    if input_string[i] == '{':
                        input_string = input_string[i:]
                        break
                
                for i in range(len(input_string)-1, -1, -1):
                    if input_string[i] == '}':
                        input_string = input_string[:i+1]
                        break

                try:
                    result = eval(input_string)
                except:
                    result = "Error"
                    import sys
                    print("Sys Exit!")
                    sys.exit(1)
                return result

            llm_result = extract_json_from_string(text)
            tuple_flag = False
            for room, data in llm_result.items():
                if "tuples" not in data.keys():
                    print(f"Error: The room {room} does not contain any tuples!")
                    error_hint = f"Hint: Each room should have a tuple! It can be empty, but it should be there!"
                    tuple_flag = True
                    break
            if tuple_flag:
                continue 

            floor_room_map, conflicting_floors = self.check_floors_in_single_room(bddl_info["objects"], llm_result)
            if len(conflicting_floors) == 0:
                floor_flag = False
            else:
                floor_hint = f"Hint: The following floors are assigned to multiple rooms: {', '.join(conflicting_floors)}. Please make sure a floor in just one room!"
                print(f"Conflicting floors: {conflicting_floors}!")
                continue
            
            # check objects and tuples
            result_tuples_length = sum([len(value["tuples"]) for room, value in llm_result.items()])
            if result_tuples_length != len(init_info) - room_num:
                error_hint = f"Hint: The overall number of tuples in the result must be equal to the original number of tuples! Please check the tuples and try again!"
                print(error_hint)
                floor_flag = True
                continue

            # import pdb; pdb.set_trace()
            result_objects_length = sum([len(value["objects"]) for room, value in llm_result.items()])
            if result_objects_length != overall_object_number:
                error_hint = f"Hint: The overall number of objects in the result must be equal to the original number of objects! Please check the objects and try again!"
                floor_flag = True
                print(error_hint)
                continue

            already_in_room = set()
            for room, value in llm_result.items():
                tuple = value["tuples"]
                for obj1, rel, obj2 in tuple:
                    if obj1 not in value["objects"] or obj2 not in value["objects"]:
                        error_hint = f"Past Error: You should keep {obj1} and {obj2} which are in one tuple be placed in the same room !"
                        floor_flag = True
                        print(error_hint)
                        break
                if floor_flag:
                    break

                for obj in value["objects"]:
                    if obj in already_in_room:
                        error_hint = f"Past Error: You should not place the same object in multiple rooms!"
                        floor_flag = True
                        print(error_hint)
                        break
                    already_in_room.add(obj)
            
            if floor_flag:
                continue
        
        if max_trial <= 0 and floor_flag:
            print("Failed to get the correct result for floor distribution!")
            print("Sys Exit!")
            sys.exit(1)

        json_text = {}
        for room, value in llm_result.items():
            try:
                tree = self.triples_to_tree(value["tuples"], value["objects"])
            except:
                print("Error for the tuples and functions. Must Mofidy! Not use LLM to modify!")
            
            json_text[room] = tree
        # print("json_text",json_text)
        self.obj_for_each_room = self.extract_objects(json_text)
        first_time = True
        self.result_dict["scene_name"] = self.scene_model
        replaced_dict = {}
        for room ,value in json_text.items():
            old_value = value
            if "rooms" not in self.result_dict:
                self.result_dict["rooms"] = {}
            new_element = {}
            new_element["room_name"] = room
            new_element["initial_room_image_path_list"] = room
            #import pdb; pdb.set_trace()
            new_element["new_added_tree"],new_element["new_added_objects"],new_element["new_added_objects_list"],tmp_replaced_dict= self.postprocess(room,value,first_time)
            new_element["single_obj_state"] = self.getsingleobjstate(room,SingleObjState_list,tmp_replaced_dict)
            tmp_replaced_dict = self.get_real_replace_dict(tmp_replaced_dict,llm_result[room]['objects'],new_element["new_added_tree"],old_value)
            replaced_dict.update(tmp_replaced_dict)
            if len(new_element["new_added_tree"].keys()) > 0:
                if "floor" in list(new_element["new_added_tree"].keys())[0]:
                    floor_name = list(new_element["new_added_tree"].keys())[0]
                    if new_element["new_added_tree"][floor_name] == {}:
                        new_element["new_added_tree"] = {}
            if new_element["new_added_tree"] != {}:
                self.result_dict["rooms"][room] = new_element
            first_time = False
        return self.result_dict, replaced_dict, agent_tuples
    def get_obj_room(self,obj_name):
        for room,objlist in self.obj_for_each_room.items():
            objlist = [obj.rsplit("_",1)[0] for obj in objlist]
            if obj_name in objlist:
                return room
        return None

    def get_real_replace_dict(self, replace_dict, objects, new_added_tree, tuples):

        for object in objects:
            if "floor" in object:
                value = replace_dict.pop("floors")
                replace_dict[object] = value

        return replace_dict
    
    def triples_to_tree(self, triples, objects):
        """
        Convert a list of triples into a nested tree dictionary.
        Starts with root categories (e.g., floors, ceiling, wall) and iteratively adds connected objects.

        Args:
            triples (list of tuples): List of triples (obj1, relation, obj2).

        Returns:
            dict: A nested tree structure representing the triples.
        """

        new_triples = []
        for triple in triples:
            if 'floor' in triple[0]:
                new_triples.append(("floors", triple[1], triple[2]))
            else:
                new_triples.append(triple)
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

        #import pdb; pdb.set_trace()
        for obj in objects:
            flag = False
            if "floor" in obj:
                obj = "floors"
            for triple in triples:
                if obj == triple[0] or obj == triple[2]:
                    flag = True
                    break
            if not flag:
                tree[obj] = {}

        return tree
    
    def replaceSynsetdict(self,room,first_time):
        #print("obj in room\n\n",self.obj_for_each_room) 
        same_name_list = []

        for k, v in self.sceneData["objects_info"]["init_info"].items():
            if v["args"]["category"] == "floors" and room in v["args"]["in_rooms"]:
                print(f"find floors in room {room}!")
                if "floors" not in self.synset2instance_dict:
                    self.synset2instance_dict["floors"] = []
                self.synset2instance_dict["floors"].append(k)
                break 
        
        for k, terminal_synset_leaves in self.synset2instance_dict.items():
            if "floors" in k:
                continue
            matched = False
            if first_time == True:
                for leaf in terminal_synset_leaves:
                    if self.synset_mapping_tree.get_state_by_name(leaf) == "Substance":
                        self.synset2instance_dict[k] = {
                        "type": "DatasetObject",
                        "name": k,
                        "category": "substance",
                        "model": "substance",
                        "scale": [1.0, 1.0, 1.0],
                        "in_rooms": [room],
                        "exist":False 
                        }  
                        matched = True 
                    else:
                        for obj_in_scene ,obj_info in self.sceneData["objects_info"]["init_info"].items():
                            if obj_info["args"]["category"] == leaf.split('.')[0] and self.get_obj_room(leaf) in obj_info["args"]["in_rooms"]: 
                                self.synset2instance_dict[k] = {
                                    "type": "DatasetObject",
                                    "name": obj_in_scene,
                                    "category": obj_info["args"]["category"],
                                    "model": obj_info["args"]["model"],
                                    "scale": [
                                                1.0,
                                                1.0,
                                                1.0
                                            ],
                                    "in_rooms": [room], 
                                    "exist":True 
                                }
                                matched = True
                            elif obj_info["args"]["category"] == leaf.split('.')[0] and not self.get_obj_room(leaf) in obj_info["args"]["in_rooms"]:
                                same_name_list.append(obj_in_scene)
                    if not matched:
                        option = []
                        random.shuffle(terminal_synset_leaves)
                        if not terminal_synset_leaves:
                            print(f"for {k},can not find an instance name")
                            print("Sys Exit!")
                            sys.exit(1)
                        #print("terminal_synset_leaves",terminal_synset_leaves)
                        for choice in terminal_synset_leaves:
                            if choice in self.synset2obj_dict:
                                option.append(self.synset2obj_dict[choice])
                                break
                        #print("option",option)
                        if option == [] :
                            self.synset2instance_dict[k] = {
                            "type": "DatasetObject",
                            "name": k,
                            "category": "substance",
                            "model": "substance",
                            "scale": [1.0, 1.0, 1.0],
                            "in_rooms": [room],
                            "exist":False 
                        } 
                        else:
                            random_obj = random.choice(option)[0]  # 随机选择一个 obj
                            
                            #print("random_obj",random_obj)
                            self.synset2instance_dict[k] = {
                                "type": "DatasetObject",
                                "name": random_obj["Object"],
                                "category": random_obj["Category"],
                                "model": random_obj["Object"].split('_')[-1].strip(),
                                "scale": [1.0, 1.0, 1.0],
                                "in_rooms": [room],
                                "exist":False 
                            } 
            else:
                self.synset2instance_dict[k]["in_rooms"] = [room]

        # import pdb;pdb.set_trace()
        return same_name_list

    def process_tree_recursive(self,current_dict, parent_key,replaced_keys,replaced_dict, already_added_object = [], same_name_list = []):
        result = {}
        tree_replaced_keys = {}
        for key, value in current_dict.items():
            flag = False
            old_key = key
            # 如果key是 'floors'，替换成 self.synset2instance_dict["floors"][-1]
            if 'floors' in key and key not in already_added_object:
                old_key = key
                flag = True
                key = self.synset2instance_dict["floors"][-1]
                tree_replaced_keys[old_key] = key
                already_added_object.append(old_key)

            # 如果key包含下划线，查找下划线前部分是否存在于self.synset2instance_dict
            if '_' in key and "floors" not in key:
                #base_key = key.split('_')[0]  # 获取下划线前的部分
                base_key,suffix = key.rsplit('_', 1)
                if base_key in self.synset2instance_dict and key not in already_added_object:
                    flag = True
                    # 替换key并加上数字后缀
                    if self.synset2instance_dict[base_key]["exist"] == False: 
                        old_key = key
                        key = f"{self.synset2instance_dict[base_key]['name']}_{suffix}"
                        while key in same_name_list:
                            suffix = str(int(suffix) + 1)
                            key = f"{self.synset2instance_dict[base_key]['name']}_{suffix}"
                        same_name_list.append(key)
                        replaced_keys.append(key)
                        replaced_dict[old_key] = key
                    else:
                        old_key = key
                        key = self.synset2instance_dict[base_key]['name']
                        replaced_dict[old_key] = key 
                    already_added_object.append(old_key)
            # 如果value是字典，递归处理

            # 说明是连接词不用管
            if "floors" not in old_key and '_' not in old_key:
                flag = True

            if isinstance(value, dict) and flag:
                result[key] = self.process_tree_recursive(value, key,replaced_keys,replaced_dict,already_added_object)
            elif flag:
                # 如果是叶子节点，直接加入结果
                if parent_key not in result:
                    result[parent_key] = {}
                result[parent_key][key] = value
            else:
                continue

        replaced_dict.update(tree_replaced_keys)
        
        return result   
    def postprocess(self,room,value,first_time):
        
        same_name_list = self.replaceSynsetdict(room,first_time)
        needed_objects = []
        needed_objects_list = []
        replaced_dict = {}
        already_added_objects = []
        needed_tree = self.process_tree_recursive(value,'',needed_objects,replaced_dict, already_added_objects, same_name_list)
        # #print("needed_objects",needed_objects)
        # #print("synset dict",self.synset2instance_dict)
        for new_obj in needed_objects:
            for k,v in self.synset2instance_dict.items():
                if "floors" in k:
                    continue
                if v["category"] == "substance":
                       continue
                if v["name"] in new_obj :
                   
                   new_element = copy.deepcopy(v)
                   new_element['name'] =new_obj
                   new_element.pop("exist", None) 
                   needed_objects_list.append(new_element) 
        #print("replaced_dict_in_postprocess\n",replaced_dict)
        return needed_tree,needed_objects,needed_objects_list,replaced_dict
        
    
