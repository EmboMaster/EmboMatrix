import requests
import concurrent.futures
import json

# 定义要覆盖的 scene/task 列表，其中第三个参数 count 表示该组合启动的环境数量（测试时只发送一个请求即可）
selected_scenes_tasks = [
    # ("Pomaria_0_int", "recycling_office_papers", 1),
    #("restaurant_diner", "heat_cookie_stove_shelf_bar", 1)
    ("office_vendor_machine", "pickup_fries_cook_table", 1),
    # ("Pomaria_0_int", "recycling_office_papers", 1),
    # ("Pomaria_0_int", "recycling_office_papers", 1),
    # ("Pomaria_0_int", "recycling_office_papers", 1),
    # ("Pomaria_0_int", "recycling_office_papers", 1),
    # ("Pomaria_0_int", "recycling_office_papers", 1),
    # ("restaurant_urban", "clean_up_after_a_dinner_party", 1),
    # ("restaurant_brunch", "set_a_dinner_table", 1),
    # ("house_single_floor", "delivering_groceries_to_doorstep", 2)
]

# 示例的 llm_plan（由于网关中定义的模型 llmplan 是 str 类型，所以这里将 dict 转为 JSON 字符串）
dummy_llm_plan = {
    "structured_plan": {
        "action_sequence": [
            {"action": "move", "parameters": {"object_index": 1}},
            {"action": "pick_up", "parameters": {"object_index": 1}}
        ]
    },
    "plan": "Test plan: move and pick_up"
}

worse_llm_plan = "12eswalfhjasklfhb"


# object index 49: recycling bin 1, object index 50: legal document 1, object index 51: legal document 2, object index 52: legal document 3.

# smart_llm_plan_dict = {
#         "action_sequence": [
#             {'action': 'move', 'parameters': {'object_index': 1}},
#             {'action': 'pick_up', 'parameters': {'object_index': 1}},
#             {'action': 'move', 'parameters': {'object_index': 2}},
#             {'action': 'place', 'parameters': {'object_index': 2, 'relation': 'ontop'}},
#             {'action': 'move', 'parameters': {'object_index': 0}},
#             {'action': 'place', 'parameters': {'object_index': 0, 'relation': 'ontop'}},
#             {'action': 'toggle_on', 'parameters': {'object_index': 0}},
#             {'action': 'cook_object_with_tool', 'parameters': {'object_index': 1, 'source_index': 0}},
#             {'action': 'pick_up', 'parameters': {'object_index': 1}},
#             # {'action': 'move', 'parameters': {'object_index': 52}},
#             # {'action': 'pick_up', 'parameters': {'object_index': 52}},
#             # {'action': 'move', 'parameters': {'object_index': 49}},
#             # {'action': 'place', 'parameters': {'object_index': 49, 'relation': 'inside'}},
#         ],
#         "plan": "Test plan: move and pick_up"
#     }

smart_llm_plan_dict = {
        "action_sequence": [
            {'action': 'move', 'parameters': {'object_index': 0}},
            {'action': 'pick_up', 'parameters': {'object_index': 1}},
            {'action': 'move', 'parameters': {'object_index': 2}},
            {'action': 'place', 'parameters': {'object_index': 2, 'relation': 'inside'}},
            {'action': 'toggle_on', 'parameters': {'object_index': 2}},
            {'action': 'cook_object_with_tool', 'parameters': {'object_index': 1, 'source_index': 2}},
            {'action': 'toggle_off', 'parameters': {'object_index': 2}},
            {'action': 'pick_up', 'parameters': {'object_index': 1}},
            {'action': 'move', 'parameters': {'object_index': 3}},
            {'action': 'place', 'parameters': {'object_index': 3, 'relation': 'ontop'}},
            # {'action': 'move', 'parameters': {'object_index': 52}},
            # {'action': 'pick_up', 'parameters': {'object_index': 52}},
            # {'action': 'move', 'parameters': {'object_index': 49}},
            # {'action': 'place', 'parameters': {'object_index': 49, 'relation': 'inside'}},
        ],
        "plan": "Test plan: move and pick_up"
    }
smart_llm_plan = f"json:\n{json.dumps(smart_llm_plan_dict)}"

sample_llm_plan_dict =      {                                                                                                                                                                         
        "action_sequence": [                                                                                                                                                       
            {                                                                                                                                                                      
                "action": "move",                                                                                                                                                  
                "parameters": {                                                                                                                                                    
                    "object_index": 94                                                                                                                                             
                }                                                                                                                                                                  
            },                                                                                                                                                                     
            {                                                                                                                                                                      
                "action": "turn",                                                                                                                                                  
                "parameters": {                                                                                                                                                    
                    "yaw": 0                                                                                                                                                       
                }                                                                                                                                                                  
            },                                                                                                                                                                     
            {                                                                                                                                                                      
                "action": "pick_up",                                                                                                                                               
                "parameters": {                                                                                                                                                    
                    "object_index": 94                                                                                                                                             
                }                                                                                                                                                                  
            },                                                                                                                                                                     
            {                                                                                                                                                                      
                "action": "move_forward",                                                                                                                                          
                "parameters": {                                                                                                                                                    
                    "distance": 5,                                                                                                                                                 
                    "yaw": 0
                }
            },
            {
                "action": "place",
                "parameters": {
                    "object_index": 94,
                    "relation": "inside"
                }
            }
        ],
        "task_status_summary": "The robot has successfully picked up the office papers and placed them inside the recycling bin."
    }

sample_llm_plan = f"json:\n{json.dumps(sample_llm_plan_dict)}\n"

def send_request(scene, task):
    payload = {
        # "llmplan": json.dumps(dummy_llm_plan),
        # "llmplan": json.dumps(worse_llm_plan),
        "llmplan": json.dumps({"llm_plans": smart_llm_plan}),
        # "llmplan": sample_llm_plan,
        "scene": scene,
        "task": task
    }
    url = "http://0.0.0.0:8000/process_request"
    # url = "http://1.13.171.22:50007/process_request"
    try:
        print(payload)
        response = requests.post(url, json=payload, timeout=1000)
        response.raise_for_status()
        result = response.json()
        return f"Scene: {scene}, Task: {task} - Success: {result}"
    except requests.RequestException as e:
        return f"Scene: {scene}, Task: {task} - Request failed: {e}"

def main():
    # 创建线程池，最大并发数设置为列表长度
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_scenes_tasks)) as executor:
        # 提交所有请求任务
        future_to_scene_task = {
            executor.submit(send_request, scene, task): (scene, task)
            for scene, task, _ in selected_scenes_tasks
        }
        # 遍历每个完成的任务
        for future in concurrent.futures.as_completed(future_to_scene_task):
            scene, task = future_to_scene_task[future]
            try:
                data = future.result()
            except Exception as exc:
                print(f"Scene: {scene}, Task: {task} generated an exception: {exc}")
            else:
                print(data)

if __name__ == "__main__":
    main()
