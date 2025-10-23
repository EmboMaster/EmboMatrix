import requests
import json

def test_env_server():
    # 示例 llm_plan，注意这里将单个 plan 包装在列表中，因为接口要求 llm_plans 为 list 类型
    dummy_llm_plan = {
        "structured_plan": {
            "action_sequence": [
                {"action": "move", "parameters": {"object_index": 1}},
                {"action": "pick_up", "parameters": {"object_index": 1}}
            ]
        },
        "plan": "Test plan: move and pick_up"
    }
    
    payload = {
        "llm_plans": dummy_llm_plan
    }
    
    url = "http://127.0.0.1:5000/get_reward"
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except requests.RequestException as e:
        print("Request failed:", e)

if __name__ == "__main__":
    test_env_server()
