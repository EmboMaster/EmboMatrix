from openai import OpenAI
import requests
import json
import threading
# from src.utils.config_loader import config
# model = config['model']
# api_key = config['api_key']
# base_url = config['base_url']
def get_gpt_response_by_request(model, api_key, base_url, if_json_output=False, messages=[], image_paths=[], max_try=3, use_official_api=False, official_api_base_url=None, official_api_key=None):

    def execute_request(model=model, api_key=api_key, base_url=base_url, if_json_output=False, messages=[], image_paths=[], max_try=3, use_official_api=False, official_api_base_url=None, official_api_key=None):
        base_url, api_key = base_url, api_key
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
        for image_path in image_paths:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": image_path}
            })
        for i in range(max_try):
            try:
                if if_json_output:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=4096,
                        response_format={"type": "json_object"}
                    )
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=4096,
                    )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error using official API (attempt {i+1}/{max_try}): {e}")
                if i == max_try - 1:
                    return None


    def run_with_timeout():
        nonlocal result
        result = execute_request(model, api_key, base_url, if_json_output, messages, image_paths, max_try, use_official_api, official_api_base_url, official_api_key)

    result = None
    while True:
        thread = threading.Thread(target=run_with_timeout)
        thread.daemon = True
        thread.start()
        thread.join(timeout=60)  # 等待2分钟

        if thread.is_alive():
            print("Function execution timed out. Restarting...")
            continue
        else:
            if result is not None:
                return result
            else:
                print("Function failed to produce a result. Restarting...")

import io
import base64
from PIL import Image


def convert_images_to_base64(image_paths, target_width=800, save_path = None):
    """
    将图片路径列表中的图片转换为 Base64 编码的字符串，并在转换前将图片宽度设置为 target_width，
    高度按比例缩放。

    Args:
        image_paths (list): 图片路径列表。
        target_width (int): 图片缩放后的目标宽度（默认为800像素）。

    Returns:
        list: Base64 编码的图片字符串列表。
    """
    img_path_list = []  # 用于存储 Base64 编码的图片字符串

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                # 如果图片宽度超过 target_width，则等比例缩小宽度到 target_width
                
                if image.width > target_width:
                    print(image.width, image.height)
                    ratio = target_width / image.width
                    new_height = int(image.height * ratio)
                    new_size = (target_width, new_height)
                    print(new_size)
                    image = image.resize(new_size, resample=Image.Resampling.LANCZOS)

                # 将图片保存到字节流缓冲区，并转换为 Base64 字符串
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                buffer.seek(0)
                img_b64_str = base64.b64encode(buffer.read()).decode('utf-8')
                img_path_list.append(f"data:image/png;base64,{img_b64_str}")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

    return img_path_list


# 示例输入
if __name__ == "__main__":

    prompt = """
You are an experienced room designer and now you need to help to put straight_chair_dmcixv_2 on top of floors_qtdpcm_0. Currently, there are 5 objects on top of floors_qtdpcm_0, which are: ['floor_lamp_vdxlda_1', 'floor_lamp_vdxlda_2', 'straight_chair_dmcixv_0', 'straight_chair_dmcixv_1', 'straight_chair_dmcixv_2']. Please help me arrange straight_chair_dmcixv_2 in the room by assigning constraints. Here are the constraints and their definitions:

    1. global constraint: 1) edge: at the edge of the floor, close to the wall. 2) middle: not close to the edge of the room. 3) random: randomly placed on the floor.. 
    2. distance constraint: 
        1) near, object: near to the other object, but with some distance, 0 < distance < 2.    
        2) far, object: far away from the other object, distance > 2. 
    3. position constraint: 
        1) in front of, object: in front of another object. 
        2) side of, object: on the side (left or right) of another object. 
    4. alignment constraint: 
        1) center aligned, object: align the center of the object with the center of another object. 
    5. Rotation constraint: 
        1) face to, object: face to the center of another object.

    You must have one global constraint and you can select various numbers of constraints and any combinations of them and the output format must be like below: 
        object | global constraint | constraint 1 | constraint 2 | ... 
    If you choose 'random' as the global constraint, you can ignore other constraints.
    For example: 
        bottle-0 | random
        sofa-0 | edge 
        coffee table-0 | middle | near, sofa-0 | in front of, sofa-0 | center aligned, sofa-0 | face to, sofa-0 
        tv stand-0 | edge | far, coffee table-0 | in front of, coffee table-0 | center aligned, coffee table-0 | face to, coffee table-0

    Here are some guidelines for you to follow:
    1. If there are no objects on top of floors_qtdpcm_0, you just choose 'random' as the global constraint.
    2. If you think that the position and orientation of straight_chair_dmcixv_2 are not important, you can also choose 'random' as the global constraint but I suggest that you do not do that often for we want a tidy and spacious room.
    3. 'edge' can only be chosen for the case when you put objects on floor.
    4. each type of constraints can only be used not more than once. For example, you can not make 'near, object1' and 'near, object2' in the same output. 
    5. you can only use objects in the constraints that are in the below list: ['floor_lamp_vdxlda_1', 'floor_lamp_vdxlda_2', 'straight_chair_dmcixv_0', 'straight_chair_dmcixv_1', 'straight_chair_dmcixv_2'].
    
    Please first use natural language to explain your high-level design strategy, and then follow the desired format *strictly* (do not add any additional text at the beginning or end) to provide the constraints for each object.

"""

    messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": prompt
            }]
    }]

    image_paths = convert_images_to_base64(image_paths, target_width=800)


    print(get_gpt_response_by_request(messages=messages, image_paths=image_paths, use_official_api=True))