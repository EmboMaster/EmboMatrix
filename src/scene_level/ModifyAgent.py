import base64
import json
import os
import re
from PIL import Image
from io import BytesIO
from openai import OpenAI
import argparse
import time
from utils import read_position_config,update_position_config,euler2quat,quat2euler,change_ori_by_euler,change_ori_by_quat
class ObjectPositionModifyAgent:
    def __init__(self, model="gpt-4o", temperature=0.7, max_tokens=8092):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        parser = argparse.ArgumentParser()
        parser.add_argument("--with_picture", default=True, type=bool, help="Include overhead image in the prompt")
        parser.add_argument("--differnciate", default=True, type=bool, help="Differnciate between position and orientation modification")
        self.args = parser.parse_args() #Before generating the position config file, 
        #, and provide the generated JSON file within <json></json> tags\
        self.prompt = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "I will give you a suggestion on how to modify the objects in the room, the config file of the objects' positio and orientation, and a view of the room with bounding box and there name. I need you to help me modify the objects' position and orientation based on the suggestions I provided.\
                            clarify how the position coordinates should change and provide the incremental values of the coordinates (Δx, Δy, Δz, Δα, Δβ, Δγ)\
                            Estimate the size of the object to avoid collisions\
                            If an object's shape changes or it falls over, it is because the object collided with other objects. In this case, you need to increase the avoidance distance between objects"
                            },
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please respond in the following format: Provide your reasoning within <thinking></thinking> tags, give the position coordinate increments within <results></results> tags.\
                             here are some examples of the output format:\
                             example 1:\
                             <thinking> Your thinking process (Determine the approximate position of the object based on the location information in the config file, and reserve space for the position to avoid collisions based on the spatial information of the object reflected in the image) </thinking>\
                             <results> Δx=-0.1, Δy=0.2, Δz=0, Δα=0, Δβ=0, Δγ=0</results>\
                             example 2:\
                             <thinking> Based on the picture I see, the object has been moved to its target position without disturbing other objects, so there's no need to adjust its position again. </thinking>\
                             <results> Δx=0, Δy=0, Δz=0,Δα=0, Δβ=0, Δγ=0 </results>"
                            }
                        ]
                    }
                ]
    def encode_image_to_base64(self, image_path):
        """Encodes an image to base64 format."""
        with open(image_path, "rb") as image_file:
            img_b64_str = base64.b64encode(image_file.read()).decode("utf-8")
        return img_b64_str
    def save_output_to_txt(self,path,output,j):
        with open(path+"/history.txt", "a") as file:
        # 写入迭代轮数、thinking 和 results 内容
            file.write(f"Iteration {j}\n")
            file.write(f"Output: {output}\n")
            file.write("=" * 40 + "\n")  # 分隔符，方便阅读
    def generate_prompt(self, suggestions=None, object_config=None, suggest_mode=None,incremental=None,image_uri=None):
        if  len(self.prompt) == 2:
            new_content = [ {"type": "text", "text": f"Suggestions:\n{suggestions},\
                             the object position Config is as follows:\n{object_config}"},
                            {"type": "text", "text": "the view of the room is shown below.The task-related objects and their names have already been marked with bounding boxes. Please keep track of the positions of these objects at all times."},
                            {"type": "image_url", "image_url": {"url": image_uri}}
                        ]
            self.prompt.insert(-1, {"role": "user", "content": new_content})
        elif len(self.prompt) == 3:
            new_content = [
                {"type": "text", "text": f"last time of the position you change: (Δx, Δy, Δz): {incremental}"},
                {"type": "text", "text": "After you change the view of the whole room is showed below"},
                {"type": "image_url", "image_url": {"url": image_uri}}
            ]
            self.prompt.insert(-1, {"role": "user", "content": new_content})
        elif len(self.prompt) == 4:
            new_content = [
                {"type": "text", "text": f"last time of the position you change: (Δx, Δy, Δz): {incremental}"},
                {"type": "text", "text": "After you change the view of the whole room is showed below"},
                {"type": "image_url", "image_url": {"url": image_uri}}
            ]
            self.prompt[-2] = {"role": "user", "content": new_content}
        
        return self.prompt
        
        # elif suggest_mode == "ori":
        #     object_config_euler = change_ori_by_euler(object_config) 
        #     if self.args.differnciate and self.args.with_picture:
        #         prompt = [
        #             {
        #                 "role": "user",
        #                 "content": [
        #                     {"type": "text", "text": "I will give you a suggestion on how to modify the objects in the room, the config file of the objects' position and orientation, and a view of the room. I need you to help me modify the objects' orientation and position based on the suggestions I provided.\
        #                     The `ori` is represented using Euler angles (x, y, z) in radians. \
        #                     Since the object is lying flat, you only need to modify the z-axis. \
        #                     Before generating the new config file, please provide the increment in radians (Δz), \
        #                     and then present the new Euler angles as (x, y, z + Δz).\
        #                     Changing only the rotation may cause the object to pass through the wall, so you need to simultaneously modify the object's position coordinates to avoid colliding with the wall.\
        #                     Remember to retain the data that has not been modified."
        #                     },
        #                     {"type": "text", "text": f"Suggestions:\n{suggestions}"},
        #                     {"type": "text", "text": f"the object position Config is as follows:\n{object_config_euler}"},
        #                     {"type": "text", "text": "the view of the room is shown below:"},
        #                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{overhead_image_b64}"}}
        #                 ]
        #             },
        #             {
        #                 "role": "user",
        #                 "content": [
        #                     {"type": "text", "text": "Please provide the modified object config in JSON format:"}
        #                 ]
        #             }
        #         ] 
        

        #return prompt

    def modify_object_positions(self, suggestions, object_config, suggest_mode,incremental_info=None,image_uri=None,path=None,j=None):
        """Modifies object positions based on suggestions."""
        if self.args.with_picture:
            # Encode the overhead image to base64
            # image_base64 = [self.encode_image_to_base64(image_path) for image_path in image]
            # Generate the LLM prompt
            prompt = self.generate_prompt(suggestions, object_config, suggest_mode,incremental_info,image_uri)
        else:
            prompt = self.generate_prompt(suggestions, object_config,suggest_mode) 
        # Call the LLM
        client = OpenAI(api_key = , base_url = )
        #client =OpenAI()
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(5)
        
        # Parse the response to extract modified config
        text = response.choices[0].message.content
        self.save_output_to_txt(path,text,j)
        # 正则表达式提取内容
        thinking_pattern = r"<thinking>(.*?)</thinking>"
        results_pattern = r"<results>(.*?)</results>"

        # 提取内容
        thinking_content = re.search(thinking_pattern, text,re.DOTALL)
        results_content = re.search(results_pattern, text,re.DOTALL)
        thinking_text = thinking_content.group(1).strip() if thinking_content else None
        results_text = results_content.group(1).strip() if results_content else None

        # matches = re.findall(r'<thinking>(.*?)</thinking>|<results>(.*?)</results>|<json>(.*?)</json>', text)
        # print("text",text)
        # print("matches",matches)
        # thinking_content = [match[0] for match in matches if match[0]]
        # results_content = [match[1] for match in matches if match[1]]
        # json_content = [match[2] for match in matches if match[2]]

        return thinking_text, results_text

    def save_modified_config(self, modified_config, output_path):
        """Saves the modified config to a JSON file."""
        with open(output_path, "w") as f:
            json.dump(modified_config, f, indent=2)

