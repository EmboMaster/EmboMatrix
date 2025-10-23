from openai import OpenAI
import os
import re

def call_gpt(model, prompt, system_prompt="You are a helpful assistant.", temperature=0.7, max_tokens=4096):
    # 确保在此处替换为你的实际 API 密钥
    os.environ['OPENAI_API_KEY'] = 'sk-AjgPUQzxcuKCscN3R0IPEru7G4hsAku16srLfzinmmn2AZKE'

    client = OpenAI(base_url="https://ai.sorasora.top/v1")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    output = response.choices[0].message.content.strip()
    return output

def extract_problem(bddl_problem):
    pattern = r"\(\s*define\s*\(\s*problem\s+([^\s\)]+)\s*\)" #匹配define (problem problem_name)
    matches = re.findall(pattern, bddl_problem)
    return matches[0].split('-')[0]

def check_feasibility(bddl_problem,path):
    prompt=f'''You'll be given some tasks, and a series of steps to realize this tasks.
    The tasks we have are:
    1. Navigation: Locate and reach the target object in the environment.
    2. Grasp: Securely grasp the target object and maintain a stable hold.
    3. Carry: Transport the object to the designated location (typically after grasping).
    4. Put: Place the object in the specified position (typically after carrying).
    5. Others: Can't define
    Please check whether the task can be realized by the following steps.
    If it can be realized, just output it with no modification.
    If it can't be realized, give some suggestion about the modification after the steps.
    For example:
    input:
    taskname:filling_a_bird_feeder-0
    1. Find birdseed in bag. (Navigation task)
    2. Grasp bag. (Grasp task)
    3. Carry bag to bird_feeder. (Carry task)
    4. Fill bird_feeder with birdseed. (Can't define)
    you should output:
    taskname:filling_a_bird_feeder-0
    1. Find birdseed in bag. (Navigation task)
    2. Grasp bag. (Grasp task)
    3. Carry bag to bird_feeder. (Carry task)
    4. Fill bird_feeder with birdseed. (Can't define)
    suggestion: there maybe a Put bag step between step3 and step4.
    
    input:
    taskname:filling_a_bird_feeder-0
    1. Find birdseed in bag. (Navigation task)
    2. Carry bag to bird_feeder. (Carry task)
    3. Grasp bag. (Grasp task)
    4. Fill bird_feeder with birdseed. (Can't define)
    you should output:
    taskname:filling_a_bird_feeder-0
    1. Find birdseed in bag. (Navigation task)
    2. Grasp bag. (Grasp task)
    3. Carry bag to bird_feeder. (Carry task)
    4. Fill bird_feeder with birdseed. (Can't define)
    suggestion: the sequence of step2 and step3 maybe wrong.You should grasp bag before carry it.
    
    Here are the input task:{bddl_problem}.
    To be note that you shold output a complete file and only contains task definition without any annotations.
    '''
    response = call_gpt('gpt-4o-2024-08-06',prompt)
    print(response)
    filename = f'{path}/txt/subtask_bddl_reply_final.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(response)
    return response

def extract_pddl_problems(text):
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    problem_matches = []
    start_index = 0
    while True:
        start_match = text.find("(define (problem", start_index)
        if start_match == -1:
            break

        open_paren_count = 2
        end_index = start_match + len("(define (problem")
        while end_index < len(text):
            if text[end_index] == '(':
                open_paren_count += 1
            elif text[end_index] == ')':
                open_paren_count -= 1
                if open_paren_count == 0:
                    problem_matches.append(text[start_match:end_index + 1])
                    start_index = end_index + 1
                    break
            end_index += 1
        else:  # 没有找到匹配的 )，说明 PDDL 格式可能有问题
          print("warning: Unmatched parenthesis detected in the text")
          return [] #或根据实际需求抛出异常

    return problem_matches

def save(subtask_files,reply,path):
    problem = extract_pddl_problems(subtask_files)
    r = reply.split("taskname")[1:]
    for i in range (len(problem)):
        p = problem[i]
        rep = r[i]
        name = extract_problem(p)
        if "Can't" in rep:
            d="undefined_problem"
        else:
            d="defined_problem"
        filename = f"{path}/{d}/{name}/problem0.bddl"
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:  # 指定 utf-8 编码，处理中文等特殊字符
            f.write(p)
        return p,directory
    