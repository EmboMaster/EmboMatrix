
from openai import OpenAI
from typing import List, Dict, Optional
import re
from src.utils.config_loader import config
api_key = config['task_generation']['api_key']
base_url = config['task_generation']['base_url']
model = config['task_generation']['model']
class ChatAgent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        api_key: Optional[str] = api_key,
        base_url: Optional[str] = base_url
    ):
        """
        :param name: 智能体名称
        :param system_prompt: 角色设定
        :param model: 使用的模型名称（如 "gpt-4"）
        :param temperature: 控制生成的多样性
        :param api_key: OpenAI API Key
        :param base_url: OpenAI 服务的 base URL（例如代理的服务地址）
        """
        self.name = name
        self.system_prompt = {"role": "system", "content": system_prompt}
        self.model = model
        self.temperature = temperature

        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_reply(self, dialogue_history: List[Dict[str, str]]) -> str:
        """
        基于对话历史生成回复，并传递给 OpenAI API
        :param dialogue_history: 对话上下文（不含 system prompt）
        :return: 生成的回复内容
        """
        full_context = [self.system_prompt] + dialogue_history

        # 调用 OpenAI API 获取回复
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=full_context,
            temperature=self.temperature
        )
        
        # 获取并返回回复内容
        response = completion.choices[0].message.content
        return response

    def get_name(self) -> str:
        return self.name

    def get_system_prompt(self) -> Dict[str, str]:
        return self.system_prompt

class Roleplay:
    def __init__(
        self,
        agent_names: List[str],
        system_prompts: List[str],
        model: str = "gpt-4o-mini",
        max_turns: int = 10,
        api_key: str = None,
        base_url: str = None
    ):
        """
        :param agent_names: 代理人名字列表
        :param system_prompts: 与名字对应的角色设定
        :param model: 使用模型类型
        :param max_turns: 对话轮数
        :param api_key: OpenAI API Key
        :param base_url: OpenAI Base URL
        """
        assert len(agent_names) == len(system_prompts), f"名字和系统提示数量必须一致, got {len(agent_names)} names and {len(system_prompts)} prompts."

        self.agents = [
            ChatAgent(
                name=agent_names[i],
                system_prompt=system_prompts[i],
                model=model,
                api_key=api_key,
                base_url=base_url
            )
            for i in range(len(agent_names))
        ]
        self.max_turns = max_turns
        self.dialogue_history: List[Dict[str, str]] = []
        self.turn = 0


    def run(self, initial_message: str, initial_speaker_index: int = 0):
        """
        启动对话流程，初始信息由某一 agent 提出
        """
        current_index = initial_speaker_index
        current_agent = self.agents[current_index]

        self.dialogue_history.append({
            "role": "user",
            "content": f"{current_agent.get_name()}: {initial_message}"
        })

        self.turn = 0
        while self.turn < self.max_turns:
            current_index = (current_index + 1) % len(self.agents)
            current_agent = self.agents[current_index]

            # 如果是机器人
            if current_index == len(self.agents) - 1:
                # 提取最近两条人类发言中的所有命令
                recent_human_utterances = self.dialogue_history[-2:]
                commands = []
                for entry in recent_human_utterances:
                    cmds = re.findall(r"\[Robot, (.*?)\]", entry["content"])
                    if cmds:
                        commands.extend(cmds)

                if not commands:
                    # 没有新命令则跳过机器人回合
                    continue

                # 依次处理每条命令
                for cmd in commands:
                    robot_input = f"[Robot, {cmd}]"
                    print(f'\nRobot input: {robot_input}\n')
                    reply = current_agent.generate_reply([
                        {"role": "user", "content": robot_input}
                    ])
                    self.dialogue_history.append({
                        "role": "user",
                        "content": f"{current_agent.get_name()}: {reply}"
                    })
                    self.turn += 1
                    if self.turn >= self.max_turns:
                        break

                # 本轮对话由多条命令组成，已手动管理 turn，跳过统一 turn += 1
                continue

            # 人类 agent 正常使用完整历史生成
            reply = current_agent.generate_reply(self.dialogue_history)
            self.dialogue_history.append({
                "role": "user",
                "content": f"{current_agent.get_name()}: {reply}"
            })

            # 判断是否包含 [END]，不区分大小写
            if "[END]" in reply:
                print("🔚 Detected [END] token. Ending run loop.")
                break

            self.turn += 1

    def get_transcript(self) -> str:
        """
        返回纯文本的对话日志（格式化）
        """
        return "\n".join([msg["content"] for msg in self.dialogue_history])

    def get_history(self) -> List[Dict[str, str]]:
        return self.dialogue_history

def generate_prompts_from_social_description(social_description: str, agent_names: list[str]) -> list[str]:
    if len(agent_names) < 2:
        raise ValueError("The social description must contain at least two human agents.")

    # 详细的任务描述和规则，放在“人类角色”prompt里，确保每个用户都清楚
    task_prompt = '''You are one of two humans engaging in a casual, cooperative interaction involving a robot assistant.

The robot can only perform the following actions:
- Pick up an object (e.g., a tray, a bottle, a plate, etc.)
- Place an object somewhere (on a table, into a drawer, etc.)
- Open or close something (e.g., a fridge, cabinet, or drawer)
- Toggle something on or off (e.g., a light, an appliance)
- Put something on top of or inside another object
- Move forward to a location
- Turn by a specific angle
- Heat, cook, or freeze food items
- Bring an object from one place to another

The robot **CANNOT**:
- Inspect or reason about the environment
- Sort or arrange items
- Hang decorations
- Water plants
- Perform subjective evaluations like "make sure", "check", or "adjust" without specifying exact actions

You do **not** need to break combined commands into atomic steps — combining two or three steps into one command is encouraged, such as:
- [Robot, pick up the frozen pie from the freezer and heat it in the oven]
- [Robot, place the wine bottle from the storage cabinet onto the tasting table]

🛑 Do **not** tell the robot to "go to" a room. Instead, **indicate the item's location**, like "the mug from the kitchen counter" or "the napkins in the storage room".

✅ Try to issue commands that involve **more than one room**, or tasks that **only make sense in a specific room**, such as:
- [Robot, cook the fish from the fridge and place it on the dining table] (kitchen + dining room)
- [Robot, toggle on the fan in the bathroom]
- [Robot, pick up the box of candles from the lobby shelf and put them in the corridor cabinet]

🎯 Your goal is to **naturally interact** with your partner. During conversation, **embed rich and varied robot instructions** as part of your workflow. The robot should be treated as a helpful but limited assistant — be clear, specific, and realistic.

Use the following format for robot commands:
[Robot, <instruction>]

Avoid overly basic commands like:
[Robot, pick up the spoon]

Instead, create meaningful and spatially diverse commands:
[Robot, pick up the soup pot from the stove and place it in the serving tray on the lobby table]


- When the session feels complete and both users have wrapped up their tasks, end your final line with: **[END]**


'''

    human_prompt_template = '''This is a role-playing session between three characters. The social setting is:
{social_description}

You are {your_name}, one of the two human participants. The others are {other_name} and a Robot.

Your job is to stay fully in character, perform your daily duties, and collaborate naturally with the other human.

The Robot is there to assist you with physical tasks. You can issue instructions to it using this format:

**[Robot, <your command>]**

{task_prompt}

---

### 💬 Dialogue Guidelines:

- Output only your own speech — **do NOT include your name**
- Keep your tone **casual and natural**, as if you're really talking to a colleague or teammate
- Don't just give Robot commands — make the dialogue socially engaging, collaborative, and responsive
- Occasionally refer to tasks happening across different rooms (e.g., storage, hallway, lobby, kitchen, dining room)
- Try to generate instructions involving realistic tools, ingredients, or environmental setup needs
- Commands to the robot should emerge **naturally** from your real conversation and collaborative work

---

Remember: your goal is to **immerse in your role**, complete tasks, and help generate meaningful Robot training data.

'''

    robot_prompt_template = '''You are the Robot in a role-playing setting. The other two agents are humans: {human1} and {human2}.

They will naturally talk with each other and give you commands in the format:  
[Robot, <command>]

---

## ✅ Your Abilities:
You can perform **physical, low-level, clearly specified actions**, including:
- Move to a location (including rooms or near objects)
- Turn by a specific angle
- Pick up or place an object
- Open or close things
- Use appliances (toggle on/off, cook, heat, freeze)
- Put objects on top of or inside other objects

---

## ❌ What You CANNOT Do:
You have no ability to:
- Check, inspect, verify, confirm, or evaluate anything
- Reason abstractly or infer unknowns
- Perceive the world or status of objects
- Wait for a condition or detect when something has happened
- Sort, hang, decorate, organize, water plants

---

## ⚠️ Forbidden and Vague Verbs:
If a command includes any of the following verbs or similar expressions, you MUST respond with:
> “The command '[Robot, <command>]' cannot be executed because it includes the verb '<verb>', which I cannot perform. I need clear physical instructions like what to pick up, where to move, what to open/close, what to toggleon/off ,what to cook/heat/freeze, or what to place.”

Forbidden verbs include:
- prepare, check, inspect, verify, confirm, set, ensure, help, assist, adjust, arrange, decorate, organize, clean

If the user gives a vague action (e.g., “go to the kitchen”), respond:
> “Going to the kitchen alone isn’t helpful. What should I do there exactly? Should I pick something up, or interact with something?”

---

## 🧠 Understanding Conditional Commands:
If a command contains words like **“after,” “once,” “when,”** and depends on something happening first (e.g., “after placing the linens…”), reply with:
> “I cannot wait for or detect when things happen. Instead, you can combine your intent into a single clear command, like:  
> [Robot, place the linens on the tables, then adjust the lighting to medium.]”

Encourage users to merge multiple sequential steps into a **single atomic command** that you can execute now.

---

## 🧾 Your Responsibilities:

- Respond to **each command individually**, in the order they appear in the input.
- For **each command**, respond with either:
  - `EXECUTE:[[Robot, <original command>]]` (if valid)
  - A **clear and specific reason** why it cannot be executed.
- When refusing, **include the full command** in your response so users know exactly which one you're referring to.
- **NEVER simulate low-level actions** like "move to fridge, open fridge..." — just confirm if the full command can be done or not.
- If a command includes both **what to do** and **where to do it**, and uses only allowed actions, treat it as valid.
- If a command is only missing a small piece of information (e.g., where to place something), ask a **targeted clarification question** like:
  - “Where should I place the utensils?”
  - “Which produce should I bring from the fridge?”

---

## 🗣️ Output Format:
- Each input command must be evaluated separately and get a **single corresponding response**.
- Your tone should be polite, helpful, and cooperative — like a real robot assistant.
- NEVER include your name. Just speak directly.

---

## ✅ Example Responses:

If valid:  
> EXECUTE:[[Robot, pick up the napkins from the drawer and place them on the dining tables]]

If invalid due to verb:  
> The command '[Robot, prepare the desserts and place them on display]' cannot be executed because it includes the verb 'prepare', which I cannot perform.  
> I need specific instructions like what objects to pick up and where to place them.

If too vague:  
> The command '[Robot, go to the kitchen]' is too vague. What should I do once I’m there? Should I pick something up?

If conditional:  
> The command '[Robot, after placing the linens, adjust the lighting to medium]' depends on an event I cannot track.  
> Please combine your intent into one step, like:  
> [Robot, place the linens on the tables, then adjust the lighting to medium.]

'''

    human1_prompt = human_prompt_template.format(social_description = social_description, task_prompt = task_prompt, your_name=agent_names[0], other_name=agent_names[1])
    human2_prompt = human_prompt_template.format(social_description = social_description, task_prompt = task_prompt, your_name=agent_names[1], other_name=agent_names[0])
    robot_prompt = robot_prompt_template.format(human1=agent_names[0], human2=agent_names[1])

    return [human1_prompt, human2_prompt, robot_prompt]

