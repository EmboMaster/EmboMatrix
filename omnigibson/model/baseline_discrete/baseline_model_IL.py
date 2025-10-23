#            ┌───────────────────────────┐
# Language ->|  语言编码器 (BERT/Transformer) 
#            └───────────────────────────┘
#                                 │
# Vision   ->[ CNN/ViT ]-> v      │   l
#                                 │
# RobotSta ->[   MLP   ]-> s      │
#                                 ▼
#  ┌──────────────────────────────────────────────────┐
#  │            High-level Policy (RL)               │
#  │ 多模态融合 (z = f(l, v, s)) -> 输出导航/抓取目标子指令 │
#  └──────────────────────────────────────────────────┘
#                  │                  │
#                  │ Navigation cmd   │
#                  │(vx, vy, vtheta)  │
#                  └──────────────────┘
#                  │
#                  ▼
#     ┌───────────────────────────┐
#     │  Differential Drive PID   │
#     └───────────────────────────┘

#                  │
#                  │ Manipulation cmd
#                  │(joint targets, gripper open/close)
#                  └───────────────────────────────────┐
#                                                      │
#                                                      ▼
#                                         ┌─────────────────────────────┐
#                                         │    Arm IK / Joint PID       │
#                                         └─────────────────────────────┘


###################################################
# 文件：multi_modal_policy.py (示例性结构与伪代码)
###################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from omnigibson.model.baseline.siglip_vision_encoder import SiglipVisionEncoder  # 导入新的视觉编码器
from omnigibson.model.baseline_discrete.policy_network import PolicyNetwork_discrete
from transformers import AutoProcessor
import torchvision.transforms as T
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from typing import Tuple, Optional, Dict
import numpy as np
from typing import Tuple, Optional, Dict, Union
import gymnasium as gym

Task_Action_Mask = {
    'Navigation task': torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]) + 1e-6,
    'Grasp task': torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]) + 1e-6,
}

# lower_level_actions_dict = {
#     "forward": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "sharp_left": [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "slight_left": [0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "sharp_right": [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "slight_right": [1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "stop": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "backward": [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# }

lower_level_actions_dict = {
    "forward": [0.533, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "sharp_left": [0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "slight_left": [0, 0.06, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "sharp_right": [0, -0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "slight_right": [0, -0.06, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "stop": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "backward": [-0.533, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

# lower_level_actions_dict = {
#     "forward": [0.533, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "sharp_left": [0.533, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "slight_left": [0.533, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "sharp_right": [0.533, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "slight_right": [0.533, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "stop": [0.533, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "backward": [0.533, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# }


# 将每个动作转换为torch.tensor并存入列表
lower_level_action_tensors_nav = [torch.tensor(value, dtype=torch.float32).unsqueeze(0) for value in lower_level_actions_dict.values()]
lower_level_action_tensors = {
    "Navigation task": lower_level_action_tensors_nav,
    "Grasp task": torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32)
}

class LanguageEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim=128,                    # 输出的语言特征维度
        vocab_size=None,                   # 仅在 use_precomputed=False 时需要
        embed_dim=128,                     # 仅在 use_precomputed=False 时需要
        use_precomputed=True,
        pretrained_lang_dim=4096,          # 仅在 use_precomputed=True 时需要
        precomupted_lang_path="bddl_subgoals/bddl_subgoals/activity_definitions/recycling_office_papers/problem0_subgoals.pt",        # 仅在 use_precomputed=True 时需要
        device='cuda',
    ):
        """
        Args:
            hidden_dim (int): 输出语言特征的隐层维度
            vocab_size (int): 若使用内置embedding + LSTM，则需词表大小
            embed_dim (int): 若使用内置embedding + LSTM，则embedding维度
            use_precomputed (bool): 是否使用外部预提取好的embedding
            pretrained_lang_dim (int): 预提取好的语言embedding的维度 (如 LLaMA encoder 输出)
        """
        super().__init__()
        self.use_precomputed = use_precomputed
        self.hidden_dim = hidden_dim

        if self.use_precomputed:
            # 直接读取文件中的预训练语言特征
            # pass
            if precomupted_lang_path is None:
                raise ValueError("Must specify precomupted_lang_path when use_precomputed=True!")
            else:
                self.precomputed_lang_embedding = torch.load(precomupted_lang_path)
                # 将self.precomputed_lang_embedding中是tensor的部分放到device上
                for k, v in self.precomputed_lang_embedding.items():
                    if isinstance(v['embedding'], torch.Tensor):
                        self.precomputed_lang_embedding[k]['embedding'] = v['embedding'].to(device).unsqueeze(0)

            if pretrained_lang_dim is None:
                raise ValueError("Must specify pretrained_lang_dim when use_precomputed=True!")
            self.pretrained_proj = nn.Sequential(
                nn.Linear(pretrained_lang_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            # 内置 LSTM 编码
            if vocab_size is None:
                raise ValueError("Must specify vocab_size when use_precomputed=False!")
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
            self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, language_instruction=None):
        """
        Input x: [B, seq_len]
        """
        if self.use_precomputed:
            lang_feat = self.precomputed_lang_embedding[language_instruction]['embedding']
            # lang_feat = torch.randn(x.size(0), self.hidden_dim)  # 假设直接返回随机特征
        else:
            # 内置embedding + LSTM
            # x: [B, seq_len]
            embed = self.embedding(x)              # [B, seq_len, embed_dim]
            _, (h, c) = self.rnn(embed)           # h: [1, B, hidden_dim]
            lang_feat = h.squeeze(0)              # [B, hidden_dim]
        return lang_feat

class RobotStateEncoder(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        """
        机器人状态编码器，通过全连接层将状态向量编码为特征向量。

        Args:
            state_dim (int): 机器人状态向量的维度。
            hidden_dim (int): 编码后的特征维度。
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, state_input):
        """
        Args:
            state_input (torch.Tensor): [B, state_dim]

        Returns:
            torch.Tensor: [B, hidden_dim]
        """
        state_feat = self.fc(state_input)
        return state_feat

class MultiModalRobotPolicy_Discrete(nn.Module):
    def __init__(
        self,
        # --- 语言相关 ---
        observation_space=None,
        action_space=None,
        lr_schedule=None,
        use_sde=None,
        use_precomputed_lang=True,
        vocab_size=None,
        embed_dim=128,
        pretrained_lang_dim=4096,
        lang_hidden_dim=4096,
        # --- 视觉相关 ---
        vision_tower_name: str = "google/siglip-so400m-patch14-384",  # 示例模型名称
        mm_vision_select_feature: str = "cls_patch",        # 'patch' 或 'cls_patch'
        unfreeze_mm_vision_tower: bool = False,            # 是否解冻视觉塔
        vision_output_dim: int = 4096,                      # 视觉特征输出维度
        # --- 机器人状态相关 ---
        state_dim=156,
        state_hidden_dim=4096,
        # --- Policy Network ---
        fusion_embed_dim: int = 4096,
        fusion_num_heads: int = 8,
        fusion_dropout: float = 0.1,
        transformer_num_layers: int = 6,
        transformer_num_heads: int = 8,
        transformer_dim_feedforward: int = 2048,
        transformer_dropout: float = 0.1,
        mlp_hidden_dims: list = [1024, 256],
        action_dim: int = 5,
        device = "cuda",
        subgoal_ongoing = "subgoal1",
    ):
        super().__init__()
        # 1) 定义语言编码器
        self.language_encoder = LanguageEncoder(
            hidden_dim=lang_hidden_dim,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            use_precomputed=use_precomputed_lang,
            pretrained_lang_dim=pretrained_lang_dim,
            device=device
        ).to(device)
        # 2) 定义视觉编码器，使用 SiglipVisionEncoder
        # 构建 args 对象，包含必要的属性
        class Args:
            def __init__(self, mm_feat, unfreeze):
                self.mm_vision_select_feature = mm_feat
                self.unfreeze_mm_vision_tower = unfreeze

        args = Args(mm_vision_select_feature, unfreeze_mm_vision_tower)

        self.vision_encoder = SiglipVisionEncoder(
            vision_tower_name=vision_tower_name,
            args=args,
            output_dim=vision_output_dim,
            delay_load=False,
            projection=True,
        ).to(device)
        self.vision_encoder_processor = AutoProcessor.from_pretrained(vision_tower_name)
        # vision encoder do not need training:
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.vision_encoder.eval()
            
        # 3) 定义机器人状态编码器
        self.state_encoder = RobotStateEncoder(state_dim=state_dim, hidden_dim=state_hidden_dim).to(device)
        self.subgoal_ongoing_discrete = subgoal_ongoing
        self.task_type = self.language_encoder.precomputed_lang_embedding[self.subgoal_ongoing_discrete]['instruction'].split(":")[0]
        self.lower_level_action_tensors = lower_level_action_tensors[self.task_type]

        # 4) 定义 Policy Network
        self.policy_network = PolicyNetwork_discrete(
            robot_state_dim=state_hidden_dim,
            vision_dim=vision_output_dim,
            language_dim=lang_hidden_dim,
            fusion_embed_dim=fusion_embed_dim,
            fusion_num_heads=fusion_num_heads,
            fusion_dropout=fusion_dropout,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_dim_feedforward=transformer_dim_feedforward,
            transformer_dropout=transformer_dropout,
            mlp_hidden_dims=mlp_hidden_dims,
            action_dim=action_dim,
        ).to(device)
        self.policy_network.lower_level_action_tensors = lower_level_action_tensors[self.task_type]
        

        self.image_std = [0.5, 0.5, 0.5]
        self.image_mean = [0.5, 0.5, 0.5]
        # try:
        #     if 'relative_pose' not in observation_space.spaces:
        #         self.observation_space['task::relative_pose'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        #         self.observation_space['task::in_fov'] = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  
        # except:
        #     pass

    def _build_mlp_extractor(self) -> None:
        pass

    def _build(self, lr_schedule) -> None:
        try:
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]
        except:
            pass

    # values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
    def evaluate_actions(self, observation: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation dictionary containing 'text', 'image', 'state'
        :param actions: Actions to evaluate, shape [B, action_dim]
        :return: estimated value, log likelihood of taking those actions, entropy of the action distribution
        """
        # 1. 编码多模态
        lang_feat, vision_feat, state_feat = self.feat_encoder(observation)

        # 2. 特征融合
        fused = self.policy_network.fusion([lang_feat, vision_feat, state_feat])  # [B, fusion_embed_dim]

        # 3. Transformer策略网络
        transformer_out = self.policy_network.transformer(fused)  # [B, fusion_embed_dim]

        # 4. MLP投射到动作空间（输出logits）
        action_logits = self.policy_network.action_proj(transformer_out)  # [B, action_dim]

        # 5. 应用任务掩码（Action Mask）
        # task_type = self.language_encoder.precomputed_lang_embedding[self.subgoal_ongoing_discrete]['instruction'].split(":")[0]
        # action_mask = Task_Action_Mask[task_type]
        # if action_mask is not None:
        #     action_logits = action_logits + (1 - action_mask.to(action_logits.device)) * -1e9  # 禁用无效动作

        # 6. 计算动作概率分布
        action_probs = torch.softmax(action_logits, dim=-1)  # [B, action_dim]

        # 7. 创建Categorical分布
        dist = torch.distributions.Categorical(probs=action_probs)

        # actions 是B*11的矩阵，每行是一个lower_level_action_tensors_nav的一个向量，找出是哪个向量，存下索引，变成Bx1的向量, 用于计算log_prob 请转换好device
        find_action_index = torch.tensor([torch.where(torch.all(torch.eq(torch.cat(self.lower_level_action_tensors, dim=0).to(action.device), action), dim=1))[0] for action in actions], device=actions.device)

        # 8. 计算 log_prob（针对离散动作）
        log_prob = dist.log_prob(find_action_index).unsqueeze(-1)  # [B, 1]

        # 9. 计算熵（鼓励策略多样性）
        entropy = dist.entropy().unsqueeze(-1)  # [B, 1]

        # 10. 计算值函数（状态价值估计）
        value = self.policy_network.value_head(transformer_out)  # [B, 1]

        return value, log_prob, entropy

    def feat_encoder(self, observation):
        # 1. 编码多模态
        lang_feat = self.language_encoder(self.subgoal_ongoing_discrete)    # [B, lang_hidden_dim]
        device = lang_feat.device

        # 如果视觉输入是 RGB 图像 为C, H, W的张量
        if observation['robot0::robot0:eyes:Camera:0::rgb'].shape[1] == 4:
            img_input = torch.as_tensor(observation['robot0::robot0:eyes:Camera:0::rgb'][:, :3, :, :], device=device)  # [B, 3, H, W]
        else:
            img_input = torch.as_tensor(observation['robot0::robot0:eyes:Camera:0::rgb'][:, :, :, :3], device=device).permute(0, 3, 1, 2)
        # normalize vision input
        normalize_transform = T.Normalize(mean=self.image_mean, std=self.image_std)
        img_input = img_input/255.0
        img_input = normalize_transform(img_input)

        # img_input = self.vision_encoder_processor.image_processor(img_input)['pixel_values'][0]

        vision_feat = self.vision_encoder(img_input)       # [B, vision_output_dim]

        # robot_state = torch.cat([torch.as_tensor(observation['task::low_dim'], device=device), torch.as_tensor(observation['task::relative_pose'], device=device), torch.as_tensor(observation['task::in_fov'], device=device)], 1)
        robot_state = torch.as_tensor(observation['task::low_dim'], device=device)
        state_feat  = self.state_encoder(robot_state)      # [B, state_hidden_dim]
        lang_feat = lang_feat.repeat(state_feat.size(0), 1)
        return lang_feat, vision_feat, state_feat

    
    def forward(self, observation):
        """
        Args:
            text_input (torch.Tensor): [B, seq_len] 或 [B, pretrained_lang_dim]
            img_input (torch.Tensor): [B, 3, H, W]
            robot_state (torch.Tensor): [B, state_dim]

        Returns:
            torch.Tensor: [B, action_dim]
        """
        task_type = self.language_encoder.precomputed_lang_embedding[self.subgoal_ongoing_discrete]['instruction'].split(":")[0]
        lang_feat, vision_feat, state_feat = self.feat_encoder(observation)
        # 2. 通过Policy Network生成动作
        action, action_prob = self.policy_network(state_feat, vision_feat, lang_feat, Task_Action_Mask[task_type])  # [B, action_dim]
        return action, action_prob
    
    def set_training_mode(self, mode):
        # 如果mode为True，则设置为训练模式，否则设置为评估模式
        if mode:
            self.policy_network.train()
            self.state_encoder.train()
            

    def predict_values(self, observation):
        # 1. 编码多模态
        lang_feat, vision_feat, state_feat = self.feat_encoder(observation)
        # 2. 通过Policy Network生成动作
        action, value, log_prob = self.policy_network(state_feat, vision_feat, lang_feat)  # [B, action_dim]

        return value
    
    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation: A dictionary containing multi-modal inputs:
                            - 'text': [B, seq_len] or [B, pretrained_lang_dim]
                            - 'image': [B, 3, H, W]
                            - 'state': [B, state_dim]
        :param deterministic: Whether to use deterministic (argmax) or stochastic (sampled) actions
        :return: Selected action, shape [B]
        """
        self.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 禁用梯度计算
            # 1. 编码多模态
            lang_feat, vision_feat, state_feat = self.feat_encoder(observation)

            # 2. 特征融合
            fused = self.policy_network.fusion([lang_feat, vision_feat, state_feat])  # [B, fusion_embed_dim]

            # 3. Transformer策略网络
            transformer_out = self.policy_network.transformer(fused)  # [B, fusion_embed_dim]

            # 4. MLP投射到动作空间（输出logits）
            action_logits = self.policy_network.action_proj(transformer_out)  # [B, action_dim]

            # 5. 应用任务掩码（Action Mask）
            # task_type = self.language_encoder.precomputed_lang_embedding[self.subgoal_ongoing_discrete]['instruction'].split(":")[0]
            # action_mask = Task_Action_Mask[task_type]
            # if action_mask is not None:
            #     action_logits = action_logits + (1 - action_mask.to(action_logits.device)) * -1e9  # 禁用无效动作

            # 6. 计算动作概率分布
            action_probs = torch.softmax(action_logits, dim=-1)  # [B, action_dim]

            # 7. 创建Categorical分布
            dist = torch.distributions.Categorical(probs=action_probs)

            # 8. 动作采样（根据deterministic选择确定性或随机动作）
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)  # 选择最大概率动作（确定性）
            else:
                action = dist.sample()  # 按概率随机采样（探索）

            return self.lower_level_action_tensors[action]  # [B]

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
        ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # 确定设备
        device = next(self.parameters()).device

        if isinstance(observation, np.ndarray):
            # 如果观察是单一的 NumPy 数组，需转换为字典
            # 根据实际需求调整此部分
            raise NotImplementedError("Single numpy array observations are not supported. Please provide a dictionary of observations.")
        elif isinstance(observation, dict):
            # 确保所有必需的键存在

            # 转换每个模态的观察为 torch.Tensor，并移动到设备

            # 调用 _predict 方法生成动作
            action = self._predict(observation, deterministic=deterministic)  # [B, action_dim]

            # 转换动作为 NumPy 数组
            action_np = action.cpu().numpy()

            # 由于当前策略网络不具备循环结构，下一步的隐藏状态为 None
            return action_np, None
        else:
            raise TypeError("Observation must be either a numpy array or a dictionary of numpy arrays.")

    def _get_value_ObjectsInFOVOfRobot(self, robot, env):
        """
        Gets all objects in the robot's field of view.

        Returns:
            list: List of objects in the robot's field of view
        """
        if not any(isinstance(sensor, VisionSensor) for sensor in robot.sensors.values()):
            raise ValueError("No vision sensors found on robot.")
        obj_names = []
        names_to_exclude = set(["background", "unlabelled"])
        for sensor in robot.sensors.values():
            if isinstance(sensor, VisionSensor):
                _, info = sensor.get_obs()
                obj_names.extend([name for name in info["seg_instance"].values() if name not in names_to_exclude])
        return [x for x in [env.scene.object_registry("name", x) for x in obj_names] if x is not None]
    
    def _get_target_object(self, env, obj_env_name):
        # if self.target_object is None:
        # self 是否包含 target_object 属性
        if not hasattr(self, "target_object") or self.target_object is None:
            self.target_object = env.scene.object_registry("name", obj_env_name)
        return self.target_object

def example_usage():
    # 初始化参数
    batch_size = 8
    seq_len = 10
    vocab_size = 30000
    pretrained_lang_dim = 1024   # 假设 LLaMA3-8B 的 embedding 维度
    state_dim = 15
    action_dim = 11

    # 初始化策略网络，使用预提取的语言特征
    policy = MultiModalRobotPolicy_Discrete(
        use_precomputed_lang=True,
        pretrained_lang_dim=pretrained_lang_dim,
        lang_hidden_dim=128,
        vision_tower_name="google/siglip-so400m-patch14-384",  # 替换为实际使用的 Siglip 模型名称
        mm_vision_select_feature="cls_patch",
        unfreeze_mm_vision_tower=False,
        vision_output_dim=256,
        state_dim=state_dim,
        state_hidden_dim=128,
        fusion_embed_dim=256,
        fusion_num_heads=8,
        fusion_dropout=0.1,
        transformer_num_layers=20,
        transformer_num_heads=8,
        transformer_dim_feedforward=512,
        transformer_dropout=0.1,
        mlp_hidden_dims=[512, 256],
        action_dim=action_dim,
    )

    # 生成假数据
    precomputed_embedding = torch.randn(batch_size, pretrained_lang_dim)  # [B, 1024]
    img_input = torch.rand(batch_size, 3, 384, 384)  # [B, 3, 224, 224]
    robot_state = torch.rand(batch_size, state_dim)   # [B, 15]

    # 前向推理
    action_out = policy(precomputed_embedding, img_input, robot_state)
    print("Action shape:", action_out.shape)  # [8, 11]

if __name__ == "__main__":
    example_usage()
