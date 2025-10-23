import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModel, BertModel
# from methods.RoboticsDiffusionTransformer.models.rdt_runner import RDTRunner
import sys
sys.path.append('./')
sys.path.append('./methods/RoboticsDiffusionTransformer')
# from methods.RoboticsDiffusionTransformer.scripts.agilex_model import create_model, RoboticDiffusionTransformerModel
# from methods.RoboticsDiffusionTransformer.models.rdt.model import RDT
# from methods.RoboticsDiffusionTransformer.models.hub_mixin import CompatiblePyTorchModelHubMixin
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.schedulers.scheduling_dpmsolver_multistep import \
#     DPMSolverMultistepScheduler
import re
import torch.nn.functional as F
from methods.RoboticsDiffusionTransformer.models.rdt_runner import RDTRunner
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import yaml
from omnigibson.model.RDT_value import RDTValueNet
import torch.distributions as dist

class RDTModel(nn.Module):
    def __init__(self, device = 'cuda'):
        super().__init__()
        # 创建 RDTrayTrainner 模型
        # self.RDT = RDTRunner.from_pretrained("methods/RoboticsDiffusionTransformer/ckpts/rdt-1b")
        with open("methods/RoboticsDiffusionTransformer/configs/base.yaml", "r") as fp:
            self.RDTConfig = yaml.safe_load(fp)

        self.RDT = create_model(
            args=self.RDTConfig, 
            dtype=torch.bfloat16,
            # pretrained="methods/RoboticsDiffusionTransformer/ckpts/rdt-1b",
            pretrained="methods/RoboticsDiffusionTransformer/ckpts/rdt-170m",
            # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
            pretrained_vision_encoder_name_or_path="google/siglip-so400m-patch14-384",
            control_frequency=25,
            device=device
        )
        self.RDT.vision_model.eval()

        self.language_tokens = torch.load("methods/RoboticsDiffusionTransformer/outs/language_embeding/prepare_sea_salt_soak.pt")
        self.proprio = torch.tensor([0.0]*14).unsqueeze(0).unsqueeze(0).to(self.RDT.policy.model.blocks[0].attn.qkv.weight.device) # (1, 1, 14)
        self.action_mapping = {'base': torch.tensor([0, 1]), 'camera': torch.tensor([2, 3]), 'arm_0': torch.tensor([4, 5, 6, 7, 8, 9]), 'gripper_0': torch.tensor([10])}
        
        self.history_imgs = {}
        self.history_mode = 'rollout'
        

    def extractor(self, obs):
        
        device = self.RDT.policy.model.blocks[0].attn.qkv.weight.device
        dtype =  self.RDT.policy.model.blocks[0].attn.qkv.weight.dtype
        batch_size = obs['external::external_sensor0::depth'].shape[0]
    
        # The background image used for padding
        background_color = np.array([
            int(x*255) for x in self.RDT.image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)
        background_image = np.ones((
            self.RDT.image_processor.size["height"], 
            self.RDT.image_processor.size["width"], 3), dtype=np.uint8
        ) * background_color
        background_image = background_image.transpose(2, 0, 1)
        background_image = np.broadcast_to(background_image, (batch_size, *background_image.shape))
        background_image = torch.tensor(background_image).to(device, dtype=dtype)   

        lang_tokens = self.language_tokens['embeddings']
        
        if obs['external::external_sensor0::depth'].shape[0] > 1: # 应当是training模式
            if self.history_mode == 'rollout':
                self.history_imgs = {}
                self.history_mode = 'training'
        else: # 应当是rollout模式
            if self.history_mode == 'training':
                self.history_imgs = {}
                self.history_mode = 'rollout'
        
        image_keys_list = []        
        for key in obs.keys():
            if 'rgb' in key:
                image_keys_list.append(key)
        
        if len(self.history_imgs.keys()) == 0:
            for key in image_keys_list:
                self.history_imgs[key] = obs[key][:,:3,:,:]
                
            
        # images = [self.history_imgs[key] for key in self.history_imgs.keys()]
        if self.history_mode == 'rollout':
            images = [self.history_imgs[key] for key in self.history_imgs.keys()]
        elif self.history_mode == 'training':
            images = [obs[key][:,:3,:,:] for key in self.history_imgs.keys()]
            for i in range(len(image_keys_list)):
                images[i][0] = self.history_imgs[image_keys_list[i]][0]
                images[i][1:] = obs[image_keys_list[i]][:-1,:3,:,:].to(dtype)
        
        for key in image_keys_list:
            images.append(obs[key][:,:3,:,:])
        
        if len (images) <= 6:
            lack_camera_num = int((6 - len(images))/2)
            # 补齐全黑图片到6张
            # for l in range(lack_camera_num):
            #     for t in range(2):
            #         images.insert((3 - lack_camera_num) * t, background_image)
            new_images = images[:int(len(images)/2)]
            new_images += [background_image] * lack_camera_num
            new_images += images[int(len(images)/2):]
            new_images += [background_image] * lack_camera_num
            images = new_images


        image_tensor = self.preprocess_images(images, background_image).to(device, dtype=dtype)
        image_tensor_list = [image_tensor[i] for i in range(image_tensor.shape[0])]

        image_embeds = self.RDT.vision_model(image_tensor_list)
        image_embeds = torch.stack(image_embeds, dim = 0).detach()
        image_embeds = image_embeds.reshape(batch_size, -1, self.RDT.vision_model.hidden_size)

        # Prepare the proprioception states and the control frequency
        
        # proprio = obs['task::low_dim']
        proprio = obs['robot0::proprio']
        joints = proprio.to(device).unsqueeze(0)   # (1, 1, 13)
        states, state_elem_mask = self._format_joint_to_state(joints[:,:,[0,1,4,5,6,7,8,9,11]])
        # states, state_elem_mask = self.RDT._format_joint_to_state(joints)    # (1, 1, 128), (1, 128)
        states, state_elem_mask = states.to(device, dtype=dtype).repeat(batch_size, 1, 1), state_elem_mask.to(device, dtype=dtype).repeat(batch_size, 1)
        states = states[:, -1:, :]  # (1, 1, 128)
        ctrl_freqs = torch.tensor([self.RDT.control_frequency]).to(device).repeat(batch_size)
        
        text_embeds = lang_tokens.to(device, dtype=dtype).repeat(batch_size, 1, 1)
        
        if self.history_mode == 'rollout':
            for key in obs.keys():
                if 'rgb' in key:
                    self.history_imgs[key] = obs[key][-1,:3,:,:].unsqueeze(0)
        else:
            for key in obs.keys():
                if 'rgb' in key:
                    self.history_imgs[key] = obs[key][-1,:3,:,:].unsqueeze(0)
    
        # Predict the next action chunk given the inputs
        return text_embeds, image_embeds, states, state_elem_mask, ctrl_freqs

    def policy_net(self, text_embeds, image_embeds, states, state_elem_mask, ctrl_freqs):
        
        trajectorys, logprobs, variance = self.RDT.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(
                text_embeds.shape[:2], dtype=torch.bool,
                device=text_embeds.device),
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),  
            ctrl_freqs=ctrl_freqs
        )
        trajectorys = [self.RDT._unformat_action_to_joint(trajectory).to(torch.float32) for trajectory in trajectorys] 
        logprobs = [self.RDT._unformat_action_to_joint(logprob).to(torch.float32)[0].sum() for logprob in logprobs]

        return trajectorys, logprobs, variance
    
    def _format_joint_to_state(self, joints):
        """
        Format the joint proprioception into the unified action vector.

        Args:
            joints (torch.Tensor): The joint proprioception to be formatted. 
                qpos ([B, N, 14]).

        Returns:
            state (torch.Tensor): The formatted vector for RDT ([B, N, 128]). 
        """
        AGILEX_STATE_INDICES = [100,101,0,1,2,3,4,5,10]
        
        # Rescale the gripper to the range of [0, 1]
        # joints = joints / torch.tensor(
        #     [[[1, 1, 1, 1, 1, 1, 1, 1, 1]]],
        #     device=joints.device, dtype=joints.dtype
        # )
        
        B, N, _ = joints.shape
        state = torch.zeros(
            (B, N, self.RDT.args["model"]["state_token_dim"]), 
            device=joints.device, dtype=joints.dtype
        )
        # Fill into the unified state vector
        state[:, :, AGILEX_STATE_INDICES] = joints
        # Assemble the mask indicating each dimension's availability 
        state_elem_mask = torch.zeros(
            (B, self.RDT.args["model"]["state_token_dim"]),
            device=joints.device, dtype=joints.dtype
        )
        state_elem_mask[:, AGILEX_STATE_INDICES] = 1
        return state, state_elem_mask
    

    # def fetch_proprio_project(self, proprio):
    #     # get 14 dimmension proprio
        
    #     return proprio  
    
        
    def preprocess_images(self, image_batches, background_image_tensor):
        """
        批量处理图像列表，支持扩展到 batch 维度。

        Args:
            image_batches: 一个形状为 (batch_size, h, w, 3) 的 numpy 数组列表。
            background_image: 用于填充 None 图像的背景图片，形状为 (h, w, 3) 的 numpy 数组。
            RDT: 一个包含预处理配置的对象。

        Returns:
            torch.Tensor: 形状为 (batch_size, 3, h, w) 的图像张量。
        """
        processed_image_list = []
        
        for batch in image_batches:  # 遍历每个 batch
            # 如果图像有 None 值，替换为背景图像
            batch = torch.where(
                batch.isnan(),  # 假设 None 值以 NaN 表示
                background_image_tensor,  # 将背景图扩展为 (1, 3, h, w)
                batch
            )

            # 调整图像大小
            if self.RDT.image_size is not None:
                resize_transform = transforms.Resize(self.RDT.data_args.image_size)
                batch = torch.stack([resize_transform(image) for image in batch])

            # 自动调整亮度
            if self.RDT.args["dataset"].get("auto_adjust_image_brightness", False):
                # 计算每张图像的平均亮度
                average_brightness = batch.mean(dim=(1, 2, 3), keepdim=True)  # (batch_size, 1, 1, 1)
                low_brightness_mask = (average_brightness <= 0.15)
                if low_brightness_mask.any():
                    brightness_factor = 1.75
                    batch[low_brightness_mask.squeeze()] *= brightness_factor
                    batch = batch.clamp(0, 1)  # 防止超出范围

            # 填充或调整图像的宽高比
            if self.RDT.args["dataset"].get("image_aspect_ratio", "pad") == 'pad':
                _, _, h, w = batch.shape
                max_dim = max(h, w)
                padded_batch = torch.zeros(batch.size(0), 3, max_dim, max_dim, device=batch.device)
                padded_batch += torch.tensor(
                    self.RDT.image_processor.image_mean,
                    dtype=batch.dtype,
                    device=batch.device
                ).view(1, 3, 1, 1)  # 设置填充颜色
                if h > w:
                    padded_batch[:, :, :, (h - w) // 2:(h - w) // 2 + w] = batch
                else:
                    padded_batch[:, :, (w - h) // 2:(w - h) // 2 + h, :] = batch
                batch = padded_batch

            # 使用 image_processor 进行最终预处理
            batch = self.RDT.image_processor.preprocess(batch, return_tensors='pt')['pixel_values']
            processed_image_list.append(batch)
        # 合并所有 batch 的张量为一个整体
        return torch.stack(processed_image_list, dim=1)
        # return processed_image_list

class CustomRDTPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        if 'device' in kwargs:
            device = kwargs['device']
            # 把device弹出kwargs
            kwargs.pop('device')
        else:
            device = 'cuda'
        super(CustomRDTPolicy, self).__init__(*args, **kwargs)
        self.RDTPolicy = RDTModel(device=device)
        # self.RDTPolicy.extractor = self.RDTPolicy.extractor
        # self.RDTPolicy.policy_net = self.RDTPolicy.policy_net
        self.value_net = RDTValueNet().to(device)
        # def forward(self, obs, deterministic=False):
        #     # 使用 Diffusion Policy 生成动作
        #     actions = self.RDTPolicy(obs)
            
        #     return actions, values, log_prob
    def forward(self, obs: torch.Tensor, deterministic: bool = False, method='PPO') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        with torch.no_grad():
            text_embeds, image_embeds, states, state_elem_mask, ctrl_freqs = self.RDTPolicy.extractor(obs)
        trajectorys, log_probs, variance = self.RDTPolicy.policy_net(text_embeds, image_embeds, states, state_elem_mask, ctrl_freqs)
        values = self.value_net(text_embeds, image_embeds, states, state_elem_mask, ctrl_freqs).to(torch.float32)
        actions = [trajectory[:, 0] for trajectory in trajectorys]
        # 9->11 add camera 'base': tensor([0, 1]), 'camera': tensor([2, 3]), 'arm_0': tensor([4, 5, 6, 7, 8, 9]), 'gripper_0': tensor([10])
        # insert 2,3 as zero to actions
        actions_11dims = [torch.zeros((actions[0].shape[0], 11)).to(actions[0].device) for _ in actions ]
        for i in range(len(trajectorys)):
            actions_11dims[i][:, [0, 1, 4, 5, 6, 7, 8, 9, 10]] = actions[i]
        if method == 'DPPO':
            return actions_11dims, values, log_probs[-1]
        else:
            return actions_11dims[-1], values, log_probs[-1]
        
        
    def predict_values(self, obs):
        text_embeds, image_embeds, states, state_elem_mask, ctrl_freqs = self.RDTPolicy.extractor(obs)
        values = self.value_net(text_embeds, image_embeds, states, state_elem_mask, ctrl_freqs).to(torch.float32)
        return values
    
    def evaluate_actions(self, obs, actions_raw, method = 'PPO'):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        text_embeds, image_embeds, states, state_elem_mask, ctrl_freqs = self.RDTPolicy.extractor(obs)
        trajectorys, log_probs, variance = self.RDTPolicy.policy_net(text_embeds, image_embeds, states, state_elem_mask, ctrl_freqs)
        values = self.value_net(text_embeds, image_embeds, states, state_elem_mask, ctrl_freqs).to(torch.float32)
        actions = [trajectory[:, 0] for trajectory in trajectorys]
        # 9->11 add camera 'base': tensor([0, 1]), 'camera': tensor([2, 3]), 'arm_0': tensor([4, 5, 6, 7, 8, 9]), 'gripper_0': tensor([10])
        # insert 2,3 as zero to actions
        # actions_11dims = [torch.zeros((actions[0].shape[0], 11)).to(actions[0].device) for _ in actions ]
        variance_t = variance[-1]
        # variance -> NxN matrix, 对角线元素为variance_t
        # variance_vector = torch.zeros((actions[-1].shape[-1])).to(actions[-1].device)
        variance_vector = torch.tensor([variance_t for _ in range(actions[-1].shape[-1])]).unsqueeze(0).repeat(actions[-1].shape[0],1).to(actions[-1].device)
        # for i in range(variance_matrix.shape[0]):
            # variance_matrix[i][i] = variance_t
        actions_raw_9dim = torch.zeros_like(actions[-1])
        actions_raw_9dim = actions_raw[:, [0, 1, 4, 5, 6, 7, 8, 9, 10]].to(actions[-1].device)
        log_prob, entropy = self.compute_logprob_and_entropy(actions[-1], actions_raw_9dim, variance_vector)
        # log 算清楚variance是什么东西
    
        if method == 'DPPO':
            return values, log_prob, entropy
        else:
            return values, log_prob, entropy
        
        
    def compute_logprob_and_entropy(self, actions, actions_raw, variance):
        """
        Calculate the log probability of actions_raw under a Gaussian distribution 
        defined by mean (actions) and variance, and compute the entropy of the distribution.
        
        :param actions: Mean actions, tensor of shape (batch_size, action_dim)
        :param actions_raw: Actual actions taken, tensor of shape (batch_size, action_dim)
        :param variance: Variance of the Gaussian distribution, could be a scalar or tensor
        :return: log_prob, entropy
        """
        # Assuming variance is diagonal, i.e., we have independent Gaussian for each action dimension
        std = torch.sqrt(variance)
        
        # Define a Gaussian distribution with given mean and standard deviation
        gaussian_dist = dist.Normal(actions, std)

        # Calculate log probability of actions_raw
        log_prob = gaussian_dist.log_prob(actions_raw)  # shape: (batch_size, action_dim)
        log_prob = log_prob.mean(dim=-1)  # Sum across action dimensions, shape: (batch_size,)

        # Calculate entropy of the Gaussian distribution
        entropy = gaussian_dist.entropy()  # shape: (batch_size, action_dim)
        entropy = entropy.mean(dim=-1)  # Sum across action dimensions, shape: (batch_size,)

        return log_prob, entropy
    
    def _predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Predict actions based on the observation.

        :param obs: Observation tensor
        :param deterministic: If True, select the mean action (deterministic); 
                            otherwise, sample from the action distribution.
        :return: Predicted actions (tensor)
        """
        # Extract features using the RDTPolicy extractor
        self.eval()  # 将模型切换到评估模式
        with torch.no_grad(): 
            text_embeds, image_embeds, states, state_elem_mask, ctrl_freqs = self.RDTPolicy.extractor(obs)
        trajectorys, log_probs, variance = self.RDTPolicy.policy_net(text_embeds, image_embeds, states, state_elem_mask, ctrl_freqs)
        values = self.value_net(text_embeds, image_embeds, states, state_elem_mask, ctrl_freqs).to(torch.float32)
        actions = [trajectory[:, 0] for trajectory in trajectorys]
        
        # Add camera-related zero-dimensions to actions to make them 11-dimensional
        actions_11dims = [torch.zeros((actions[0].shape[0], 11)).to(actions[0].device) for _ in actions ]
        for i in range(len(trajectorys)):
            actions_11dims[i][:, [0, 1, 4, 5, 6, 7, 8, 9, 10]] = actions[i]
        
        # If deterministic, return the mean actions directly (now only determinstic)
        return actions_11dims[-1]