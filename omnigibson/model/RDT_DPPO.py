import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModel, BertModel
# from methods.RoboticsDiffusionTransformer.models.rdt_runner import RDTRunner
import sys
sys.path.append('./')
sys.path.append('./methods/RoboticsDiffusionTransformer')
from methods.RoboticsDiffusionTransformer.scripts.agilex_model import create_model, RoboticDiffusionTransformerModel
from methods.RoboticsDiffusionTransformer.models.rdt.model import RDT
from methods.RoboticsDiffusionTransformer.models.hub_mixin import CompatiblePyTorchModelHubMixin
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler
import re
import torch.nn.functional as F
from methods.RoboticsDiffusionTransformer.models.rdt_runner import RDTRunner
import numpy as np


class RDTDPPOModel(nn.Module):
    def __init__(
        self,
        gamma_denoising: float,
        clip_ploss_coef: float,
        clip_ploss_coef_base: float = 1e-3,
        clip_ploss_coef_rate: float = 3,
        clip_vloss_coef: Optional[float] = None,
        clip_advantage_lower_quantile: float = 0,
        clip_advantage_upper_quantile: float = 1,
        norm_adv: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Whether to normalize advantages within batch
        self.norm_adv = norm_adv

        # Clipping value for policy loss
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate

        # Clipping value for value loss
        self.clip_vloss_coef = clip_vloss_coef

        # Discount factor for diffusion MDP
        self.gamma_denoising = gamma_denoising

        # Quantiles for clipping advantages
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile
        # 创建 RDTrayTrainner 模型
        self.rdt = RDTRunner.from_pretrained("methods/RoboticsDiffusionTransformer/ckpts/rdt-1b")
        self.language_tokens = torch.load("methods/RoboticsDiffusionTransformer/outs/language_embeding/clean_a_coffee_maker.pt")

    def forward(self, input_dict, state, seq_lens):
        device = self.rdt.device
        dtype = self.rdt.dtype
    
        # The background image used for padding
        background_color = np.array([
            int(x*255) for x in self.image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)
        background_image = np.ones((
            self.rdt.image_processor.size["height"], 
            self.rdt.image_processor.size["width"], 3), dtype=np.uint8
        ) * background_color

        lang_tokens = self.language_tokens
        images = input_dict['images']
        proprio = input_dict['proprio']


        # Preprocess the images by order and encode them
        image_tensor_list = []
        for image in images:
            if image is None:
                # Replace it with the background image
                image = Image.fromarray(background_image)
            
            if self.rdt.image_size is not None:
                image = transforms.Resize(self.rdt.data_args.image_size)(image)
            
            if self.rdt.args["dataset"].get("auto_adjust_image_brightness", False):
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75,1.75))(image)
                    
            if self.rdt.args["dataset"].get("image_aspect_ratio", "pad") == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in self.rdt.image_processor.image_mean))
            image = self.rdt.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor_list.append(image)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)

        image_embeds = self.rdt.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self.rdt.vision_model.hidden_size).unsqueeze(0)

        # Prepare the proprioception states and the control frequency
        joints = proprio.to(device).unsqueeze(0)   # (1, 1, 14)
        states, state_elem_mask = self.rdt._format_joint_to_state(joints)    # (1, 1, 128), (1, 128)
        states, state_elem_mask = states.to(device, dtype=dtype), state_elem_mask.to(device, dtype=dtype)
        states = states[:, -1:, :]  # (1, 1, 128)
        ctrl_freqs = torch.tensor([self.rdt.control_frequency]).to(device)
        
        text_embeds = text_embeds.to(device, dtype=dtype)
        
        # Predict the next action chunk given the inputs
        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(
                text_embeds.shape[:2], dtype=torch.bool,
                device=text_embeds.device),
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),  
            ctrl_freqs=ctrl_freqs
        )
        trajectory = self._unformat_action_to_joint(trajectory).to(torch.float32)

        return trajectory
