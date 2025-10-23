# siglip_vision_encoder.py

import torch
import torch.nn as nn
from transformers import AutoConfig, SiglipImageProcessor, SiglipVisionModel

class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, `load_model` called again, skipping.')
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.eval()

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        if self.select_feature == 'patch':
            image_features = image_forward_outs.last_hidden_state  # (B, 729, 1536)
        elif self.select_feature == 'cls_patch':
            image_features = image_forward_outs.pooler_output  # (B, 1, 1536)
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
        

class SiglipVisionEncoder(nn.Module):
    """
    使用 SiglipVisionTower 作为视觉编码器，并添加一个线性层将特征映射到固定维度。
    """
    def __init__(
        self,
        vision_tower_name: str,
        args,
        output_dim: int = 256,
        delay_load: bool = False,
        projection: bool = True,  # 是否添加投影层
    ):
        """
        Args:
            vision_tower_name (str): SiglipVisionTower 使用的模型名称
            args: 包含 `mm_vision_select_feature` 和 `unfreeze_mm_vision_tower` 等参数的对象
            output_dim (int): 最终输出的特征维度
            delay_load (bool): 是否延迟加载模型
            projection (bool): 是否添加投影层
        """
        super().__init__()
        self.siglip_tower = SiglipVisionTower(vision_tower=vision_tower_name, args=args, delay_load=delay_load)

        hidden_size = self.siglip_tower.hidden_size  # e.g., 1536

        if projection:
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, output_dim),
                nn.ReLU(),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): [B, 3, H, W] 的图像张量
        Returns:
            torch.Tensor: [B, output_dim] 的视觉特征
        """
        # 使用 SiglipVisionTower 提取特征
        vision_features = self.siglip_tower(images)  # 根据 select_feature, e.g., [B, 1, 1536] 或 [B, 729, 1536]

        # 如果特征包含序列长度维度（如 [B, 1, D] 或 [B, N, D]），通常取 CLS token 或做池化
        if vision_features.ndim == 3:
            # 假设 'cls_patch' 被选中，shape 为 [B, 1, D]
            vision_features = vision_features.squeeze(1)  # [B, D]
        elif vision_features.ndim == 2:
            # 已经是 [B, D] 形状
            pass
        else:
            raise ValueError(f'Unexpected vision_features shape: {vision_features.shape}')

        # 投影到 output_dim
        vision_embed = self.projection(vision_features)  # [B, output_dim]

        return vision_embed
