import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from torch.distributions import Categorical



class MLPProjection(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        多层感知机用于将特征映射到动作空间。

        Args:
            input_dim (int): 输入特征维度。
            hidden_dims (list of int): 隐藏层的维度列表。
            output_dim (int): 输出动作空间的维度。
        """
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [B, input_dim]

        Returns:
            softmaxed_action (torch.Tensor): [B, output_dim]
        """
        return self.mlp(x)


class TransformerPolicy(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, dim_feedforward=512, dropout=0.1):
        """
        基于Transformer的策略网络。

        Args:
            embed_dim (int): 输入和输出的嵌入维度。
            num_layers (int): Transformer编码器层的数量。
            num_heads (int): 注意力头的数量。
            dim_feedforward (int): 前馈网络的维度。
            dropout (float): Dropout率。
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [B, embed_dim]

        Returns:
            torch.Tensor: [B, embed_dim]
        """
        # Transformer编码器期望输入形状为 [seq_len, B, embed_dim]，因此需要调整
        x = x.unsqueeze(0)  # [1, B, embed_dim]

        # 应用Transformer编码器
        transformer_out = self.transformer_encoder(x)  # [1, B, embed_dim]

        # 恢复到 [B, embed_dim]
        transformer_out = transformer_out.squeeze(0)  # [B, embed_dim]

        # 层归一化
        out = self.layer_norm(transformer_out)

        return out


class PolicyNetwork_discrete(nn.Module):
    def __init__(
        self,
        robot_state_dim: int,
        vision_dim: int,
        language_dim: int,
        fusion_embed_dim: int = 256,
        fusion_num_heads: int = 8,
        fusion_dropout: float = 0.1,
        transformer_num_layers: int = 6,
        transformer_num_heads: int = 8,
        transformer_dim_feedforward: int = 512,
        transformer_dropout: float = 0.1,
        mlp_hidden_dims: list = [512, 256],
        action_dim: int = 5,
    ):
        """
        Args:
            robot_state_dim (int): 机器人状态特征的维度。
            vision_dim (int): 视觉特征的维度。
            language_dim (int): 语言特征的维度。
            fusion_embed_dim (int): 融合后特征的嵌入维度。
            fusion_num_heads (int): 融合MHA的注意力头数量。
            fusion_dropout (float): 融合MHA的Dropout率。
            transformer_num_layers (int): Transformer编码器层的数量。
            transformer_num_heads (int): Transformer编码器中的注意力头数量。
            transformer_dim_feedforward (int): Transformer编码器中前馈网络的维度。
            transformer_dropout (float): Transformer编码器的Dropout率。
            mlp_hidden_dims (list): MLP的隐藏层维度列表。
            action_dim (int): 最终动作空间的维度。
        """
        super().__init__()
        # 1. 特征融合
        self.fusion = FusionMHA(
            input_dims=[language_dim, vision_dim, robot_state_dim],
            embed_dim=fusion_embed_dim,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
        )
        # 2. 基于Transformer的策略网络
        self.transformer = TransformerPolicy(
            embed_dim=fusion_embed_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
        )
        # 3. MLP投射到动作空间
        self.action_proj = MLPProjection(
            input_dim=fusion_embed_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=action_dim,
        )
        # 4. 值函数头
        self.value_head = nn.Linear(fusion_embed_dim, 1)

    def forward(self, robot_state, vision_feat, language_feat, action_mask=None):
        """
        Args:
            robot_state (torch.Tensor): [B, robot_state_dim]
            vision_feat (torch.Tensor): [B, vision_dim]
            language_feat (torch.Tensor): [B, language_dim]

        Returns:
            dict: 包含 'action', 'value', 'log_prob'
        """
        # 1. 特征融合
        fused = self.fusion([language_feat, vision_feat, robot_state])  # [B, fusion_embed_dim]
        # 2. Transformer策略网络
        transformer_out = self.transformer(fused)  # [B, fusion_embed_dim]
        # 3. MLP投射到动作空间
        action_logits = self.action_proj(transformer_out)  # [B, action_dim]
        action_probs = F.softmax(action_logits, dim=-1)  # [B, action_dim]

        dist = Categorical(probs=action_probs)
        
        # 6. 采样动作
        action_index = dist.sample()  # [B, action_dim] 使用rsample()以便支持reparameterization trick
        action = self.lower_level_action_tensors[action_index]
        print(f"take action {action}")
        # 7. 计算 log_prob
        log_prob = dist.log_prob(action).unsqueeze(-1)  # [B, 1]
        entropy = dist.entropy().unsqueeze(-1)  # [B, 1]

        # 9. 值函数计算
        value = self.value_head(transformer_out)  # [B, 1]

        # return self.lower_level_action_tensors[action], action_probs
        return action, value, log_prob
    
class FusionMHA(nn.Module):
    def __init__(self, input_dims, embed_dim, num_heads, dropout=0.1):
        """
        使用多头注意力机制融合多个输入特征。

        Args:
            input_dims (list of int): 每个输入特征的维度列表。
            embed_dim (int): 融合后特征的嵌入维度。
            num_heads (int): 注意力头的数量。
            dropout (float): Dropout率。
        """
        super().__init__()
        self.num_features = len(input_dims)
        self.embed_dim = embed_dim

        # 将每个输入特征投影到相同的嵌入维度
        self.projections = nn.ModuleList([
            nn.Linear(dim, embed_dim) for dim in input_dims
        ])

        # 多头注意力机制
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # 层归一化
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, features):
        """
        Args:
            features (list of torch.Tensor): 特征张量列表，每个形状为 [B, D_i]
        
        Returns:
            torch.Tensor: 融合后的特征张量，形状为 [B, embed_dim]
        """
        # 投影每个特征到嵌入维度
        projected_features = [proj(feat) for proj, feat in zip(self.projections, features)]  # list of [B, embed_dim]

        # 将特征堆叠为序列
        x = torch.stack(projected_features, dim=1)  # [B, num_features, embed_dim]

        # 应用多头注意力
        attn_output, _ = self.mha(x, x, x)  # [B, num_features, embed_dim]

        # 残差连接和层归一化
        x = self.layer_norm(attn_output + x)  # [B, num_features, embed_dim]

        # 聚合序列为单一向量，例如通过平均池化
        fused = x.mean(dim=1)  # [B, embed_dim]

        return fused
