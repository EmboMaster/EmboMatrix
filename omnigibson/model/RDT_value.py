# write a RDTValueNet class to predict the value of a RDT test

# --------------------------------------------------------
# RDTValueNet
# --------------------------------------------------------
import torch
import torch.nn as nn
import re
from collections import OrderedDict
import logging

activation_dict = nn.ModuleDict(
    {
        "ReLU": nn.ReLU(),
        "ELU": nn.ELU(),
        "GELU": nn.GELU(),
        "Tanh": nn.Tanh(),
        "Mish": nn.Mish(),
        "Identity": nn.Identity(),
        "Softplus": nn.Softplus(),
    }
)

class MultiModalCriticWithGlobalMHSA(nn.Module):
    def __init__(
        self,
        input_dim=2048,   # 输入特征维度
        embed_dim=1024,   # MHSA 内部特征维度
        num_heads=8,      # MHSA 的头数
        mlp_dims=[512, 256],  # 输出 MLP 的维度
        dropout=0.1,      # Dropout 概率
        activation_type="ReLU",  # 激活函数
    ):
        super().__init__()

        # 多头自注意力模块
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        # 线性降维，用于将输入特征投影到 MHSA 的 embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # 输出网络（MLP）
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[1], 1),  # 输出标量
        )

        # 归一化层
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, lang_c, img_c, state_c):
        """
        lang_c: (B, L, 2048) 语言特征
        img_c: (B, I, 2048) 图像特征
        state_c: (B, S, 2048) 状态特征
        """
        B = lang_c.shape[0]

        # 拼接输入特征 (B, L+I+S, 2048)
        concat_features = torch.cat([lang_c, img_c, state_c], dim=1)

        # 投影到 MHSA 的特征空间 (B, L+I+S, embed_dim)
        concat_features = self.input_proj(concat_features)

        # 转换为 (L+I+S, B, embed_dim) 以适配 MultiheadAttention
        concat_features = concat_features.permute(1, 0, 2)

        # MHSA
        attn_output, _ = self.attention(concat_features, concat_features, concat_features)

        # 残差连接 + LayerNorm
        attn_output = self.layer_norm(attn_output + concat_features)

        # 转换回 (B, L+I+S, embed_dim)
        attn_output = attn_output.permute(1, 0, 2)

        # 池化（平均池化）生成全局特征 (B, embed_dim)
        global_feature = attn_output.mean(dim=1)

        # MLP 输出标量 (B, 1)
        value = self.fusion_mlp(global_feature)

        return value.squeeze(1)  # 返回标量


class MHSAEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        """
        embed_dim: 输入的特征维度 (例如 2048)
        num_heads: 自注意力头的数量
        dropout: 注意力机制中的dropout
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D), T 为序列长度
        """
        B, T, D = x.shape

        # Multihead Attention 需要 (T, B, D) 的输入
        x = x.permute(1, 0, 2)  # 转换为 (T, B, D)

        # 自注意力层 (T, B, D)
        attn_output, _ = self.attention(x, x, x)  # Q = K = V = x

        # 残差连接 + LayerNorm
        x = self.layer_norm(x + attn_output)

        # MLP + 残差连接
        x = x + self.mlp(x)

        # 转换回 (B, T, D)
        x = x.permute(1, 0, 2)

        # 池化以生成全局特征 (B, D)
        x = x.mean(dim=1)  # 如果需要更动态，可以考虑其他池化方式
        return x


class MLP(nn.Module):
    def __init__(
        self,
        dim_list,
        append_dim=0,
        append_layers=None,
        activation_type="Tanh",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        use_drop_final=False,
        verbose=False,
    ):
        super(MLP, self).__init__()

        # Construct module list: if use `Python List`, the modules are not
        # added to computation graph. Instead, we should use `nn.ModuleList()`.
        self.moduleList = nn.ModuleList()
        self.append_layers = append_layers
        num_layer = len(dim_list) - 1
        for idx in range(num_layer):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in append_layers:
                i_dim += append_dim
            linear_layer = nn.Linear(i_dim, o_dim)

            # Add module components
            layers = [("linear_1", linear_layer)]
            if use_layernorm and (idx < num_layer - 1 or use_layernorm_final):
                layers.append(("norm_1", nn.LayerNorm(o_dim)))
            if dropout > 0 and (idx < num_layer - 1 or use_drop_final):
                layers.append(("dropout_1", nn.Dropout(dropout)))

            # add activation function
            act = (
                activation_dict[activation_type]
                if idx != num_layer - 1
                else activation_dict[out_activation_type]
            )
            layers.append(("act_1", act))

            # re-construct module
            module = nn.Sequential(OrderedDict(layers))
            self.moduleList.append(module)
        if verbose:
            logging.info(self.moduleList)

    def forward(self, x, append=None):
        for layer_ind, m in enumerate(self.moduleList):
            if append is not None and layer_ind in self.append_layers:
                x = torch.cat((x, append), dim=-1)
            x = m(x)
        return x


class ResidualMLP(nn.Module):
    """
    Simple multi layer perceptron network with residual connections for
    benchmarking the performance of different networks. The resiudal layers
    are based on the IBC paper implementation, which uses 2 residual lalyers
    with pre-actication with or without dropout and normalization.
    """

    def __init__(
        self,
        dim_list,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
    ):
        super(ResidualMLP, self).__init__()
        hidden_dim = dim_list[1]
        num_hidden_layers = len(dim_list) - 3
        assert num_hidden_layers % 2 == 0
        self.layers = nn.ModuleList([nn.Linear(dim_list[0], hidden_dim)])
        self.layers.extend(
            [
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation_type=activation_type,
                    use_layernorm=use_layernorm,
                    dropout=dropout,
                )
                for _ in range(1, num_hidden_layers, 2)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, dim_list[-1]))
        if use_layernorm_final:
            self.layers.append(nn.LayerNorm(dim_list[-1]))
        self.layers.append(activation_dict[out_activation_type])

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x


class TwoLayerPreActivationResNetLinear(nn.Module):
    def __init__(
        self,
        hidden_dim,
        activation_type="Mish",
        use_layernorm=False,
        dropout=0,
    ):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = activation_dict[activation_type]
        if use_layernorm:
            self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-06)
            self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-06)
        if dropout > 0:
            raise NotImplementedError("Dropout not implemented for residual MLP!")

    def forward(self, x):
        x_input = x
        if hasattr(self, "norm1"):
            x = self.norm1(x)
        x = self.l1(self.act(x))
        if hasattr(self, "norm2"):
            x = self.norm2(x)
        x = self.l2(self.act(x))
        return x + x_input

class RDTValueNet(nn.Module):
    def __init__(self, hidden_size=2048, output_dim=1):
        super().__init__()
        self.valuenet = MultiModalCriticWithGlobalMHSA().to(torch.bfloat16)
        
        self.lang_adaptor = self.build_condition_adapter(
            "mlp2x_gelu", 
            in_features=4096, 
            out_features=hidden_size
        ).to(torch.bfloat16)
        self.img_adaptor = self.build_condition_adapter(
            "mlp2x_gelu", 
            in_features=1152, 
            out_features=hidden_size
        ).to(torch.bfloat16)
        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            "mlp3x_gelu", 
            in_features=128,    # state + state mask (indicator)
            out_features=hidden_size
        ).to(torch.bfloat16)
        
    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)
        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    def forward(self, text_embeds, image_embeds, states, state_elem_mask, ctrl_freqs):
        lang_c = self.lang_adaptor(text_embeds)        
        img_c = self.img_adaptor(image_embeds)
        state_c = self.state_adaptor(states)
        value = self.valuenet(lang_c, img_c, state_c)
        return value