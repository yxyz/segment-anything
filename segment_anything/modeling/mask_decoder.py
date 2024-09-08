import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type

# 从自定义的 common 模块中导入 LayerNorm2d
from .common import LayerNorm2d

# 定义 MaskDecoder 类，继承自 nn.Module
class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,               # Transformer 的通道维度
        transformer: nn.Module,             # 用于预测掩码的 Transformer 模块
        num_multimask_outputs: int = 3,     # 多掩码输出的数量，用于掩码消歧时
        activation: Type[nn.Module] = nn.GELU, # 用于上采样掩码时的激活函数，默认为 GELU
        iou_head_depth: int = 3,            # 用于预测掩码质量的 MLP 的深度
        iou_head_hidden_dim: int = 256,     # MLP 中隐藏层的维度
    ) -> None:
        """
        给定图像和提示嵌入，使用 Transformer 架构预测掩码。

        参数：
          transformer_dim (int): Transformer 的通道维度
          transformer (nn.Module): 用于掩码预测的 Transformer 模块
          num_multimask_outputs (int): 当需要输出多个掩码时，生成的掩码数量
          activation (nn.Module): 用于上采样掩码时的激活函数
          iou_head_depth (int): 用于预测掩码质量的 MLP 的深度
          iou_head_hidden_dim (int): 用于预测掩码质量的 MLP 隐藏层的维度
        """
        super().__init__()
        self.transformer_dim = transformer_dim  # 存储 transformer 的通道维度
        self.transformer = transformer  # 存储 transformer 模块

        self.num_multimask_outputs = num_multimask_outputs  # 存储输出掩码的数量

        # 定义 IOU 预测所需的嵌入
        self.iou_token = nn.Embedding(1, transformer_dim)  # IOU 嵌入
        self.num_mask_tokens = num_multimask_outputs + 1  # 掩码 token 的数量，多输出时加 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)  # 掩码 token 嵌入

        # 定义输出的上采样操作
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),  # 反卷积进行上采样
            LayerNorm2d(transformer_dim // 4),  # LayerNorm2d 层归一化
            activation(),  # 激活函数
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),  # 继续上采样
            activation(),  # 激活函数
        )
        
        # 为每个掩码 token 定义一个超网络（MLP）用于生成掩码
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)  # 定义 MLP，深度为 3
                for i in range(self.num_mask_tokens)  # 每个掩码 token 对应一个 MLP
            ]
        )

        # 定义用于预测掩码质量的 MLP
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    # 前向传播函数，用于根据图像嵌入和提示嵌入预测掩码
    def forward(
        self,
        image_embeddings: torch.Tensor,  # 图像编码器生成的嵌入
        image_pe: torch.Tensor,          # 位置编码
        sparse_prompt_embeddings: torch.Tensor,  # 稀疏提示（点和框）嵌入
        dense_prompt_embeddings: torch.Tensor,   # 密集提示（掩码）嵌入
        multimask_output: bool,          # 是否输出多个掩码
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        给定图像和提示嵌入，预测掩码。

        参数：
          image_embeddings (torch.Tensor): 图像编码器生成的嵌入
          image_pe (torch.Tensor): 位置编码，形状与图像嵌入相同
          sparse_prompt_embeddings (torch.Tensor): 点和框提示的嵌入
          dense_prompt_embeddings (torch.Tensor): 掩码提示的嵌入
          multimask_output (bool): 是否返回多个掩码

        返回：
          torch.Tensor: 批量预测的掩码
          torch.Tensor: 批量预测的掩码质量
        """
        # 调用 predict_masks 函数进行掩码预测
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # 根据 multimask_output 参数选择是返回单个掩码还是多个掩码
        if multimask_output:
            mask_slice = slice(1, None)  # 如果多输出，从第一个开始取掩码
        else:
            mask_slice = slice(0, 1)  # 单输出时，只取第一个掩码
        masks = masks[:, mask_slice, :, :]  # 获取最终的掩码输出
        iou_pred = iou_pred[:, mask_slice]  # 获取相应的 IOU 预测

        # 返回掩码和 IOU 预测
        return masks, iou_pred

    # 掩码预测函数，调用 Transformer 并对输出进行处理
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,  # 图像嵌入
        image_pe: torch.Tensor,          # 位置编码
        sparse_prompt_embeddings: torch.Tensor,  # 稀疏提示嵌入
        dense_prompt_embeddings: torch.Tensor,   # 密集提示嵌入
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测掩码，详细信息参见 'forward' 函数。"""
        # 连接输出 token，包括 IOU token 和掩码 token
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # 扩展 output_tokens 以匹配 batch 大小
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        # 将 output_tokens 与稀疏提示嵌入连接
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 扩展图像嵌入，batch 维度与 token 数匹配
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings  # 加上密集提示嵌入
        # 位置编码也需要扩展
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape  # 获取 src 的 batch 大小和形状

        # 通过 Transformer 处理 src 和 tokens
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]  # IOU token 的输出
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]  # 掩码 token 的输出

        # 上采样掩码嵌入并通过掩码 token 预测掩码
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)  # 上采样嵌入
        hyper_in_list: List[torch.Tensor] = []  # 存储 MLP 的输出
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))  # 通过 MLP 处理掩码 token
        hyper_in = torch.stack(hyper_in_list, dim=1)  # 堆叠生成的掩码

        # 计算最终掩码
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # 生成掩码质量预测
        iou_pred = self.iou_prediction_head(iou_token_out)

        # 返回掩码和 IOU 预测
        return masks, iou_pred


# MLP 类，定义一个简单的多层感知机
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,       # 输入维度
        hidden_dim: int,      # 隐藏层维度
        output_dim: int,      # 输出维度
        num_layers: int,      # MLP 的层数
        sigmoid_output: bool = False,  # 是否对输出应用 sigmoid 激活
    ) -> None:
        super().__init__()
        self.num_layers = num_layers  # 存储层数
        h = [hidden_dim] * (num_layers - 1)  # 创建隐藏层的维度列表
        # 创建 MLP 层
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])  # 输入层到输出层
        )
        self.sigmoid_output = sigmoid_output  # 是否使用 sigmoid 激活函数

    def forward(self, x):
        # 遍历每一层并应用 ReLU 激活（除了最后一层）
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        # 如果需要，最后应用 sigmoid 激活
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
