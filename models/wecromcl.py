import math
import copy
import numpy as np
import torch
import torch.nn as nn
from models.modules.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import torch.nn.functional as F
from models.modules.components import ResidualAttentionBlock, Bottleneck, LayerNorm,  \
     conv3x3_bn_relu, conv1x1,OrderedDict,QuickGELU
from models.ops.modules import MSDeformAttn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(-2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class BiFPN(nn.Module):
    def __init__(self, cfgs, first_time):
        super(BiFPN, self).__init__()
        backbone = cfgs.model.backbone
        self.cfgs = cfgs

        if backbone in ['resnet18', 'resnet34']:
            nin = [64, 128, 256, 512]
        else:
            nin = [256, 512, 1024, 2048]

        ndim = 256

        self.first_time = first_time
        if self.first_time == 1:
            self.fpn_in5 = conv3x3_bn_relu(nin[-1], ndim)
            self.fpn_in4 = conv3x3_bn_relu(nin[-2], ndim)
            self.fpn_in3 = conv3x3_bn_relu(nin[-3], ndim)
            self.fpn_in2 = conv3x3_bn_relu(nin[-4], ndim)

        self.w_4_5 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_3_4 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_2_3 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_2up_3in_3up = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w_5in_5up = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_3up_4in_4up = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)

        self.c5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv4_up = conv1x1(ndim, 256)
        self.conv3_up = conv1x1(ndim, 256)
        self.conv2_up = conv1x1(ndim, 256)

        self.conv3_out = conv1x1(ndim, 256)
        self.conv4_out = conv1x1(ndim, 256)
        self.conv5_out = conv1x1(ndim, 256)
        if self.first_time == 1:
            self.fpn_in2.apply(self.weights_init)
            self.fpn_in3.apply(self.weights_init)
            self.fpn_in4.apply(self.weights_init)
            self.fpn_in5.apply(self.weights_init)
        self.conv4_up.apply(self.weights_init)
        self.conv3_up.apply(self.weights_init)
        self.conv2_up.apply(self.weights_init)
        self.conv3_out.apply(self.weights_init)
        self.conv4_out.apply(self.weights_init)
        self.conv5_out.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def swish(self, x):
        return x * x.sigmoid()

    def forward(self, inputs):
        if self.first_time:
            c2, c3, c4, c5 = inputs

            c5_in = self.fpn_in5(c5)
            c4_in = self.fpn_in4(c4)
            c3_in = self.fpn_in3(c3)
            c2_in = self.fpn_in2(c2)
        else:
            c2_in, c3_in, c4_in, c5_in = inputs

        weights = F.relu(self.w_4_5)
        norm_weights = weights / (weights.sum() + 0.0001)
        # 10
        c4_up = self.conv4_up(self.swish(norm_weights[0] * c4_in + norm_weights[1] * self.c5_upsample(c5_in)))
        weights = F.relu(self.w_3_4)
        norm_weights = weights / (weights.sum() + 0.0001)
        # 11
        c3_up = self.conv3_up(self.swish(norm_weights[0] * c3_in + norm_weights[1] * self.c4_upsample(c4_up)))
        weights = F.relu(self.w_2_3)
        norm_weights = weights / (weights.sum() + 0.0001)
        # 12
        c2_out = self.conv2_up(self.swish(norm_weights[0] * c2_in + norm_weights[1] * self.c3_upsample(c3_up)))
        weights = F.relu(self.w_2up_3in_3up)
        norm_weights = weights / (weights.sum() + 0.0001)
        # 13
        c3_out = self.conv3_out(self.swish(
            norm_weights[0] * c3_in + norm_weights[1] * c3_up + norm_weights[2] * F.max_pool2d(c2_out, kernel_size=3,
                                                                                               stride=2, padding=1)))
        weights = F.relu(self.w_3up_4in_4up)
        norm_weights = weights / (weights.sum() + 0.0001)
        # 14
        c4_out = self.conv4_out(self.swish(
            norm_weights[0] * c4_in + norm_weights[1] * c4_up + norm_weights[2] * F.max_pool2d(c3_out, kernel_size=3,
                                                                                               stride=2, padding=1)))
        weights = F.relu(self.w_5in_5up)
        norm_weights = weights / (weights.sum() + 0.0001)
        # 15
        c5_out = self.conv5_out(self.swish(
            norm_weights[0] * c5_in + norm_weights[1] * F.max_pool2d(c4_out, kernel_size=3,
                                                                     stride=2, padding=1)))

        return c2_out, c3_out, c4_out, c5_out



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        '''
        print(self.with_pos_embed(src, pos).shape,reference_points.shape)
        torch.Size([1, 36944, 256]) torch.Size([1, 36944, 4, 2])

        '''
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class PositionalEncoding2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale

    def forward(self, tensors, mask):
        x = tensors
        mask = mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos



class Feature_encoder(nn.Module):
    def __init__(self, deformable_layer):
        super(Feature_encoder, self).__init__()
        self.pos = PositionalEncoding2D(192 // 2, normalize=True)
        self.num_feature_levels = 2
        self.feature_strides = [4, 8]
        encoder_layer = DeformableTransformerEncoderLayer(d_model=192, d_ffn=256,
                                                          dropout=0.1, activation="relu",
                                                          n_levels=2, n_heads=8, n_points=4)
        self.encoder = DeformableTransformerEncoder(encoder_layer, deformable_layer)
        strides = [8, 16]
        num_channels = [192, 192]
        num_backbone_outs = len(strides)
        input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = num_channels[_]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, 192, kernel_size=1),
                nn.GroupNorm(32, 192),
            ))
        for _ in range(self.num_feature_levels - num_backbone_outs):
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, 192,
                          kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, 192),
            ))
            in_channels = 192
        self.input_proj = nn.ModuleList(input_proj_list)
        self.level_embed = nn.Parameter(torch.Tensor(2, 192))
        self._reset_parameters()

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                img_idx,
                : int(np.ceil(float(h) / self.feature_strides[idx])),
                : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embed)

    def forward(self, features, imgs):
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features],
            [imgs.shape[2:] for _ in range(len(imgs))],
            imgs.device,
        )
        pos_embeds = []
        srcs = []
        for l in range(len(features)):
            srcs.append(self.input_proj[l](features[l]))
            pos_embeds.append(self.pos(features[l], masks[l]).to(features[l].dtype))

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # [bs,num_pixels,channels]
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)


        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)

        encoded_memory = []

        encoded_memory.append(memory[:, 0:level_start_index[1], :].view(-1, spatial_shapes[0][0], spatial_shapes[0][1], 192).permute(0, 3,1, 2))
        encoded_memory.append(memory[:, level_start_index[1]:, :].view(-1, spatial_shapes[1][0], spatial_shapes[1][1], 192).permute(0, 3, 1, 2))
        return encoded_memory

    def inference(self, features, imgs):
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features],
            [imgs.shape[2:] for _ in range(len(imgs))],
            imgs.device,
        )
        pos_embeds = []
        srcs = []
        for l in range(len(features)):
            srcs.append(self.input_proj[l](features[l].to("cuda:1")).to("cuda:0"))
            pos_embeds.append(self.pos(features[l], masks[l]).to(features[l].dtype))

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # [bs,num_pixels,channels]
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)


        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)

        encoded_memory = []

        encoded_memory.append(memory[:, 0:level_start_index[1], :].view(-1, spatial_shapes[0][0], spatial_shapes[0][1], 192).permute(0, 3,1, 2))
        encoded_memory.append(memory[:, level_start_index[1]:, :].view(-1, spatial_shapes[1][0], spatial_shapes[1][1], 192).permute(0, 3, 1, 2))
        return encoded_memory
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
class AttentionPool2d_text_as_query(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                img_idx,
                : int(np.ceil(float(h) / self.feature_strides[idx])),
                : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks

    def forward(self, text_feature, x,mask=None):

        n, c, h, w = x.shape

        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC

        x, attn_map = F.multi_head_attention_forward(
            query=text_feature, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True
        )

        return x.squeeze(0), attn_map


class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            # self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.window_size = None
        self.relative_position_bias_table = None
        self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None):
        B, N, C = x.shape

        k = k.flatten(start_dim=2).permute(0,2,1)

        v = k
        if k is None:
            k = x
            v = x
            N_k = N
            N_v = N
        else:
            N_k = k.shape[1]
            N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)  # (B, N_q, dim)
        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)  # (B, N_k, dim)
        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)

        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, num_heads, N_q, dim)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, num_heads, N_k, dim)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, num_heads, N_v, dim)

        q = q * self.scale
        #print(q.shape, k.transpose(-2, -1).shape)
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N, N)

        attn = attn.softmax(dim=-1)
        #attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

class WeCromCL(nn.Module):
    def __init__(self, cfgs):
        super(WeCromCL, self).__init__()
        backbone = cfgs.model.backbone
        self.cfgs = cfgs
        self.context_length = cfgs.max_text_length
        self.use_stride = cfgs.model.use_stride
        self.train_input_size = cfgs.data.train.input_size
        self.backbone = eval(backbone)(pretrained=True)
        self.repeated_bifpn = nn.ModuleList([BiFPN(cfgs, first_time=1), BiFPN(cfgs, first_time=0)])
        ndim = 256
        self.fpn_out5_3 = nn.Sequential(
            conv3x3_bn_relu(ndim, 64),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))

        self.fpn_out4_3 = nn.Sequential(
            conv3x3_bn_relu(ndim, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

        self.fpn_out4_2 = nn.Sequential(
            conv3x3_bn_relu(ndim, 64),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))

        self.fpn_out3_3 = nn.Sequential(
            conv3x3_bn_relu(ndim, 64))

        self.fpn_out3_2 = nn.Sequential(
            conv3x3_bn_relu(ndim, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        self.fpn_out2_2 = conv3x3_bn_relu(ndim, 64)
        self.ln_final = LayerNorm(512)
        self.encoder = Feature_encoder(cfgs.model.deformable_layer)
        self.text_encoder = Transformer(
            width=cfgs.model.transformer_width,
            layers=cfgs.model.transformer_layers,
            heads=cfgs.model.transformer_heads,
            attn_mask=None
        )
        self.channel_to_512 = nn.Sequential(
            conv1x1(192, 512))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.token_embedding = nn.Embedding(cfgs.data.vocab_size + 1, cfgs.model.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, cfgs.model.transformer_width))
        self.text_projection = nn.Parameter(torch.empty(cfgs.model.transformer_width, cfgs.model.embed_dim))
        self.attn_pool = AttentionPool2d_text_as_query(512, 1, 512)
        self.fpn_out2_2.apply(self.weights_init)
        self.fpn_out3_2.apply(self.weights_init)
        self.fpn_out4_2.apply(self.weights_init)
        self.fpn_out4_3.apply(self.weights_init)
        self.fpn_out5_3.apply(self.weights_init)
        self.fpn_out3_3.apply(self.weights_init)
        self.initialize_parameters()
    @property
    def dtype(self):
        return self.backbone.conv1.weight.dtype
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.text_encoder.width ** -0.5) * ((2 * self.text_encoder.layers) ** -0.5)
        attn_std = self.text_encoder.width ** -0.5
        fc_std = (2 * self.text_encoder.width) ** -0.5
        for block in self.text_encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.text_encoder.width ** -0.5)

    def encode_image(self, imgs):
        feats = self.backbone(imgs)
        for bifpn in self.repeated_bifpn:
            feats = bifpn(feats)
        p5_3 = self.fpn_out5_3(feats[3])
        p4_3 = self.fpn_out4_3(feats[2])
        p3_3 = self.fpn_out3_3(feats[1])

        p4_2 = self.fpn_out4_2(feats[2])
        p3_2 = self.fpn_out3_2(feats[1])
        p2_2 = self.fpn_out2_2(feats[0])
        features = [torch.cat((p4_2, p3_2, p2_2), 1), torch.cat((p5_3, p4_3, p3_3), 1)]
        if self.cfgs.model.with_deformable_encoder:
            encoded_features = self.encoder(features, imgs)
            features = encoded_features
        if self.cfgs.model.use_stride == 8:
            return features[1]
        elif self.cfgs.model.use_stride == 4:
            return features[0]

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        bs, n_char, dim = x.shape
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x.reshape(bs * n_char, x.shape[-1]) @ self.text_projection
        x = x.reshape(bs, n_char, x.shape[-1]).mean(dim=1)
        return x

    
    def forward(self, image, random_text_features, false_text_features):
        random_text_features = random_text_features.tensor.to(image.device)
        false_text_features = false_text_features.to(image.device)
        image_conv_features = self.channel_to_512(self.encode_image(image))
        image_features = []
        for k in range(image.shape[0]):
            temp_text_features = torch.cat([random_text_features,false_text_features[k*self.cfgs.false_label_num:(k+1)*self.cfgs.false_label_num]],dim=0).unsqueeze(1)
            image_features.append(self.attn_pool(temp_text_features,image_conv_features[k].unsqueeze(0))[0])
        image_features = torch.cat(image_features,dim=1)
        return image_features.permute(1,0,2)

