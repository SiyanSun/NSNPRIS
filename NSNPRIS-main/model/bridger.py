import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional

from .layers import conv_layer, deconv_layer, SNPconv_layer, SNPdeconv_layer, InteractorT, Interactor


from model.attn import bilateral_prompt


class Bridger_RN(nn.Module):
    def __init__(self,
                 d_img = [512, 1024, 2048],
                 d_txt = 512,
                 d_model = 64,
                 d_p = 1024,
                 nhead = 8,
                 num_stages = 3,
                 strides = [2, 1, 2],
                 num_layers = 12,
                 fusion_stage = 3,
                ):
        super().__init__()
        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.d_p = d_p
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.fusion_stage = fusion_stage


        self.fusion_v, self.fusion_t = nn.ModuleList(), nn.ModuleList()
        #Zoom Layer通常用于图像处理中的上采样和下采样操作。上采样是指将图像放大，增加图像的尺寸和细节；下采样是指将图像缩小，减少图像的尺寸和细节。
        # Zoom Layer可以通过插值或卷积等方式实现图像的上下采样操作，常用于图像处理和计算机视觉任务中。
        # Linear层则是神经网络中常用的一种全连接层，也称为仿射层。它将输入的每个特征都与权重相乘并加上偏置，从而实现输入特征的线性变换。
        # Linear层在神经网络中起到了特征提取和映射的作用，是神经网络中最基础和常用的一种层类型。
        self.zoom_in = nn.ModuleList()
        self.zoom_out = nn.ModuleList()
        self.linear1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.ln_v = nn.ModuleList()
        self.ln_t = nn.ModuleList()
        for i in range(num_stages):
            if i >= num_stages - fusion_stage:
                self.fusion_v.append(Interactor(d_model=d_model, nhead=nhead))
                self.fusion_t.append(Interactor(d_model=d_model, nhead=nhead))
                if i < num_stages - 1:
                    self.zoom_in.append(nn.Conv2d(d_img[i], d_model, kernel_size=strides[i], stride=strides[i], bias=False))
                    #self.vis_project = nn.Conv2d(d_img[i], d_p, kernel_size=1, bias=True)
                    self.zoom_out.append(nn.ConvTranspose2d(d_model, d_img[i], kernel_size=strides[i], stride=strides[i], bias=False))
                    self.linear1.append(nn.Linear(d_txt, d_model))
                    self.linear2.append(nn.Linear(d_model, d_txt))
                    if fusion_stage > 1:
                        self.ln_v.append(nn.LayerNorm(d_model))
                        self.ln_t.append(nn.LayerNorm(d_model))
                else:
                    self.zoom_in.append(nn.ConvTranspose2d(d_img[i], d_model,kernel_size=strides[i], stride=strides[i], bias=False))
                    #self.vis_project = nn.Conv2d(d_img[i], d_p, kernel_size=1, bias=True)
                    self.zoom_out.append(nn.Conv2d(d_model, d_img[i], kernel_size=strides[i], stride=strides[i], bias=False))
                    self.linear1.append(nn.Linear(d_txt, d_model))
                    self.linear2.append(nn.Linear(d_model, d_txt))
            else:
                self.fusion_v.append(None)
                self.fusion_t.append(None)
                self.zoom_in.append(None)
                self.zoom_out.append(None)
                self.linear1.append(None)
                self.linear2.append(None)
                self.ln_v.append(None)
                self.ln_t.append(None)
        self.initialize_parameters()

        self.attn_fusion = bilateral_prompt(d_p,d_p)
        self.lan_project = nn.Linear(self.d_txt, d_p)
        #self.vis_project = nn.Conv2d(d_img[2],d_p,1)
        self.vis_project = nn.Conv2d(512,1024,1)
        self.lan_projectout = nn.Linear(1024,512)
        self.vis_projectout = nn.Conv2d(1024, 512, 1)

        self.res_gate = nn.Sequential(
            #nn.ReLU(),
            nn.Linear(d_txt, d_model, bias=False),
            #nn.ReLU(),
            nn.ELU(),
            #nn.Tanh(),
            nn.Linear(d_model, d_txt, bias=False),
            #nn.Tanh()
            nn.Sigmoid()
        )






    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, vis, text, backbone):
        # 可改：改成SNPbridger.pybridger.py
        def stem(x):    #用于对输入进行一系列的卷积操作和池化操作，以提取图像特征。
            for conv, bn in [(vis_enc.conv1, vis_enc.bn1), (vis_enc.conv2, vis_enc.bn2),
                             (vis_enc.conv3, vis_enc.bn3)]:
                #x = bn(conv(vis_enc.relu(x)))
                x = vis_enc.relu(bn(conv(x)))

            x = vis_enc.avgpool(x)
            return x
        # vision
        vis_enc = backbone.visual
        vis = vis.type(vis_enc.conv1.weight.dtype)
        vis = stem(vis)
        vis = vis_enc.layer1(vis)
        vis_enc_layers = [vis_enc.layer2, vis_enc.layer3, vis_enc.layer4]

        # language
        txt = backbone.token_embedding(text).type(
            backbone.dtype)  # [batch_size, n_ctx, d_model]

        txt_enc = backbone.transformer
        txt = txt + backbone.positional_embedding.type(backbone.dtype)[:txt.size(1)]
        txt = txt.permute(1, 0, 2)  # NLD -> LND

        # fusion
        stage_i = 0
        vis_outs = []
        for i in range(self.num_layers):
            if (i+1)%4 != 0:
                txt = txt_enc.resblocks[i](txt)
            else:
                # feed into this layer
                txt = txt_enc.resblocks[i](txt)
                vis = vis_enc_layers[stage_i](vis)
                if stage_i >= self.num_stages - self.fusion_stage:
                    # residual operation
                    v = vis.clone() #通过clone方法创建的副本是原始对象的独立副本，对副本的修改不会影响原始对象。
                    t = txt.clone()
                    # dimension reduction
                    v = self.zoom_in[stage_i](v)
                    t = self.linear1[stage_i](t)
                    # multi modal fusion
                    B, C, H, W = v.shape
                    #v.reshape(B, C, -1): 这个函数调整张量的形状，其中B是批量大小，C是通道数，-1表示自动计算剩余的维度，这里是将原始形状为B x C x H x W的张量调整为B x C x (H*W)的形状。
                    #permute(2, 0, 1): 这个函数对张量的维度进行重新排列，这里是将原始形状为B x C x (H*W)的张量调整为(H*W) x B x C的形状。
                    #综合起来，这行代码的作用是将原始形状为B x C x H x W的张量v转换为(H*W) x B x C的形状。
                    v = v.reshape(B, C, -1).permute(2, 0, 1) # B, C, H, W -> B, C, HW -> HW, B, C(676, 64, 256)
                    if self.fusion_stage > 1 and stage_i > self.num_stages - self.fusion_stage:
                        v, t = self.ln_v[stage_i-1](v+last_v), self.ln_t[stage_i-1](t+last_t)
                    last_v, last_t = v, t
                    v, t = self.fusion_v[stage_i](v, t), self.fusion_t[stage_i](t, v)
                    v = v.permute(1, 2, 0).reshape(B, -1, H, W) # HW, B, C -> B, C, HW -> B, C, H, W
                    # dimension recovery
                    v = self.zoom_out[stage_i](v)
                    t = self.linear2[stage_i](t)

                    



                    # residual connect
                    vis = vis + v
                    txt = txt + t

                    #改，gate加
                    #vis_res = vis
                    #vis = vis + (self.res_gate(vis_res)*vis_res)
                    txt_gate = txt
                    txt_gate = self.res_gate(txt_gate) * txt_gate

                    txt = txt + txt_gate




                stage_i += 1
                if stage_i < self.num_stages:
                    vis_outs.append(vis)
        # After fusion
        vis = vis_enc.attnpool(vis)
        vis_outs.append(vis)

        # language
        txt = txt.permute(1, 0, 2)  # LND -> NLD
        txt = backbone.ln_final(txt).type(backbone.dtype)

        lan = txt
        vp = vis
        #vp = vis.to(torch.half)
        B, _, H, _ = vp.size()
        lan = self.lan_project(lan)
        vp = self.vis_project(vp)



        h_, w_ = vp.shape[2:]
        vis_trans = vp.flatten(2).transpose(1, 2)
        #lan = lan.unsqueeze(0).repeat(B, 1, 1)

        norm_vis = vis_trans / vis_trans.norm(dim=-1, keepdim=True)
        norm_vis = norm_vis

        norm_lan = lan / lan.norm(dim=-1, keepdim=True)



        new_vis, new_lan = self.attn_fusion(norm_vis.permute(0, 2, 1).reshape(B, -1, h_, w_),
                                            norm_lan.transpose(1, 2))

       # print(new_lan.shape)
        #print(txt.shape)

        new_lan = self.lan_projectout(new_lan)
      #  print(new_vis.shape)
       # print(vis_outs[2].shape)
        new_vis = self.vis_projectout(new_vis)

        vis_outs[2] = new_vis + vis_outs[2]
        #vis_outs.append(new_vis)
        txt = new_lan + txt



        # take features from the eot embedding (eot_token is the highest number in each sequence)
        state = txt[torch.arange(txt.shape[0]),
                  text.argmax(dim=-1)] @ backbone.text_projection

        # forward
        return vis_outs, txt, state


class Bridger_ViT(nn.Module):
    def __init__(self,
                 d_img = [768, 768, 768],
                 d_txt = 512,
                 d_model = 64,
                 nhead = 8,
                 num_stages = 3,
                 strides = [1, 1, 1],
                 num_layers = 12,
                 shared_weights = False,
                ):
        super().__init__()
        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers

        self.zoom_in, self.zoom_out = nn.ModuleList(), nn.ModuleList()
        self.linear1, self.linear2 = nn.ModuleList(), nn.ModuleList()
        self.fusion_v, self.fusion_t = nn.ModuleList(), nn.ModuleList()
        self.ln_v = nn.ModuleList()
        self.ln_t = nn.ModuleList()
        for i in range(num_stages):
            self.fusion_v.append(InteractorT(d_model=d_model, nhead=nhead))
            self.fusion_t.append(InteractorT(d_model=d_model, nhead=nhead))

            self.linear1.append(nn.Linear(d_txt, d_model))
            self.linear2.append(nn.Linear(d_model, d_txt))
            if i < num_stages - 1:
                self.zoom_in.append(SNPconv_layer(d_img[i], d_model, strides[i], 0, strides[i]))
                self.zoom_out.append(SNPdeconv_layer(d_model, d_img[i], kernel_size=strides[i], stride=strides[i]))
                self.ln_v.append(nn.LayerNorm(d_model))
                self.ln_t.append(nn.LayerNorm(d_model))
            else:
                self.zoom_in.append(SNPdeconv_layer(d_img[i], d_model, kernel_size=strides[i], stride=strides[i]))
                self.zoom_out.append(SNPconv_layer(d_model, d_img[i], strides[i], 0, strides[i]))
        self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, img, text, backbone):
        # vision
        img = img.type(backbone.dtype)
        vis_enc = backbone.visual
        vis = vis_enc.conv1(img)  # shape = [*, width, grid, grid]
        vis = vis.reshape(vis.shape[0], vis.shape[1], -1)  # shape = [*, width, grid ** 2]
        vis = vis.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        vis = torch.cat([
            vis_enc.class_embedding.to(vis.dtype) + torch.zeros(vis.shape[0], 1, vis.shape[-1],
                dtype=vis.dtype, device=vis.device), vis], dim=1)  # shape = [*, grid ** 2 + 1, width]
        vis = vis + vis_enc.positional_embedding.to(vis.dtype)
        vis = vis_enc.ln_pre(vis)

        vis = vis.permute(1, 0, 2)  # NLD -> LND

        # language
        txt = backbone.token_embedding(text).type(
            backbone.dtype)  # [batch_size, n_ctx, d_model]

        txt_enc = backbone.transformer
        txt = txt + backbone.positional_embedding.type(backbone.dtype)[:txt.size(1)]
        txt = txt.permute(1, 0, 2)  # NLD -> LND

        # fusion
        stage_i = 0
        vis_outs = []
        for i in range(self.num_layers):
            if (i+1)%4 != 0:
                vis = vis_enc.transformer.resblocks[i](vis)
                txt = txt_enc.resblocks[i](txt)
            else:
                vis = vis_enc.transformer.resblocks[i](vis)
                txt = txt_enc.resblocks[i](txt)
                # residual operation
                v = vis.clone()
                t = txt.clone()
                v = v[1:, :, :] # N, B, D
                v = v.permute(1, 2, 0) # B, D, N
                B, C, N = v.shape
                H = int(N ** 0.5)
                W = N // H
                v = v.reshape(B, C, H, W) # B, D, H, W
                v = self.zoom_in[stage_i](v)
                t = self.linear1[stage_i](t)
                # multi modal fusion
                B, C, H, W = v.shape
                v = v.reshape(B, C, -1).permute(2, 0, 1) # B, C, H, W -> B, C, HW -> HW, B, C(676, 64, 256)
                if stage_i > 0:
                    v, t = self.ln_v[stage_i-1](v+last_v), self.ln_t[stage_i-1](t+last_t)
                last_v, last_t = v, t
                v, t = self.fusion_v[stage_i](v, t), self.fusion_t[stage_i](t, v)
                v = v.permute(1, 2, 0).reshape(B, -1, H, W) # HW, B, C -> B, C, HW -> B, C, H, W
                # dimension recovery
                v = self.zoom_out[stage_i](v)
                t = self.linear2[stage_i](t)
                # residual connect
                B, C, _, _ = v.shape
                # B, C, H, W -> B, C, N -> N, B, C
                v = v.reshape(B, C, -1).permute(2, 0, 1)
                vis[1:, :, :] += v
                txt = txt + t
                stage_i += 1
                if stage_i < self.num_stages:
                    vis_out = vis[1:, :, :].permute(1, 2, 0) # B, D, N
                    B, C, N = vis_out.shape
                    H = int(N ** 0.5)
                    W = N // H
                    vis_out = vis_out.reshape(B, C, H, W) # B, D, H, W
                    vis_outs.append(vis_out)

        # After fusion
        # vision
        # 197, 64, 768 -> 64, 197, 768
        vis = vis.permute(1, 0, 2)  # LND -> NLD

        # x = vis_enc.ln_post(x[:, 0, :])
        # 64, 197, 768 -> 64, 196, 768
        vis = vis_enc.ln_post(vis[:, 1:, :])

        if vis_enc.proj is not None:
            vis = vis @ vis_enc.proj

        # 64, 196, 512 -> 64, 512, 196
        B, N, C = vis.shape
        H = int(N ** 0.5)
        W = N // H
        vis = vis.permute(0, 2, 1).reshape(B, C, H, W) # B, N, D -> B, D, N -> B, D, H, W
        vis_outs.append(vis)

        # language
        txt = txt.permute(1, 0, 2)  # LND -> NLD
        txt = backbone.ln_final(txt).type(backbone.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        state = txt[torch.arange(txt.shape[0]),
                  text.argmax(dim=-1)] @ backbone.text_projection

        # forward
        output = vis_outs, txt, state

        return output