import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
from models.transformer import *
from models.SAFR_block import *
from einops import rearrange, repeat
import math
from models.resnet_dilation import resnet18 as resnet18_dilation

### merge the handwriting style and printed content
class Mix_TR(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=1, num_decoder_layers=1,
                 dim_feedforward=2048, dropout=0.1, activation="relu", return_intermediate_dec=False,
                 normalize_before=True):
        super(Mix_TR, self).__init__()
        
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        style_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.style_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, style_norm)

        # fre_norm = nn.LayerNorm(d_model) if normalize_before else None
        # self.fre_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, fre_norm)

        con_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.con_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, con_norm)


        ### fusion the content and style in the transformer decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec)
        
        # fre_decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        # self.fre_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, fre_decoder_norm,
        #                                 return_intermediate=return_intermediate_dec)
        
        # self.add_position1D = PositionalEncoding(dropout=0.1, dim=d_model) # add 1D position encoding
        self.add_position2D = PositionalEncoding2D(dropout=0.1, d_model=d_model) # add 2D position encoding
        # self.high_pro_mlp = nn.Sequential(
        #     nn.Linear(512, 4096), nn.GELU(), nn.Linear(4096, 256))
        self.low_pro_mlp = nn.Sequential(
            nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 256))
        # self.low_feature_filter = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

        self.SAFR_module = SAFR_Block()

        self._reset_parameters()

        ### low frequency style encoder
        self.Feat_Encoder = self.initialize_resnet18()
        self.style_dilation_layer = resnet18_dilation().conv5_x
        
        ### hig frequency style encoder
        # self.freq_encoder = self.initialize_resnet18()
        # self.freq_dilation_layer = resnet18_dilation().conv5_x

        ### content encoder
        # self.content_encoder = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(models.resnet18(weights='ResNet18_Weights.DEFAULT').children())[1:-2]))
        self.content_encoder = self.initialize_resnet18()
        self.content_dilation_layer = resnet18_dilation().conv5_x

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def initialize_resnet18(self,):
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.layer4 = nn.Identity()
        resnet.fc = nn.Identity()
        resnet.avgpool = nn.Identity()
        return resnet

    def process_style_feature(self, encoder, dilation_layer, style, add_position2D, style_encoder):
        style = encoder(style)
        style = rearrange(style, 'n (c h w) ->n c h w', c=256, h=16).contiguous()
        style = dilation_layer(style)
        style = add_position2D(style)
        style = rearrange(style, 'n c h w ->(h w) n c').contiguous()
        style = style_encoder(style)
        return style

    
    def get_low_style_feature(self, style):
        return self.process_style_feature(self.Feat_Encoder, self.style_dilation_layer, style, self.add_position2D, self.style_encoder)

    # def get_high_style_feature(self, laplace):
    #     return self.process_style_feature(self.freq_encoder, self.freq_dilation_layer, laplace, self.add_position2D, self.fre_encoder)

    def get_content_style_feature(self, content):
        return self.process_style_feature(self.content_encoder, self.content_dilation_layer, content, self.add_position2D, self.con_encoder)

    
    def forward(self, style, laplace, content):
        # get the high frequency and style feature
        anchor_style = style[:, 0, :, :].clone().unsqueeze(1).contiguous()
        pos_style = style[:, 1, :, :].clone().unsqueeze(1).contiguous()

        # get the low frequency and style feature
        anchor_low = anchor_style
        anchor_low_feature = self.get_low_style_feature(anchor_low) # anchor_low_feature 256, 24, 512

        anchor_low_feature = self.SAFR_module(anchor_low_feature) # anchor_low_feature 256, 24, 512

        anchor_low_nce = self.low_pro_mlp(anchor_low_feature) # t n c
        anchor_low_nce = torch.mean(anchor_low_nce, dim=0)

        pos_low = pos_style 
        pos_low_feature = self.get_low_style_feature(pos_low)  # pos_low_feature 256, 24, 512

        pos_low_nce = self.low_pro_mlp(pos_low_feature)
        pos_low_nce = torch.mean(pos_low_nce, dim=0)

        low_nce_emb = torch.stack([anchor_low_nce, pos_low_nce], dim=1) # B 2 C
        low_nce_emb = nn.functional.normalize(low_nce_emb, p=2, dim=2)


        # content encoder
        if content.shape[1] == 1:
            anchor_content = content
        else:
            anchor_content = content[:, 0, :, :].unsqueeze(1).contiguous()

        content_feat = self.get_content_style_feature(anchor_content)


        style_hs = self.decoder(content_feat, anchor_low_feature, tgt_mask=None)

        # hs = self.fre_decoder(style_hs[0], anchor_high_feature, tgt_mask=None)
        #
        # return hs[0].permute(1, 0, 2).contiguous(), high_nce_emb, low_nce_emb # n t c
        return style_hs[0].permute(1, 0, 2).contiguous(), low_nce_emb # n t c



    def generate(self, style, laplace, content):
        if style.shape[1] == 1:
            anchor_style = style
            # anchor_high = laplace
        else:
            anchor_style = style[:, 0, :, :].unsqueeze(1).contiguous()
            # anchor_high = laplace[:, 0, :, :].unsqueeze(1).contiguous()

        # get the highg frequency and style feature
        # anchor_high_feature = self.get_high_style_feature(anchor_high) # t n c
        # get the low frequency and style feature
        anchor_low = anchor_style
        anchor_low_feature = self.get_low_style_feature(anchor_low)
        # anchor_mask = self.low_feature_filter(anchor_low_feature)
        # anchor_low_feature = anchor_low_feature * anchor_mask
        anchor_low_feature = self.SAFR_module(anchor_low_feature) # anchor_low_feature 256, 24, 512

        # content encoder
        if content.shape[1] == 1:
            anchor_content = content
        else:
            anchor_content = content[:, 0, :, :].unsqueeze(1).contiguous()
        content_feat = self.get_content_style_feature(anchor_content)

        # fusion of content and style features
        style_hs = self.decoder(content_feat, anchor_low_feature, tgt_mask=None)
        # hs = self.fre_decoder(style_hs[0], anchor_high_feature, tgt_mask=None)
        
        # return hs[0].permute(1, 0, 2).contiguous()
        return style_hs[0].permute(1, 0, 2).contiguous()