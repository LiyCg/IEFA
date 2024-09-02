import torch
import torch.nn as nn
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

## ConEncoder and ExpEncoder are exactly the same? 
class ConEncoder(nn.Module):
    def __init__(self, hparams):
        super(ConEncoder, self).__init__()
        self.vtx_map = nn.Linear(hparams.vtx_dim, hparams.feature_dim)
        self.PPE = PositionalEncoding(hparams.feature_dim)

        con_encoder_layer = nn.TransformerEncoderLayer(d_model=hparams.feature_dim,
                                                       nhead=hparams.num_heads,
                                                       dim_feedforward=2 * hparams.feature_dim,
                                                       dropout=0.1,
                                                       activation="gelu")
        self.con_encoder = nn.TransformerEncoder(con_encoder_layer, num_layers=hparams.num_layers)

    def forward(self, vtx_diff):
        vtx = self.vtx_map(vtx_diff)
        vtx = self.PPE(vtx)
        content = self.con_encoder(vtx)
        return content

class ExpEncoder(nn.Module):
    def __init__(self, hparams):
        super(ExpEncoder, self).__init__()
        self.vtx_map = nn.Linear(hparams.vtx_dim, hparams.feature_dim)
        self.PPE = PositionalEncoding(hparams.feature_dim)

        exp_encoder_layer = nn.TransformerEncoderLayer(d_model=hparams.feature_dim,
                                                          nhead=hparams.num_heads,
                                                          dim_feedforward=2 * hparams.feature_dim,
                                                          dropout=0.1,
                                                          activation="gelu")
        self.exp_encoder = nn.TransformerEncoder(exp_encoder_layer, num_layers=hparams.num_layers) # 4 layers by default 

    def forward(self, vtx_diff):
        vtx = self.vtx_map(vtx_diff)
        vtx = self.PPE(vtx)
        content = self.exp_encoder(vtx)
        return content # (batch_size, frame_num, hparams.feature_dim)
    

class PFExpEncoder(nn.Module):
    def __init__(self, hparams):
        super(PFExpEncoder, self).__init__()
        self.vtx_map = nn.Linear(hparams.vtx_dim, hparams.feature_dim)
        self.PPE = PositionalEncoding(hparams.feature_dims)
        
        exp_encoder_layer = nn.TransformerEncoderLayer(d_model=hparams.feature_dim,
                                                       nhead=hparams.num_heads,
                                                       dim_feedforward=2*hparams.feature_dim,
                                                       dropout=0.1,
                                                       activation="gelu"
                                                       )
        self.encoder_layer = nn.Trans
        


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.concat_map = nn.Linear(hparams.feature_dim + hparams.feature_dim, hparams.feature_dim)
        self.vtx_map_r = nn.Linear(hparams.feature_dim, hparams.vtx_dim)
        nn.init.constant_(self.vtx_map_r.bias, 0)
        nn.init.constant_(self.vtx_map_r.weight, 0)
        self.PPE = PositionalEncoding(hparams.feature_dim)

        # transformer encoder as the decoder ( content + expression -> face )
        decoder_layer = nn.TransformerEncoderLayer(d_model=hparams.feature_dim,
                                                   nhead=hparams.num_heads,
                                                   dim_feedforward=2 * hparams.feature_dim,
                                                   dropout=0.1,
                                                   activation="gelu")
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=hparams.num_layers)

    def forward(self, content, expression):
        # [1, frame num, 1024]
        features = torch.cat([content, expression], 2) # concat along feature dimension -> [batch_size, frame_num, 1024 * 2]
        features = self.concat_map(features)
        features = self.PPE(features)
        x = self.decoder(features)
        vtx_diff = self.vtx_map_r(x)
        return vtx_diff

class AutoEncoder(nn.Module):
    def __init__(self, hparams):
        super(AutoEncoder, self).__init__()
        self.hparams = hparams
        self.device = torch.device("cuda:" + str(hparams.device - 1)) if hparams.device > 0 else torch.device("cpu")
        self.template = torch.Tensor(np.load(f"{hparams.root_dir + hparams.feature_dir}/{hparams.neutral_vtx_file}")).to(self.device)

        # Model
        self.con_encoder = ConEncoder(hparams)
        self.exp_encoder = ExpEncoder(hparams)
        self.decoder = Decoder(hparams)

        # Loss
        self.mse = torch.nn.MSELoss()
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=hparams.triplet_margin)

    def forward(self, vtx_c1e1, vtx_c2e1, vtx_c1e2, vtx_c2e2):
        # import pdb;pdb.set_trace()
        template = self.template.unsqueeze(0) # (1, V*3)
        template = template.unsqueeze(1) # (1, 1, V*3)
        vtx_diff_c1e1 = vtx_c1e1-template
        vtx_diff_c2e1 = vtx_c2e1-template
        vtx_diff_c1e2 = vtx_c1e2-template
        vtx_diff_c2e2 = vtx_c2e2-template
        loss = {}

        c_c1e1 = self.con_encoder(vtx_diff_c1e1)
        e_c1e1 = self.exp_encoder(vtx_diff_c1e1)

        c_c2e1 = self.con_encoder(vtx_diff_c2e1)
        e_c2e1 = self.exp_encoder(vtx_diff_c2e1)

        c_c1e2 = self.con_encoder(vtx_diff_c1e2)
        e_c1e2 = self.exp_encoder(vtx_diff_c1e2)

        c_c2e2 = self.con_encoder(vtx_diff_c2e2)
        e_c2e2 = self.exp_encoder(vtx_diff_c2e2)

        # Cross Reconstruction loss
        recon_c1e2 = self.decoder(c_c1e1, e_c2e2) + template
        recon_c2e1 = self.decoder(c_c2e2, e_c1e1) + template
        loss['cross'] = self.hparams.w_cross * self.mse(vtx_c1e2, recon_c1e2) \
                      + self.hparams.w_cross * self.mse(vtx_c2e1, recon_c2e1)

        # Self Reconstruction loss
        recon_c1e1 = self.decoder(c_c1e1, e_c1e1) + template
        recon_c2e2 = self.decoder(c_c2e2, e_c2e2) + template
        loss['self'] = self.hparams.w_self * self.mse(vtx_c1e1, recon_c1e1) \
                     + self.hparams.w_self * self.mse(vtx_c2e2, recon_c2e2)

        # Content Triplet loss # (anchor, positive, negative)
        loss['con_tpl'] = self.hparams.w_tpl * self.triplet_loss(c_c1e1, c_c1e1, c_c2e1) \
                        + self.hparams.w_tpl * self.triplet_loss(c_c1e1, c_c1e1, c_c2e2) \
                        + self.hparams.w_tpl * self.triplet_loss(c_c1e1, c_c1e2, c_c2e1) \
                        + self.hparams.w_tpl * self.triplet_loss(c_c1e1, c_c1e2, c_c2e2) \
                        + self.hparams.w_tpl * self.triplet_loss(c_c2e2, c_c2e2, c_c1e1) \
                        + self.hparams.w_tpl * self.triplet_loss(c_c2e2, c_c2e2, c_c1e2) \
                        + self.hparams.w_tpl * self.triplet_loss(c_c2e2, c_c2e1, c_c1e1) \
                        + self.hparams.w_tpl * self.triplet_loss(c_c2e2, c_c2e1, c_c1e2)

        # Expression Triplet loss # (anchor, positive, negative)
        loss['exp_tpl'] = self.hparams.w_tpl * self.triplet_loss(e_c1e1, e_c1e1, e_c1e2) \
                        + self.hparams.w_tpl * self.triplet_loss(e_c1e1, e_c1e1, e_c2e2) \
                        + self.hparams.w_tpl * self.triplet_loss(e_c1e1, e_c2e1, e_c1e2) \
                        + self.hparams.w_tpl * self.triplet_loss(e_c1e1, e_c2e1, e_c2e2) \
                        + self.hparams.w_tpl * self.triplet_loss(e_c2e2, e_c2e2, e_c1e1) \
                        + self.hparams.w_tpl * self.triplet_loss(e_c2e2, e_c2e2, e_c2e1) \
                        + self.hparams.w_tpl * self.triplet_loss(e_c2e2, e_c1e2, e_c1e1) \
                        + self.hparams.w_tpl * self.triplet_loss(e_c2e2, e_c1e2, e_c2e1)

        return loss

    def reconstruct(self, vtx):
        # import pdb;pdb.set_trace()
        template = self.template.unsqueeze(0) # (1, V*3)
        template = template.unsqueeze(1) # (1, 1, V*3)

        vtx_diff = vtx[0]-template
        vtx_exp_diff = vtx[1]-template

        c1 = self.con_encoder(vtx_diff) # (1, Fn, feature size)
        e1 = self.exp_encoder(vtx_exp_diff) # (1, Fn, feature size)

        vtx_recon = self.decoder(c1, e1) + template

        return vtx_recon
        


if __name__ == '__main__':
    
    class HParams:
        def __init__(self):
            self.vtx_dim = 5023*3
            self.feature_dim = 512
            self.num_heads = 8
            self.num_layers = 6
            self.triplet_margin = 1.0
            self.w_cross = 1.0
            self.w_self = 1.0
            self.w_tpl = 1.0
            self.device = 0
            self.root_dir = '/source/inyup/TeTEC/faceClip/'
            self.feature_dir = 'data/feature'
            self.neutral_vtx_file = 'M003_front_neutral_1_011_last_fr.npy'
            
    hparams = HParams()
    
    model = AutoEncoder(hparams)
    DEVICE = model.device
    model = model.to(DEVICE)
    
    batch_size = 1
    frame_num = 10
    vtx_c1e1 = torch.randn(batch_size, frame_num, hparams.vtx_dim).to(DEVICE)
    vtx_c2e1 = torch.randn(batch_size, frame_num, hparams.vtx_dim).to(DEVICE)
    vtx_c1e2 = torch.randn(batch_size, frame_num, hparams.vtx_dim).to(DEVICE)
    vtx_c2e2 = torch.randn(batch_size, frame_num, hparams.vtx_dim).to(DEVICE)
    
    loss = model(vtx_c1e1, vtx_c2e2, vtx_c1e2, vtx_c2e1)
    print(loss)
    