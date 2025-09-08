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
    

# class PFExpEncoder(nn.Module):
#     def __init__(self, hparams):
#         super(PFExpEncoder, self).__init__()
#         self.vtx_map = nn.Linear(hparams.vtx_dim, hparams.feature_dim)
#         self.PPE = PositionalEncoding(hparams.feature_dims)
        
#         exp_encoder_layer = nn.TransformerEncoderLayer(d_model=hparams.feature_dim,
#                                                        nhead=hparams.num_heads,
#                                                        dim_feedforward=2*hparams.feature_dim,
#                                                        dropout=0.1,
#                                                        activation="gelu"
#                                                        )
#         self.encoder_layer = nn.Trans
        


class Decoder(nn.Module):
    def __init__(self, hparams, face_model = None):
        super(Decoder, self).__init__()
        self.concat_map = nn.Linear(hparams.feature_dim + hparams.feature_dim, hparams.feature_dim)
        self.vtx_map_r = nn.Linear(hparams.feature_dim, hparams.vtx_dim)
        nn.init.constant_(self.vtx_map_r.bias, 0)
        nn.init.constant_(self.vtx_map_r.weight, 0)
        self.PPE = PositionalEncoding(hparams.feature_dim)
        self.face_model = face_model

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
    def __init__(self, hparams, face_model = None):
        super(AutoEncoder, self).__init__()
        self.hparams = hparams
        self.device = torch.device("cuda:" + str(hparams.device - 1)) if hparams.device > 0 else torch.device("cpu")
        self.template = torch.Tensor(np.load(f"{hparams.root_dir + hparams.feature_dir}/{hparams.neutral_vtx_file}")).to(self.device)

        # Model 
        self.con_encoder = ConEncoder(hparams)
        self.exp_encoder = ExpEncoder(hparams)
        self.decoder = Decoder(hparams)

        self.face_model = face_model
        ## 3D landmark indices
        # self.upper_lip_verts = [49, 50, 51, 52, 53, 61, 62, 63 ] 
        # self.lower_lip_verts = [59, 58, 57, 56, 55, 67, 66, 65 ] 
        self.upper_lip_verts = [3725, 3732, 5708, 5695, 2081, 0, 4275, 6200]
        self.lower_lip_verts = [6213, 6346, 6461, 5518, 5957, 5841, 5702, 5711]
        self.upper_lip_vert_contact = [4275]
        self.lower_lip_vert_contact = [5702]
        
        # settings
        self.use_lap = hparams.use_lap
        self.use_lip_contact_loss = hparams.use_lip_contact_loss
        self.use_soft_lip_contact_loss = hparams.use_soft_lip_contact_loss
        
        # Loss
        self.mse = torch.nn.MSELoss()
        self.lip_mse = torch.nn.MSELoss()
        self.lip_strong_mse = torch.nn.MSELoss()
        self.lip_strong_l1 = torch.nn.L1Loss()
        self.lip_loose_mse = torch.nn.MSELoss()
        self.lip_loose_l1 = torch.nn.L1Loss()
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=hparams.triplet_margin)

    def compute_lip_dist(self, verts, upper_lip_idx, lower_lip_ids):
        # import pdb;pdb.set_trace()
        verts = verts.view(verts.shape[0], verts.shape[1], -1, 3)
        upper_lip = verts[:, :, upper_lip_idx,:] # [f_1,N,3]
        lower_lip = verts[:, :, lower_lip_ids,:] # [f_2,N,3]
        distances = torch.norm(upper_lip - lower_lip, dim=3) # [batch_size, Frame_num, len(upper_lip_idx)]
        return distances.mean(dim=2)
    
    ## version1
    def _lip_contact_loss(self, pred_verts, trgt_verts, upper_lip_ids, lower_lip_ids, thershold):
        
        pred_lip_dist = self.compute_lip_dist(pred_verts, upper_lip_ids, lower_lip_ids) 
        trgt_lip_dist = self.compute_lip_dist(trgt_verts, upper_lip_ids, lower_lip_ids)
        # import pdb;pdb.set_trace()

        loss = self.lip_mse(pred_lip_dist, trgt_lip_dist) # Penalize to make predicted lip distance and target lip distance stay close
        penalty = torch.relu(pred_lip_dist - thershold).mean() # Penalize if lip contact exceeds threshold
        
        return loss*0.1 + penalty
        # return penalty
    
    ## version2 that applies contact mask. 
    def __lip_contact_loss(self, pred_vertices, target_vertices, upper_lip_ids, lower_lip_ids, loose_scale = 0.1675, epsilon=3.8, use_mse_strong=False, use_mse_loose=False):
        # Compute the predicted and target lip distances
        pred_lip_distance = self.compute_lip_dist(pred_vertices, upper_lip_ids, lower_lip_ids)
        target_lip_distance = self.compute_lip_dist(target_vertices, upper_lip_ids, lower_lip_ids)
        # import pdb;pdb.set_trace()
        # Ensure shape compatibility
        if pred_lip_distance.shape != target_lip_distance.shape:
            raise ValueError(f"Shape mismatch: pred_lip_distance {pred_lip_distance.shape}, target_lip_distance {target_lip_distance.shape}")

        # Define a mask where lip contact happens (distance is less than epsilon)
        contact_mask = (target_lip_distance < epsilon).float()  # 1 when contact happens, 0 otherwise

        # When lips are in contact, follow source closely (use regular MSE)
        if use_mse_strong:
            close_contact_loss = self.lip_strong_mse(pred_lip_distance, target_lip_distance)
        else:
            close_contact_loss = self.lip_strong_l1(pred_lip_distance, target_lip_distance)
        close_contact_loss = close_contact_loss * contact_mask  # Apply mask to only enforce this when contact happens
        
        if use_mse_loose:
            # When lips are apart, follow source loosely (scale down the MSE)
            loose_contact_loss = self.lip_loose_mse(pred_lip_distance, target_lip_distance)
        else:
            loose_contact_loss = self.lip_loose_l1(pred_lip_distance, target_lip_distance)

        loose_contact_loss = loose_contact_loss * (1 - contact_mask)  # Apply mask to enforce this when lips are apart
        loose_contact_loss = loose_contact_loss * loose_scale  # Scale down the loss for non-contact

        # Combine both losses
        total_loss = close_contact_loss + loose_contact_loss

        # Return the average loss over the batch and frames
        return total_loss.mean(), [close_contact_loss.mean(), loose_contact_loss.mean()]
    
    ## version3 that applies contact mask with 
    def ___lip_contact_loss(self, pred_vertices, target_vertices, upper_lip_ids, lower_lip_ids, loose_scale = 0.1675, epsilon=4.2, use_mse_strong=False, use_mse_loose=False):
        # Compute the predicted and target lip distances
        pred_lip_distance = self.compute_lip_dist(pred_vertices, upper_lip_ids, lower_lip_ids)
        target_lip_distance = self.compute_lip_dist(target_vertices, upper_lip_ids, lower_lip_ids)
        # import pdb;pdb.set_trace()
        # Ensure shape compatibility
        if pred_lip_distance.shape != target_lip_distance.shape:
            raise ValueError(f"Shape mismatch: pred_lip_distance {pred_lip_distance.shape}, target_lip_distance {target_lip_distance.shape}")

        # Define a mask where lip contact happens (distance is less than epsilon)
        contact_mask = (target_lip_distance < epsilon).float()  # 1 when contact happens, 0 otherwise

        # When lips are in contact, follow source closely (use regular MSE)
        # vertex base
        # import pdb;pdb.set_trace()
        lip_vert_indices = self.upper_lip_verts + self.lower_lip_verts

        pred_vertices = pred_vertices.view(pred_vertices.shape[0], pred_vertices.shape[1], -1, 3)
        lip_pred_vertices = pred_vertices[:, :, lip_vert_indices,:]
        target_vertices = target_vertices.view(target_vertices.shape[0], target_vertices.shape[1], -1, 3)
        lip_target_vertices = target_vertices[:, :, lip_vert_indices,:]

        if use_mse_strong:
            close_contact_loss = self.lip_strong_mse(lip_pred_vertices, lip_target_vertices)
        else:
            close_contact_loss = self.lip_strong_l1(lip_pred_vertices, lip_target_vertices)
        close_contact_loss = close_contact_loss * contact_mask  # Apply mask to only enforce this when contact happens
        
        if use_mse_loose:
            # When lips are apart, follow source loosely (scale down the MSE)
            loose_contact_loss = self.lip_loose_mse(lip_pred_vertices, lip_target_vertices)
        else:
            loose_contact_loss = self.lip_loose_l1(lip_pred_vertices, lip_target_vertices)

        loose_contact_loss = loose_contact_loss * (1 - contact_mask)  # Apply mask to enforce this when lips are apart
        loose_contact_loss = loose_contact_loss * loose_scale  # Scale down the loss for non-contact

        # Combine both losses
        total_loss = close_contact_loss + loose_contact_loss

        # Return the average loss over the batch and frames
        return total_loss.mean(), [close_contact_loss.mean(), loose_contact_loss.mean()]

    def lip_contact_loss(self, pred_vertices, c_target_vertices, e_target_vertices, upper_lip_ids, lower_lip_ids, loose_scale = 0.1675, epsilon=3.8, use_mse_strong=False, use_mse_loose=True):
        # Compute the predicted and target lip distances
        c_target_lip_distance = self.compute_lip_dist(c_target_vertices, upper_lip_ids, lower_lip_ids)
        # import pdb;pdb.set_trace()
        # Ensure shape compatibility
        # Define a mask where lip contact happens (distance is less than epsilon)
        contact_mask = (c_target_lip_distance < epsilon).float()  # 1 when contact happens, 0 otherwise

        # When lips are in contact, follow source closely (use regular MSE)
        # vertex base
        lip_vert_indices = self.upper_lip_verts + self.lower_lip_verts

        pred_vertices = pred_vertices.view(pred_vertices.shape[0], pred_vertices.shape[1], -1, 3)
        lip_pred_vertices = pred_vertices[:, :, lip_vert_indices,:]
        c_target_vertices = c_target_vertices.view(c_target_vertices.shape[0], c_target_vertices.shape[1], -1, 3)
        c_lip_target_vertices = c_target_vertices[:, :, lip_vert_indices,:]
        e_target_vertices = e_target_vertices.view(e_target_vertices.shape[0], e_target_vertices.shape[1], -1, 3)
        e_lip_target_vertices = e_target_vertices[:, :, lip_vert_indices,:]

        if use_mse_strong:
            close_contact_loss = self.lip_strong_mse(lip_pred_vertices, c_lip_target_vertices)
        else:
            close_contact_loss = self.lip_strong_l1(lip_pred_vertices, c_lip_target_vertices)
        close_contact_loss = close_contact_loss * contact_mask  # Apply mask to only enforce this when contact happens
        
        if use_mse_loose:
            # When lips are apart, follow source loosely (scale down the MSE)
            loose_contact_loss = self.lip_loose_mse(lip_pred_vertices, e_lip_target_vertices)
        else:
            loose_contact_loss = self.lip_loose_l1(lip_pred_vertices, e_lip_target_vertices)

        loose_contact_loss = loose_contact_loss * (1 - contact_mask)  # Apply mask to enforce this when lips are apart
        loose_contact_loss = loose_contact_loss * loose_scale  # Scale down the loss for non-contact

        # Combine both losses
        total_loss = close_contact_loss + loose_contact_loss

        # Return the average loss over the batch and frames
        return total_loss.mean(), [close_contact_loss.mean(), loose_contact_loss.mean()]
    
    def compute_laplacian_loss(self, vertices, adjacency_list, lip_vert_indices):
        laplacian_loss = 0.0
        num_vertices = vertices.shape[2]  # Number of vertices in the mesh
        batch_size, num_frames, _, _ = vertices.shape  # vertices shape is (batch_size, num_frames, num_vertices, 3)

        # Filter adjacency list to include only lip vertices
        lip_adjacency_list = [adjacency_list[i] for i in lip_vert_indices]

        for i, neighbors in zip(lip_vert_indices, lip_adjacency_list):
            # Filter out invalid neighbors that are out of bounds
            valid_neighbors = [n for n in neighbors if 0 <= n < num_vertices]
            if not valid_neighbors:
                continue

            # Gather the target vertex and its neighbors
            v_i = vertices[:, :, i, :]  # Vertex i for all frames and batches
            v_neighbors = vertices[:, :, valid_neighbors, :]  # Neighboring vertices

            # Compute the average position of neighbors
            v_i_avg = v_neighbors.mean(dim=2)  # Average position of neighbors

            # Compute the Laplacian vector and add to the loss
            laplacian_loss += torch.norm(v_i - v_i_avg, dim=2).mean()  # Mean across batch and frames

        # Normalize by the number of lip vertices
        laplacian_loss /= len(lip_vert_indices)
        
        return laplacian_loss
    
    ## version that use soft mask & laplacian loss
    def soft_lip_contact_loss(self, pred_vertices, c_target_vertices, e_target_vertices, upper_lip_ids, lower_lip_ids, loose_scale=0.1675, epsilon=3.5, smoothing_factor=0.6, use_mse_strong=False, use_mse_loose=True):
        # Compute the predicted and target lip distances
        c_target_lip_distance = self.compute_lip_dist(c_target_vertices, upper_lip_ids, lower_lip_ids)
        
        # Define a soft mask for gradual transitions
        soft_contact_mask = torch.sigmoid((epsilon - c_target_lip_distance) / smoothing_factor)

        # Extract the lip vertices for each target
        lip_vert_indices = self.upper_lip_verts + self.lower_lip_verts
        pred_vertices = pred_vertices.view(pred_vertices.shape[0], pred_vertices.shape[1], -1, 3)
        lip_pred_vertices = pred_vertices[:, :, lip_vert_indices, :]
        c_target_vertices = c_target_vertices.view(c_target_vertices.shape[0], c_target_vertices.shape[1], -1, 3)
        c_lip_target_vertices = c_target_vertices[:, :, lip_vert_indices, :]
        e_target_vertices = e_target_vertices.view(e_target_vertices.shape[0], e_target_vertices.shape[1], -1, 3)
        e_lip_target_vertices = e_target_vertices[:, :, lip_vert_indices, :]

        # When lips are in contact, follow source closely
        if use_mse_strong:
            # close_contact_loss = self.lip_strong_mse(lip_pred_vertices, c_lip_target_vertices) # only lip vertices
            close_contact_loss = self.lip_strong_mse(pred_vertices, c_target_vertices) # only lip vertices
        else:
            # close_contact_loss = self.lip_strong_l1(lip_pred_vertices, c_lip_target_vertices)
            close_contact_loss = self.lip_strong_l1(pred_vertices, c_target_vertices)

        close_contact_loss = close_contact_loss * soft_contact_mask  # Apply soft mask

        # When lips are apart, follow expression loosely
        if use_mse_loose:
            # loose_contact_loss = self.lip_loose_mse(lip_pred_vertices, e_lip_target_vertices)
            loose_contact_loss = self.lip_loose_mse(pred_vertices, e_target_vertices)
        else:
            loose_contact_loss = self.lip_loose_l1(lip_pred_vertices, e_lip_target_vertices)

        loose_contact_loss = loose_contact_loss * (1 - soft_contact_mask)  # Apply inverse soft mask
        loose_contact_loss = loose_contact_loss * loose_scale  # Scale down the loss for non-contact

        # Wehn use laplacian loss, 
        if self.use_lap:
            import trimesh
            ICT_neutral_mesh_path = '/input/inyup/ICT-FaceKit/FaceXModel/generic_neutral_mesh.obj'
            mesh = trimesh.load(ICT_neutral_mesh_path, force='mesh')
            # import pdb;pdb.set_trace()
            adjacency_list = mesh.vertex_neighbors
            laplacian_loss = self.compute_laplacian_loss(lip_pred_vertices, adjacency_list, lip_vert_indices) if adjacency_list is not None else 0.0
            # Combine both losses
            total_loss = close_contact_loss + loose_contact_loss + laplacian_loss
            # Return the average loss over the batch and frames
            return total_loss.mean(), [close_contact_loss.mean(), loose_contact_loss.mean(), laplacian_loss.mean()]
        else: 
            total_loss = close_contact_loss + loose_contact_loss
            return total_loss.mean(), [close_contact_loss.mean(), loose_contact_loss.mean()]

          
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
        loss['con_tpl'] = self.hparams.w_con * self.triplet_loss(c_c1e1, c_c1e1, c_c2e1) \
                        + self.hparams.w_con * self.triplet_loss(c_c1e1, c_c1e1, c_c2e2) \
                        + self.hparams.w_con * self.triplet_loss(c_c1e1, c_c1e2, c_c2e1) \
                        + self.hparams.w_con * self.triplet_loss(c_c1e1, c_c1e2, c_c2e2) \
                        + self.hparams.w_con * self.triplet_loss(c_c2e2, c_c2e2, c_c1e1) \
                        + self.hparams.w_con * self.triplet_loss(c_c2e2, c_c2e2, c_c1e2) \
                        + self.hparams.w_con * self.triplet_loss(c_c2e2, c_c2e1, c_c1e1) \
                        + self.hparams.w_con * self.triplet_loss(c_c2e2, c_c2e1, c_c1e2)

        # Expression Triplet loss # (anchor, positive, negative)
        loss['exp_tpl'] = self.hparams.w_tpl * self.triplet_loss(e_c1e1, e_c1e1, e_c1e2) \
                        + self.hparams.w_tpl * self.triplet_loss(e_c1e1, e_c1e1, e_c2e2) \
                        + self.hparams.w_tpl * self.triplet_loss(e_c1e1, e_c2e1, e_c1e2) \
                        + self.hparams.w_tpl * self.triplet_loss(e_c1e1, e_c2e1, e_c2e2) \
                        + self.hparams.w_tpl * self.triplet_loss(e_c2e2, e_c2e2, e_c1e1) \
                        + self.hparams.w_tpl * self.triplet_loss(e_c2e2, e_c2e2, e_c2e1) \
                        + self.hparams.w_tpl * self.triplet_loss(e_c2e2, e_c1e2, e_c1e1) \
                        + self.hparams.w_tpl * self.triplet_loss(e_c2e2, e_c1e2, e_c2e1)

        # Lip Contact loss 
        # if self.face_model is not None:
            #  deformed_vertices = torch.tensor(self.face_model._deformed_vertices).to(self.device)
            #  loss['lip_contact'] = self.hparams.w_lip * self.lip_contact_loss(deformed_vertices, vtx_c1e1, self.upper_lip_verts, self.lower_lip_verts, self.lip_contact_threshold)
        
        if self.use_lip_contact_loss:
            
            if not self.use_soft_lip_contact_loss:
            ## version 2 lip loss
            # c1e1_agg_lip_contact_loss, c1e1_sep_lip_contact_losses = self.lip_contact_loss(recon_c1e1, vtx_c1e1, self.upper_lip_verts, self.lower_lip_verts)
            # c2e1_agg_lip_contact_loss, c2e1_sep_lip_contact_losses = self.lip_contact_loss(recon_c2e1, vtx_c2e1, self.upper_lip_verts, self.lower_lip_verts)
            # c1e2_agg_lip_contact_loss, c1e2_sep_lip_contact_losses = self.lip_contact_loss(recon_c2e1, vtx_c2e1, self.upper_lip_verts, self.lower_lip_verts)
            # c2e2_agg_lip_contact_loss, c2e2_sep_lip_contact_losses = self.lip_contact_loss(recon_c2e2, vtx_c2e2, self.upper_lip_verts, self.lower_lip_verts)
            ## version 3 lip loss
            # c1e1_agg_lip_contact_loss, c1e1_sep_lip_contact_losses = self.lip_contact_loss(recon_c1e1, vtx_c1e1, self.upper_lip_vert_contact, self.lower_lip_vert_contact)
            # c2e1_agg_lip_contact_loss, c2e1_sep_lip_contact_losses = self.lip_contact_loss(recon_c2e1, vtx_c2e1, self.upper_lip_vert_contact, self.lower_lip_vert_contact)
            # c1e2_agg_lip_contact_loss, c1e2_sep_lip_contact_losses = self.lip_contact_loss(recon_c2e1, vtx_c2e1, self.upper_lip_vert_contact, self.lower_lip_vert_contact)
            # c2e2_agg_lip_contact_loss, c2e2_sep_lip_contact_losses = self.lip_contact_loss(recon_c2e2, vtx_c2e2, self.upper_lip_vert_contact, self.lower_lip_vert_contact)
            ## version 4 lip loss (assume only cross recon case, b/c seeing the same case for each con and exp input won't help much i think)
                c1e2_agg_lip_contact_loss, c1e2_sep_lip_contact_losses = self.lip_contact_loss(recon_c1e2, vtx_c1e1, vtx_c2e2, self.upper_lip_vert_contact, self.lower_lip_vert_contact)
                c2e1_agg_lip_contact_loss, c2e1_sep_lip_contact_losses = self.lip_contact_loss(recon_c2e1, vtx_c2e2, vtx_c1e1, self.upper_lip_vert_contact, self.lower_lip_vert_contact)
            else:
                ## soft loss
                c1e2_agg_lip_contact_loss, c1e2_sep_lip_contact_losses = self.soft_lip_contact_loss(recon_c1e2, vtx_c1e1, vtx_c2e2, self.upper_lip_vert_contact, self.lower_lip_vert_contact) # epsilon got smaller
                c2e1_agg_lip_contact_loss, c2e1_sep_lip_contact_losses = self.soft_lip_contact_loss(recon_c2e1, vtx_c2e2, vtx_c1e1, self.upper_lip_vert_contact, self.lower_lip_vert_contact) # epsillon got smaller

                
                # c1e2_agg_lip_contact_loss, c1e2_sep_lip_contact_losses = self.soft_lip_contact_loss(recon_c1e2, vtx_c1e1, vtx_c2e2, self.upper_lip_vert_contact, self.lower_lip_vert_contact, epsilon=2.0) # epsilon got smaller
                # c2e1_agg_lip_contact_loss, c2e1_sep_lip_contact_losses = self.soft_lip_contact_loss(recon_c2e1, vtx_c2e2, vtx_c1e1, self.upper_lip_vert_contact, self.lower_lip_vert_contact, epsilon=2.0) # epsillon got smaller

                
                # c1e2_agg_lip_contact_loss, c1e2_sep_lip_contact_losses = self.soft_lip_contact_loss(recon_c1e2, vtx_c1e1, vtx_c2e2, self.upper_lip_vert_contact, self.lower_lip_vert_contact)
                # c2e1_agg_lip_contact_loss, c2e1_sep_lip_contact_losses = self.soft_lip_contact_loss(recon_c2e1, vtx_c2e2, vtx_c1e1, self.upper_lip_vert_contact, self.lower_lip_vert_contact)
                    
            ## version 1,2,3 case
            # loss['lip_contact'] = self.hparams.w_lip * c1e1_agg_lip_contact_loss \
            #     + self.hparams.w_lip * c2e1_agg_lip_contact_loss \
            #     + self.hparams.w_lip * c1e2_agg_lip_contact_loss \
            #     + self.hparams.w_lip * c2e2_agg_lip_contact_loss
            # loss['lip_strong'] = c1e1_sep_lip_contact_losses[0] + c2e1_sep_lip_contact_losses[0] + c1e2_sep_lip_contact_losses[0] + c2e2_sep_lip_contact_losses[0]
            # loss['lip_loose'] = c1e1_sep_lip_contact_losses[1] + c2e1_sep_lip_contact_losses[1] + c1e2_sep_lip_contact_losses[1] + c2e2_sep_lip_contact_losses[1]   
            
            ## version 4 case - old 
            # loss['lip_contact'] =  c1e2_agg_lip_contact_loss + c2e1_agg_lip_contact_loss 
            # loss['lip_strong'] = self.hparams.w_lip_strong * (c1e2_sep_lip_contact_losses[0] + c1e2_sep_lip_contact_losses[0])
            # loss['lip_loose'] = self.hparams.w_lip_loose * (c2e1_sep_lip_contact_losses[1] + c2e1_sep_lip_contact_losses[1])
            # if self.use_lap:
            #     loss['laplacian'] = self.hparams.w_lap * c2e1_sep_lip_contact_losses[2] + self.hparams.w_lap * c2e1_sep_lip_contact_losses[2]
            ## version 4 case - new from eve-s01
            loss['lip_contact'] = self.hparams.w_lip * c1e2_agg_lip_contact_loss \
                + self.hparams.w_lip * c2e1_agg_lip_contact_loss 
            loss['lip_strong'] = c1e2_sep_lip_contact_losses[0] + c1e2_sep_lip_contact_losses[0] 
            loss['lip_loose'] = c2e1_sep_lip_contact_losses[1] + c2e1_sep_lip_contact_losses[1] 
        
        return loss

    def reconstruct(self, vtx):
        # import pdb;pdb.set_trace()
        template = self.template.unsqueeze(0) # (1, V*3)
        template = template.unsqueeze(1) # (1, 1, V*3)
        # import pdb;pdb.set_trace()
        vtx_diff = vtx[0]-template
        vtx_exp_diff = vtx[1]-template

        c1 = self.con_encoder(vtx_diff) # (1, Fn, feature size)
        e1 = self.exp_encoder(vtx_exp_diff) # (1, Fn, feature size)

        if c1.shape[1] > e1.shape[1]:
            c1 = c1[:,:e1.shape[1],:]
        elif c1.shape[1] < e1.shape[1]:
            e1 = e1[:,:c1.shape[1],:]
            
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
    