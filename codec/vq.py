from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.nn.functional as F
from .layers import WNConv1d


class VectorQuantizeLLM(nn.Module):
    """
    a new vector quantization layer with LLM embedding layer
    """
    def __init__(self, input_dim: int, stride: int = 1, use_cblinear=True, checkpoint=None):
        super().__init__()
        self.codebook_size = checkpoint.shape[0]
        self.codebook_dim = checkpoint.shape[1]
        self.use_cblinear = use_cblinear
        self.stride = stride
        if self.use_cblinear:
            self.cb_proj = nn.Linear(self.codebook_dim, input_dim) # map the codebook dim into input_dim
        else:
            # map the input into large-scale dim like codebook dim
            self.in_proj = WNConv1d(input_dim, self.codebook_dim, kernel_size=1)
            self.out_proj = WNConv1d(self.codebook_dim, input_dim, kernel_size=1)
        # init the codebook, and fix
        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.codebook.weight.data = checkpoint
        self.codebook.weight.data = self.codebook.weight.data.float()
        self.codebook.weight.requires_grad = False

    def forward(self, z, features=None, text_features=None):
        if self.stride > 1:
            z = torch.nn.functional.avg_pool1d(z, self.stride, self.stride)
        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        if self.use_cblinear == False:
            z_e = self.in_proj(z)  # z_e : (B x D x T)
        else:
            z_e = z
        z_q, indices = self.decode_latents(z_e)
        # print('z_q ', z_q.shape)
        # print('z_e ', z_e.shape)
        # assert 1==2
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
        z_q = z_e + (z_q - z_e).detach()  # noop in forward pass, straight-through gradient estimator in backward pass
        if self.use_cblinear == False:
            z_q = self.out_proj(z_q)
        if self.stride > 1:
            z_q = z_q.repeat_interleave(self.stride, dim=-1)
            # z_q = F.interpolate(z_q, size=[z_q.shape[2]*self.stride], mode='nearest')
        return z_q, indices, commitment_loss, codebook_loss

    def embed_code(self, embed_id):
        if self.use_cblinear:
            codebook = self.cb_proj(self.codebook.weight)
        else:
            codebook = self.codebook.weight 
        return F.embedding(embed_id, codebook)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        if self.use_cblinear:
            codebook = self.cb_proj(self.codebook.weight)
        else:
            codebook = self.codebook.weight  # codebook: (N x D)
        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)
        #print('codebook ', codebook.shape)
        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices

class ResidualVectorQuantizeLLM(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        vq_strides: List[int] = [4, 2, 1],
        checkpoint = None,
        g_ckpt=None,
        use_cblinear = True,
    ):
        super().__init__()
        self.n_codebooks = len(vq_strides)
        checkpoints = [g_ckpt,checkpoint,checkpoint]
        self.quantizers = nn.ModuleList(
            [VectorQuantizeLLM(input_dim, stride, use_cblinear, checkpoints[i]) for i, stride in enumerate(vq_strides)]
        )
        #print('self.quantizers ', self.quantizers)
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, z, features=None, text_features=None):
        z_q = 0
        commitment_loss = 0
        codebook_loss = 0
        global_semantic_loss = 0
        local_semantic_loss = 0
        residual = z
        codes = []
        for i, quantizer in enumerate(self.quantizers):
            z_q_i, indices_i, tmp_commit_loss, tmp_codebook_loss = quantizer(residual)
            if i == 0 and text_features is not None:
                tmp_g_se_loss = self.loss_fn(z_q_i.mean(2), text_features)
                global_semantic_loss += tmp_g_se_loss
                #print('z_q_i ', z_q_i.shape, text_features.shape, tmp_g_se_loss)
                # assert 1==2
            if i == 1 and features is not None:
                #print(features.shape, z_q_i.shape)
                de_fe = F.interpolate(features.transpose(1,2), size=[z_q_i.shape[2]], mode='nearest')
                # print('de_fe ', de_fe.shape)
                # assert 1==2
                tmp_l_se_loss = self.loss_fn(z_q_i, de_fe)
                local_semantic_loss += tmp_l_se_loss
                # print('tmp_l_se_loss ', tmp_l_se_loss)
                # print('de_fe ', de_fe.shape)
                # assert 1==2
                # assert 1==2
            z_q = z_q + z_q_i
            residual = residual - z_q_i
            codes.append(indices_i)
            commitment_loss += tmp_commit_loss
            codebook_loss += tmp_codebook_loss

        return z_q, codes, commitment_loss, codebook_loss, global_semantic_loss, local_semantic_loss

    def from_codes(self, codes: List[torch.Tensor]) -> torch.Tensor:
        z_q = 0.0
        for i in range(self.n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[i])
            #print('z_p_i ', z_p_i.shape)
            #z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q_i = z_p_i
            #z_q_i = F.interpolate(z_q_i, size=[z_q_i.shape[2]*self.quantizers[i].stride], mode='nearest') 
            z_q_i = z_q_i.repeat_interleave(self.quantizers[i].stride, dim=-1) # up-sampling
            z_q += z_q_i
        return z_q

