import json
import math
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from .layers import Encoder, Decoder
from .vq import ResidualVectorQuantizeLLM

import torch
import torch.nn as nn
from functools import partial
from transformers import BertTokenizer, BertModel
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenBERTFLANEmbedder(AbstractEncoder):
    """Uses the Bert transformer encoder for text from microsoft"""
    def __init__(self, freeze=True, device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # 
        self.caption_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        self.to(device=device)
        if freeze: self.freeze()
        #print(f"{self.caption_encoder.__class__.__name__} comes with {count_params(self.caption_encoder) * 1.e-6:.2f} M params.")

    def freeze(self):
        self.caption_encoder = self.caption_encoder.eval()
        for param in self.caption_encoder.parameters():
            param.requires_grad = False
        
    def to(self,device):
        self.caption_encoder.to(device)
        self.device = device

    def encode(self, text):
        clap_batch_encoding = self.bert_tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        ori_tokens = clap_batch_encoding["input_ids"].to(self.device)
        outputs = self.caption_encoder(input_ids=ori_tokens) # 
        z = outputs.last_hidden_state
        return z

class FrozenT5(AbstractEncoder):
    """Uses the Bert transformer encoder for text from microsoft"""
    def __init__(self, t5version='t5-small', freeze=True, device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5version)
        self.t5_transformer = T5EncoderModel.from_pretrained(t5version)
        self.max_length = max_length
        self.to(device=device)
        if freeze: self.freeze()
        #print(f"{self.caption_encoder.__class__.__name__} comes with {count_params(self.caption_encoder) * 1.e-6:.2f} M params.")

    def freeze(self):
        self.t5_transformer = self.t5_transformer.eval()
        for param in self.t5_transformer.parameters():
            param.requires_grad = False
        
    def to(self,device):
        self.t5_transformer.to(device)
        self.device = device

    def encode(self, text):
        t5_batch_encoding = self.t5_tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        struct_tokens = t5_batch_encoding["input_ids"].to(self.device)
        z = self.t5_transformer(input_ids=struct_tokens).last_hidden_state
        return z


class MSCodecLM(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        encoder_dim=64,
        encoder_rates=[2, 4, 5, 8],
        latent_dim=None,
        decoder_dim=1536,
        decoder_rates=[8, 5, 4, 2],
        attn_window_size=8,
        codebook_size=4096,
        codebook_dim=8,
        vq_strides=[4, 2, 1],
        use_cblinear=True,
        local_embedding_path=None,
        global_embedding_path=None,
        noise=True,
        depthwise=True,
    ):
        super().__init__()
        self.sampling_rate = sample_rate
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(
            encoder_dim,
            encoder_rates,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )
        self.n_codebooks = len(vq_strides)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.vq_strides = vq_strides
        self.attn_window_size = attn_window_size
        # print('latent_dim ', latent_dim)
        # assert 1==2
        checkpoint = torch.load(local_embedding_path, map_location="cpu")['weight']
        g_ckpt = torch.load(global_embedding_path, map_location="cpu")
        self.quantizer = ResidualVectorQuantizeLLM(
            input_dim=latent_dim,
            vq_strides=vq_strides,
            checkpoint=checkpoint,
            g_ckpt =g_ckpt,
            use_cblinear=use_cblinear,
        )
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            noise,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )

    def get_lcm(self, a, b):
        return abs(a * b) // math.gcd(a, b)

    def preprocess(self, audio_data):
        length = audio_data.shape[-1]
        # print('self.vq_strides[0] ', self.vq_strides[0], self.vq_strides)
        # print('self.attn_window_size ', self.attn_window_size)
        lcm = self.get_lcm(self.vq_strides[0], self.attn_window_size or 1)
        #print('lcm ', lcm)
        pad_to = self.hop_length * lcm
        #print('pad_to ', pad_to)
        right_pad = math.ceil(length / pad_to) * pad_to - length
        #print('right_pad ', right_pad)
        audio_data = nn.functional.pad(audio_data, (0, right_pad)) # make padding
        return audio_data

    def forward(self, audio_data: torch.Tensor, features=None, text_features=None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        length = audio_data.shape[-1]
        #print('length ', length)
        audio_data = self.preprocess(audio_data)
        #print('audio_data ', audio_data.shape)
        z = self.encoder(audio_data) # encode the audio
        z_q, codes, commitment_loss, codebook_loss, global_semantic_loss, local_semantic_loss = self.quantizer(z, features, text_features)
        audio_hat = self.decoder(z_q)
        return audio_hat[..., :length], codes, commitment_loss, codebook_loss, global_semantic_loss, local_semantic_loss

    def encode(self, audio_data: torch.Tensor) -> List[torch.Tensor]:
        audio_data = self.preprocess(audio_data)
        #print('audio_data ', audio_data.shape)
        z = self.encoder(audio_data)
        #print('z ', z.shape)
        _, codes, commitment_loss, codebook_loss, global_semantic_loss, local_semantic_loss = self.quantizer(z)
        return codes

    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        z_q = self.quantizer.from_codes(codes)
        audio_hat = self.decoder(z_q)
        return audio_hat

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        model = cls(**config)
        return model

    def inference(self, x):
        codes = self.encode(x)
        # print('codes ', codes)
        # assert 1==2
        wav = self.decode(codes)
        return wav, codes

