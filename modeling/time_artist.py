"""Building blocks for TimeArtist.

Copyright (2025) TimeArtist's Author

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Reference:
    https://github.com/TimeArtist-AAAI2026
"""


import torch
import torch.nn as nn
from einops import rearrange

from modeling.modules.base_model import BaseModel
from modeling.modules.blocks import TemporalEncoder, TemporalDecoder
from modeling.quantizer.quantizer import VectorQuantizer

import json
from omegaconf import OmegaConf
from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin


class TimeArtist(BaseModel, PyTorchModelHubMixin):
    def __init__(self, config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        super().__init__()
        self.config = config

        self.encoder = TemporalEncoder(config)
        self.decoder = TemporalDecoder(config)
        
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        self.apply(self._init_weights)

        self.quantize = VectorQuantizer(
            codebook_size=config.model.vq_model.codebook_size,
            token_size=config.model.vq_model.token_size,
            commitment_cost=config.model.vq_model.commitment_cost,
            use_l2_norm=config.model.vq_model.use_l2_norm,
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # x -> z -> q, q_ind
    def encode(self, x):
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        q, dict = self.quantize(z)
        q_ind = dict['min_encoding_indices']
        return q, q_ind

    # q -> x_recon
    def decode(self, q):
        x_recon = self.decoder(q)
        return x_recon

    # q_ind -> x_recon
    def decode_tokens(self, tokens):
        tokens = tokens.squeeze(1)
        batch, seq_len = tokens.shape
        tokens = tokens.reshape(-1)

        q = self.quantize.get_codebook_entry(tokens)
        q = q.reshape(batch, 1, seq_len, -1)
        q = rearrange(q, 'b h w c -> b c h w').contiguous()

        x_recon = self.decode(q)
        return x_recon

    def decode_multi_tokens(self, tokens1, tokens2, alpha=0.5):
        tokens1, tokens2 = tokens1.squeeze(1), tokens2.squeeze(1)
        tokens = torch.cat((tokens1, tokens2), dim=1)
        batch, seq_len = tokens.shape
        tokens = tokens.reshape(-1)

        q = self.quantize.get_codebook_entry(tokens)
        q = q.reshape(batch, 1, seq_len, -1)
        q1, q2 = torch.chunk(q, 2, dim=2)
        q = q1 * alpha + q2 * (1 - alpha)
        q = rearrange(q, 'b h w c -> b c h w').contiguous()

        x_recon = self.decode(q)
        return x_recon
    
    def forward(self, x):
        q, q_ind = self.encode(x)
        x_recon = self.decode(q)
        return x_recon, q_ind

