# Modified from transformers.models.t5.modeling_t5
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.


import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizers import HuggingfaceTokenizer
from llama_cpp import Llama


class QuantizedT5EncoderModel:
    def __init__(
        self,
        text_len,
        model_path,
        tokenizer_path,
        n_ctx=512,
        n_threads=4,
        n_batch=512,
        device="cpu"
    ):
        self.text_len = text_len
        self.device = device

        # Initialize GGUF T5 encoder
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            embedding=True,   # IMPORTANT: encoder-only mode
            pooling_type=0
        )

        # Tokenizer interface
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=text_len,
            clean="whitespace"
        )

    def __call__(self, texts, device=None):
        device = self.device if device is None else device

        # Tokenize
        ids, mask = self.tokenizer(
            texts, return_mask=True, add_special_tokens=True
        )
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()

        # Create embeddings via GGUF
        embeddings_list = []
        for text in texts:
            result = self.llm.create_embedding(text)
            emb = torch.tensor(result['data'][0]['embedding'], device=device)
            embeddings_list.append(emb)

        # Slice/pad like original T5EncoderModel
        context = [u[:v] for u, v in zip(embeddings_list, seq_lens)]
        return context
