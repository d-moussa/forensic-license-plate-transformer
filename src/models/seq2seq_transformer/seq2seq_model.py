import torch
import torch.nn as nn
import math
from torch import Tensor
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
from torch.nn.functional import softmax


class Seq2SeqTransformer(nn.Module):
    def __init__(self, tgt_vocab_size: int, feature_length: int, projection_dim, nhead: int, dim_feedforward: int,
                 num_encoder_layers: int, num_decoder_layers: int, dropout: float = 0.1, cnn_stem=None,
                 knowledge_classes=None):
        super(Seq2SeqTransformer, self).__init__()
        self.model_type = 'Transformer'

        # cnn to extract features before transformer is trained on them, None if not used
        self.cnn_stem = cnn_stem

        print("INFO -- Using image slices as model input")
        self.slice_projection = nn.Linear(feature_length, projection_dim) if projection_dim is not None else None
        dim = projection_dim if projection_dim is not None else feature_length

        # define encoder and decoder
        encoder_layers = TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.encoder = TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        decoder_layers = TransformerDecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.decoder = TransformerDecoder(decoder_layers, num_layers=num_decoder_layers)

        # linear layer to compute likeliness for every alphabet character to occur at  curr. position
        self.linear = nn.Linear(dim, tgt_vocab_size)

        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, dim)
        self.positional_encoding = PositionalEncoding(dim, dropout=dropout)
        self.knowledge_encoding = SimpleUpscaler(num_classes=knowledge_classes, embed_size=dim, dropout=dropout) if knowledge_classes is not None else None

    def forward(self, src: Tensor, trg: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor, tgt_mask: Tensor, knowledge_class: Tensor = None):


        if self.slice_projection is not None:
            src = self.slice_projection(src)
        src_emb = self.positional_encoding(src)
        if self.knowledge_encoding is not None:
            #print(knowledge_class)
            src_emb = self.knowledge_encoding(knowledge_input=knowledge_class, src=src_emb)


        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))

        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        weights = None
        outs = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        return self.linear(outs), weights

    def encode(self, src: Tensor, knowledge_class: Tensor = None):
        # Check whether additional projection layer is used
        if self.slice_projection is not None:
            src = self.slice_projection(src)
        src_emb = self.positional_encoding(src)
        if self.knowledge_encoding is not None:
            src_emb = self.knowledge_encoding(knowledge_input=knowledge_class, src=src_emb)

        return self.encoder(src_emb)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory,
            tgt_mask)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, src: Tensor):
        return self.dropout(src +
                            self.pos_embedding[:src.size(0), :])

class SimpleUpscaler(nn.Module):
    """
    Implementation of this class based on: https://github.com/elisabethfischer/KeBERT4Rec/blob/master/models/bert_modules/embedding/content.py
    """
    def __init__(self, num_classes: int, embed_size: int, dropout: float):
        super().__init__()
        self.upscaler = nn.Linear(num_classes, embed_size)
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)

    def forward(self, knowledge_input, src):
        # print(knowledge_input)
        one_hot = torch.nn.functional.one_hot(knowledge_input, self.num_classes).float()
        knowledge_embedding = self.upscaler(one_hot).unsqueeze(0)
        return self.dropout(src + knowledge_embedding)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)




