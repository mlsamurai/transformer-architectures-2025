"""Tests for the simplified transformer components in the kaggle package."""

import pytest
import torch

from .architecture import (
    DecoderLayer,
    EncoderLayer,
    FeedForward,
    MultiHeadAttention,
    PositionalEncoding,
    Transformer,
)


def test_positional_encoding_output_shape():
    pe = PositionalEncoding(d_model=32, dropout=0.1)
    x = torch.randn(2, 10, 32)
    out = pe(x)
    assert out.shape == (2, 10, 32)


def test_multi_head_attention_output_shape():
    mha = MultiHeadAttention(d_model=32, num_heads=4)
    q = k = v = torch.randn(2, 10, 32)
    out = mha(q, k, v)
    assert out.shape == (2, 10, 32)


def test_multi_head_attention_with_mask():
    mha = MultiHeadAttention(d_model=32, num_heads=4)
    q = k = v = torch.randn(2, 10, 32)
    mask = torch.tril(torch.ones(10, 10)).expand(2, 1, 10, 10)
    out = mha(q, k, v, mask)
    assert out.shape == (2, 10, 32)


def test_multi_head_division():
    with pytest.raises(AssertionError):
        MultiHeadAttention(d_model=32, num_heads=7)


def test_feed_forward_output_shape():
    ffn = FeedForward(d_model=32, d_ff=64)
    x = torch.randn(2, 10, 32)
    assert ffn(x).shape == (2, 10, 32)


def test_encoder_layer_output_shape():
    layer = EncoderLayer(d_model=32, num_heads=4)
    x = torch.randn(2, 10, 32)
    assert layer(x).shape == (2, 10, 32)


def test_decoder_layer_output_shape():
    layer = DecoderLayer(d_model=32, num_heads=4)
    x = torch.randn(2, 10, 32)
    enc = torch.randn(2, 10, 32)
    assert layer(x, enc).shape == (2, 10, 32)


def test_transformer_output_shape():
    model = Transformer(src_vocab_size=20, tgt_vocab_size=25, d_model=32, num_heads=4, num_layers=1)
    src = torch.randint(0, 20, (2, 10))
    tgt = torch.randint(0, 25, (2, 10))
    src_mask = torch.ones(2, 1, 1, 10)
    tgt_mask = torch.tril(torch.ones(10, 10)).expand(2, 1, 10, 10)
    out = model(src, tgt, src_mask, tgt_mask)
    assert out.shape == (2, 10, 25)


def test_transformer_gradients():
    model = Transformer(src_vocab_size=20, tgt_vocab_size=25, d_model=32, num_heads=4, num_layers=1)
    src = torch.randint(0, 20, (2, 10))
    tgt = torch.randint(0, 25, (2, 10))
    src_mask = torch.ones(2, 1, 1, 10)
    tgt_mask = torch.tril(torch.ones(10, 10)).expand(2, 1, 10, 10)
    out = model(src, tgt, src_mask, tgt_mask)
    out.sum().backward()
    assert any(p.grad is not None for p in model.parameters())
