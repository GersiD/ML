import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
# Credit : https://nlp.seas.harvard.edu/2018/04/03/attention.html

class LayerNorm(nn.Module):
    """
    Layer normalization module

    2016 Paper showed that this applied in the forward pass of the network improves training time.
    """
    def __init__(self, layer_size: int, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.layer_size = layer_size
        self.alpha = nn.Parameter(torch.ones(layer_size))
        self.bias = nn.Parameter(torch.zeros(layer_size))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        Normalize the last dimension of input tensor x. Gaussian normalization.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.alpha * ((x - mean) / (std + self.eps)) + self.bias
        return x

class PositionwiseFeedForward(nn.Module):
    """
    linear(max(0, linear(x))) to further embed the input with its position in the sequence. 
    This idea comes from ResNets.

    Typically, d_ff is bigger than d_model, e.g. d_ff = 2048 and d_model = 512.
    """
    def __init__(self, d_model: int, d_ff: int, dropout:float=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor):
        """
        Apply the positionwise feed forward network to input x.
        x is expected to be of shape (batch_size, seq_len, d_model).
        """
        return self.w2(self.dropout(F.relu(self.w1(x))))

class Generator(nn.Module):
    """
    Define how to generate the output from the decoder. 
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        We use log_softmax here to allow for the loss function to be computed more efficiently.
        As calling log(softmax(x)) is not numerically stable, we use the log_softmax function directly.
        This could also just return the logits, which is more common in practice.
        Note the last dimension of x is expected to be the output embedding.

        x is expected to be of shape (batch_size, seq_len, d_model).
        returns (batch_size, seq_len, vocab_size) after applying log_softmax.
        """
        return F.log_softmax(self.proj(x), dim=-1)

class Embeddings(nn.Module):
    """
    A standard embedding class
    """
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model) # lut is short for lookup table 
        self.scale_term = math.sqrt(d_model)
        # The scaling here is not for normalization, but rather to ensure that the embeddings are of a suitable scale for the model to learn effectively.
        # By default th embeddings are ~ N(mean=0, var=1), which are small enough to not cause numerical instability.

    def forward(self, x):
        """
        Lookup embeddings for input x and scale them by sqrt(d_model).
        x is expected to be of shape (batch_size, seq_len).
        """
        # print("x shape embedding: ", x.shape)
        return self.lut(x) * self.scale_term

class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    pe is a matrix of shape (max_len, d_model) where each row corresponds to a position in the sequence.
    pe is represented as a (1, max_len, d_model) tensor to allow for broadcasting when indexed by an input sequence.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input tensor x.
        x is expected to be of shape (batch_size, seq_len, d_model).
        """
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) # pyright: ignore[reportIndexIssue]
        return self.dropout(x)

class SubLayerConnection(nn.Module): # all those arrows in the diagram are sublayer connections
    """
    A residual connection followed by a layer norm.
    """
    def __init__(self, size: int, dropout: float):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        Here residual connection means that the output of the sublayer is added to the input x.
        """
        # Don't apply norm on the outside since whoever calls this will do it
        return x + self.dropout(sublayer(self.norm(x))) 

class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SubLayerConnection(size, dropout) for _ in range(2)])
        self.size = size

    def forward(self, x, mask):
        """
        Apply self-attention and feed forward to input x with the given mask. 
        """
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, mask)) # This applies self-attention to the input x with the mask
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    """
    Core encoder is a stack of N EncoderLayer layers, this is what we mean by "Multi-Head Attention"
    """
    def __init__(self, layer: EncoderLayer, N: int):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)]) 
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn. NOTE MUTATES x."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """
    Decoder is made up of self-attn, src-attn and feed forward
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SubLayerConnection(size, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Apply self-attention, source attention and feed forward to input x with the given masks.
        """
        m = memory
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, tgt_mask))
        x = self.sublayer[1](x, lambda y: self.src_attn(y, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """
    def __init__(self, layer: DecoderLayer, N: int):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)]) 
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """NOTE MUTATES x."""
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """
    def __init__(self, d_model:int, encoder:Encoder, decoder:Decoder, src_embed:nn.Sequential, tgt_embed:nn.Sequential, generator:Generator):
        super(EncoderDecoder, self).__init__()
        self.d_model = d_model
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed # This is a sequential module that applies the embedding and positional encoding to the source input
        self.tgt_embed = tgt_embed # This is a sequential module that applies the embedding and positional encoding to the target input
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    @staticmethod
    def make_model(src_vocab:int, tgt_vocab:int, N:int=6, d_model:int=512, d_ff:int=1024, h:int=8, dropout:float=0.1):
        """
        Helper: Construct a model from hyperparameters.
        """
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        assert src_vocab == tgt_vocab, "Source and target vocab sizes must be the same for this model."
        emb = Embeddings(d_model, src_vocab)
        model = EncoderDecoder(
            d_model,
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            # different weights for src and tgt not what the 2017 paper did
            # nn.Sequential(Embeddings(d_model, src_vocab), c(position)), 
            # nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            nn.Sequential(emb, c(position)),
            nn.Sequential(emb, c(position)),
            Generator(d_model, tgt_vocab))
        
        # This was important from the original code
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        return model

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is all you need"
    """
    def __init__(self, h:int, d_model:int, dropout:float=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask = None, dropout = None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k) # because we want to normalize the dot product to have mean 0 and variance 1
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, V), p_attn

class Batch:
    """
    Object for holding a batch of data with mask during training.
    """
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2) # TODO: What does this do?
        if tgt is not None:
            self.trg = tgt[:, :-1] # TODO: What does this do?
            self.trg_y = tgt[:, 1:]
            self.trg_mask = self.make_std_mask(tgt[:, :-1], pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def subsequent_mask(size):
        """
        Mask out subsequent positions.
        Returns a square mask of shape (1, size, size) where the last two dimensions are upper triangular.
        This is used to prevent the decoder from attending to future positions in the sequence.
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    @staticmethod
    def make_std_mask(tgt, pad): # TODO: Dont understand this
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(Batch.subsequent_mask(tgt.size(-1)).type(tgt_mask.data.type()))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        # print("Batch: ", i)
        # print("Batch src shape: ", batch.src.shape)
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size:int, padding_idx:int, smoothing:float=0.0):
        super(LabelSmoothing, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return nn.KLDivLoss(size_average=False)(x, Variable(true_dist, requires_grad=False))

class NoamOpt:
    "Optim wrapper that implements the rate perscribed in the original work."
    def __init__(self, model_size:int, factor:int, warmup:int, optimizer:torch.optim.Optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

class SimpleLoss:
    def __init__(self, generator:Generator, criterion:LabelSmoothing, optim:NoamOpt):
        self.generator = generator
        self.criterion = criterion
        self.optim = optim

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))/norm
        loss.backward()
        self.optim.step()
        return loss.data.item() * norm
# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.1)
model = EncoderDecoder.make_model(V, V, N=2)
model_opt = NoamOpt(model.d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def data_gen(V, batch, nbatches):
    for _ in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1 # TODO: What does this do? Removing it doesnt change anything
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)

for epoch in range(10):
    model.train() # Set the model to training mode, needed for dropout to work during training
    run_epoch(data_gen(V, 30, 20), model, SimpleLoss(model.generator, criterion, model_opt))
    model.eval() # Set the model to evaluation mode, needed for dropout to stop for inference
    # run_epoch(data_gen(V, 30, 5), model, SimpleLoss(model.generator, criterion, None))
