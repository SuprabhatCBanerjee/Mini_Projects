import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from collections import Counter

import pandas as pd
import math
import re

df = pd.read_csv("dataset/Conversation.csv")

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

#preprocess
pattern = '[^a-zA-Z0-9]'

def clean_data(data):
    return re.sub(pattern, ' ', data)

df['question'] = df["question"].apply(clean_data)
df['answer'] = df["answer"].apply(clean_data)

def format_data(data):
    text = f"<SOS> {data} <EOS>"
    return text

def join_data(data):
    text = f"{data['question']} {data['answer']}"
    return text

df["text"] = df.apply(join_data, axis=1)

vocab = {}


def build_vocab(data, vocab_size=5000):
    words = Counter()
    for sentence in data:
        words.update(re.findall(r'\w+', sentence))

    vocab = {word: i+4 for i, (word,_) in enumerate(words.most_common(vocab_size-4)) }
    vocab["<UNK>"] = 1
    vocab["<PAD>"] = 0
    vocab["<SOS>"] = 2
    vocab["<EOS>"] = 3
    return vocab

vocab = build_vocab(df["text"])


def tokenize(data):
    word_to_idx = []
    words = re.findall(r'\w+|<\w+>',data)
    # print(words)
    for word in words:
        if word in vocab:
            # print(word)
            word_to_idx.append(vocab[word])
        else:
            word_to_idx.append(vocab["<UNK>"])
    # print(word_to_idx)
    return word_to_idx

max_len_lst = max(max(df['question'].apply(len)), max(df['answer'].apply(len)))

#padding
def padding(data):
    padding = []
    padded_seq = []
    for i in range(max_len_lst-len(data)):
        padding.append(vocab["<PAD>"])

    padded_seq = data + padding 
    # print(padded_seq)
    return padded_seq


#imlpemeting transformer
#Self Attention ==> Positional Encoding ==> Multihead Attention
#Position-wise Feed-Forward Networks ==> Encoder-Decoder Architecure

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_probability = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probability, V)
        return output
    
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attention_output))
        return output

#FFN
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
#Postional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
#Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    

#Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, source_mask, target_mask):
        attention_output = self.self_attention(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(attention_output))

        attention_output = self.cross_attention(x, encoder_output, encoder_output, source_mask)
        x = self.norm2(x + self.dropout(attention_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
#Complete Transformer
class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length=500, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(source_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, source, target):
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2)
        target_mask = (target != 0).unsqueeze(1).unsqueeze(2)

        seq_length = target.size(1)

        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        nopeak_mask = nopeak_mask.to(device)
        target_mask = target_mask & nopeak_mask
        target_mask = target_mask.to(device)
        return source_mask, target_mask
    
    def forward(self, source, target):
        source_mask, target_mask = self.generate_mask(source, target)

        source_embed = self.dropout(self.positional_encoding(self.encoder_embedding(source)))
        target_embed = self.dropout(self.positional_encoding(self.decoder_embedding(target)))

        encoder_output = source_embed
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, source_mask)

        decoder_output = target_embed
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, source_mask, target_mask)

        output = self.fc(decoder_output)
        return output
    


vocab_size = len(vocab)
d_model = 128
num_heads = 8
num_layers = 4
d_ff = 256
max_seq_length = 500
dropout = 0.1

def get_model():
    model = Transformer(vocab_size, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    model.load_state_dict(torch.load("final_bot.pth"))
    model.eval()
    return model

idx_to_word = {idx : word for word, idx in vocab.items() }

def infer(model, x, max_len=100):
    
    x = tokenize(x)
    x = torch.tensor(x).to(device).unsqueeze(0)


    SOS = [vocab["<SOS>"]]

    for _ in range(max_len):
        dec_inp = torch.tensor(SOS, device=device).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(x, dec_inp)[:, -1]

        next_token = torch.argmax(logits, dim=1).item()
        SOS.append(next_token)

        if next_token == vocab["<EOS>"]:
            break

    return " ".join(idx_to_word[i] for i in SOS if i not in (2,3))

def predict(model : nn.Module, data):
    model.to(device)
    output = infer(model, data)
    return output


# print(predict(get_model(), "why"))

