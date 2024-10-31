import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32 #number of embedding dimensions
n_head = 4
n_layer = 4
dropout = 0.2
# ------------

torch.manual_seed(1337)

with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#characters that appear in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
#mapping of the chars to ints and vice-versa
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#splitting the data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    
class Head(nn.Module):
    #one head of self-attention

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        #compute affinities
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei) # prevents some nodes to communicates to prevent
        #weighted aggregation
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    #multiple attention heads in parallel

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # projection back into residual pathway
        return out

class FeedForward(nn.Module):
    #Linear layer + Non-linearity
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), #multiply by 4 as in the paper
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), #projection layer into residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    #Transformer block: communication(attention) followed by computation (ffwd)

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #normalizing and implementing residual connections 
        x = x + self.ffwd(self.ln2(x)) #normalizing and implementing residual connections
        #in the paper normalization comes after the layer but we implemented this reversed as it became more common
        return x
        


# bigram model
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        #each token reads logits for next tkn from lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #self.sa_head = Head(n_embd)
        #self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 heads with 8-dim self-attention. 4 communication channels in parallel
        #self.ffwd = FeedForward(n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets = None):
        B, T = idx.shape
        # idx and targets are (B, T) tensor of ints (Batch, Time, Channel)
        #in this context time represents the sequential nature of the data (block_size). Channel is representative of the logits for each token in the embedding table(vocab_size)
        tok_emb = self.token_embedding_table(idx) # (B,T,C) C=n_embd 
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T,C)
        x = tok_emb + pos_emb #(B,T,C) x holds the token and positional identity
        #x = self.sa_head(x) #apply one head of self-attention (B,T,C)
        #x = self.sa_heads(x) #apply multi-head of self-attention (B,T,C)
        #x = self.ffwd(x) #(B,T,C)
        x = self.blocks(x) #(B,T,C)
        x = self.ln_f(x) #final layer of normalization (B,T,C)
        logits = self.lm_head(x) #(B,T, vocab_size)

        if targets is None:
            loss = None
        else:
        #we reshape the logits and targets in order to use the cross_entropy function
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array of indices in current context
        for _ in range(max_new_tokens):
            #crop idx to block_size
            idx_cond = idx[:, -block_size:]
            #get predictions
            logits, loss = self(idx_cond)
            #focus on last time step (last element)
            logits = logits[:,-1,:] #becomes (B,C)
            #apply softmax
            probs = F.softmax(logits, dim=-1) #(B,C)
            #sample from prob distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            #append sampled idx to the sequence
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#training
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))