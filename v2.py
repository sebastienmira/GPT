import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32 #number of embedding dimensions
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
    

# bigram model
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        #each token reads logits for next tkn from lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets = None):
        B, T = idx.shape
        # idx and targets are (B, T) tensor of ints (Batch, Time, Channel)
        #in this context time represents the sequential nature of the data (block_size). Channel is representative of the logits for each token in the embedding table(vocab_size)
        tok_emb = self.token_embedding_table(idx) # (B,T,C) C=n_embd 
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T,C)
        x = tok_emb + pos_emb #(B,T,C) x holds the token and positional identity
        logits = self.lm_head(tok_emb) #(B,T, vocab_size)

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
            #get predictions
            logits, loss = self(idx)
            #focus on last time step (last element)
            logits = logits[:,-1,:] #becomes (B,C)
            #apply softmax
            probs = F.softmax(logits, dim=-1) #(B,C)
            #sample from prob distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            #append sampled idx to the sequence
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
        return idx
    
model = BigramLanguageModel(vocab_size)
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