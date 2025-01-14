#creates a streamlined script for the whole process
import torch
import torch.nn as nn 
from torch.nn import functional as F

#hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
#-------------------------------------------------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f: #extracts training data
    text = f.read()

#Organizes the vocabulary (in this case unique characters, rather than words)
chars = sorted(list(set(text)))
vocab_size = len(chars)

#Tokenizing the input text, encoding individual characters into vectors or integers
string_to_token = { ch: i for i, ch in enumerate(chars)} #creates dictionary with character as key and token as value
token_to_string = {i: ch for i, ch in enumerate(chars)} #creates dictionary with token as key and character as value

encode = lambda s : [string_to_token[c] for c in s] #given a string, give the corresponding series of tokens
decode = lambda l : ''.join([token_to_string[i] for i in l])#given a series of tokens, return the strings (concats into a single string)

data = torch.tensor(encode(text), dtype = torch.long) # the data tensor is tikenized in its entirety
#splitting into training and validation sets, 90% for training
n = int(0.9*len(data)) #casts 0.9*len(data) to int
train_data = data[:n] #indexes 90% as training
val_data = data[n:] #the rest is validation, checking for overfitting

def get_batch(split):
    #generates batch of inputs x and targets y
    data = train_data if split =='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() #indicates no backprop will be done
def estimate_loss():
    out = {}
    model.eval() #switches on eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() #switches on train mode
    return out

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    #every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

