import torch
import torch.nn as nn
import matplotlib.pyplot as plt

BATCH_SIZE = 32
CONTEXT_LENGTH = 3
EMBEDDING_SIZE = 15
# VOCAB_SIZE = len(chars)
VOCAB_SIZE = 28
EPOCHS = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("names.txt") as f:
    text = f.read()
    names = text.lower().split("\n")

class n_gram_mlp(nn.Module):
    def __init__(self, batch_size: int, context_length: int, embedding_size: int, vocab_size: int):
        super().__init__()
        self.context_length  = context_length   
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size    
        self.char_embedding = nn.Parameter(torch.randn((vocab_size,embedding_size), device = device))

        self.layer1 = nn.Linear(self.context_length*self.embedding_size, 256)
        self.layer2 = nn.Linear(256,128)
        self.out_layer = nn.Linear(128, self.vocab_size)
        self.relu = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
    def forward(self, x: torch.tensor) -> torch.tensor :
        out = torch.matmul(x, self.char_embedding).view(x.shape[0],-1)
        out = self.layer1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.out_layer(out)
        return out


chars = sorted(list(set(''.join(names))))
stoi = {ch:i+1 for i,ch in enumerate(chars)}
stoi['<S>'] = 0
stoi['<E>'] = len(stoi)
itos = {i:ch for ch,i in stoi.items()}

fitos = lambda x: ''.join([itos[i] for i in x]) 
fstoi = lambda s: [stoi[i] for i in s] 

block_size = CONTEXT_LENGTH
X = []
Y = []
for name in names:
    word = ["<S>"]*block_size + list(name) + ["<E>"]*block_size
    for i in range(block_size,len(word)):
        x = word[i-block_size:i]
        y = word[i]
        ix = [stoi[i] for i in x]
        iy = stoi[y]
        X.append(ix)
        Y.append(iy)

X = torch.tensor(X).to(device)
Y = torch.tensor(Y).to(device)

# for i,j in zip(X.tolist(),Y.tolist()):
#     print(fitos(i),"->",itos[j])
Y_train = Y[:int(0.9*len(Y))].to(device)
X_train = X[:int(0.9*len(X))].to(device)
Y_test = Y[int(0.9*len(Y)):].to(device)
X_test = X[int(0.9*len(X)):].to(device)

def train():
    #TRAIN
    model = n_gram_mlp(batch_size = BATCH_SIZE, context_length = CONTEXT_LENGTH, embedding_size = EMBEDDING_SIZE, vocab_size = VOCAB_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr =0.01)
    criterion = nn.CrossEntropyLoss()
    losses = []
    tot_c = 0
    corr_c = 0 

    for _ in range(EPOCHS):
        batch_ix = torch.randint(0,len(X_train),(BATCH_SIZE,))
        batch = nn.functional.one_hot(X_train[batch_ix], num_classes = VOCAB_SIZE).float().to(device)

        out = model.forward(batch)
        loss = criterion(out, Y_train[batch_ix])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if(_%1000==0): 
            print(_, loss.item())

    #EVALUATE
    model.eval()
    with torch.no_grad():
        x1 = nn.functional.one_hot(X, num_classes = VOCAB_SIZE).float().to(device)
        x2 = nn.functional.one_hot(X_test, num_classes = VOCAB_SIZE).float().to(device)

        out1 = model.forward(x1)
        loss = criterion(out1, Y)
        out2 = model.forward(x2)
        val_loss = criterion(out2, Y_test)
        
        print(f"Total Loss: {loss}, Validation Loss: {val_loss}")

        torch.save({'model_state_dict': model.state_dict(),
         'loss': val_loss,
         }, "model.pt")

    # plt.plot(losses)
    # plt.ylabel("LOSS")
    # plt.show()



# GENERATE
def generate(no_of_names):
    inference_model = n_gram_mlp(batch_size = 1, context_length = CONTEXT_LENGTH, embedding_size = EMBEDDING_SIZE, vocab_size = VOCAB_SIZE).to(device)
    inference_model.load_state_dict(torch.load('model.pt')['model_state_dict'])

    inference_model = inference_model.to(device)

    inference_model.eval()
    with torch.no_grad():
        for i in range(no_of_names):
            s = ['<S>']*CONTEXT_LENGTH
            while True:
                x = torch.tensor(fstoi(s[-CONTEXT_LENGTH:]), dtype=torch.long).unsqueeze(0).to(device)
                x = nn.functional.one_hot(x, num_classes = VOCAB_SIZE).float()
                x = x.to(device)
                out = inference_model.forward(x)
                # print(out, out.shape)
                temperature = 1
                prob = torch.nn.functional.softmax(out/temperature ,dim=1)

                iout = torch.multinomial(prob, num_samples = 1, replacement = True)[0].item()
                s.append(itos[iout])
                if(s[-1]=="<E>"):
                    break
            print(''.join(s[CONTEXT_LENGTH:-1]))

train()
generate(no_of_names = 50)

