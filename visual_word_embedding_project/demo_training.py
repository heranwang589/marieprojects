from siamese_network import SiameseNetwork, ContrastiveLossFunction
from vwe_new import Model, Character
import torch
import torch.optim as optim
import pandas as pd
from vwe_demo_without_training import demo_model

# read training data

df = pd.read_csv('cc_similarity_pairs.csv')

# set the basics
net = SiameseNetwork()
criterion = ContrastiveLossFunction()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

# convert into tensors

data = []

for row in df.itertuples():
    character_A = Character(row.Character_A)
    character_B = Character(row.Character_B)
    label = row.Label

    character_A.generate_component_embeddings(demo_model)
    character_B.generate_component_embeddings(demo_model)

    c1 = torch.tensor(character_A.character_embedding, dtype=torch.float32)
    c2 = torch.tensor(character_B.character_embedding, dtype=torch.float32)
    l = torch.tensor(label, dtype=torch.float32)

    data_tuple = (c1, c2, l)
    data.append(data_tuple)

# Training loop

for epoch in range(100):
    running_loss = 0.0

    for i, (c1, c2, label) in enumerate(data):
        optimizer.zero_grad()
        output1, output2 = net(c1, c2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = running_loss / len(data)
        print(f"Epoch {epoch+1}/{100}, Loss: {avg_loss:.4f}")


model_path = 'siamese_model.pth'
torch.save(net.state_dict(), model_path)
print(f"Training complete! Model saved to {model_path}")