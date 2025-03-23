import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from siamese_network import SiameseNetwork
from vwe_new import Model

# load the Siamese Network

net = SiameseNetwork()
net.load_state_dict(torch.load('siamese_model.pth'))
net.eval()

# load all the demo characters

file = open('100_unique_chinese_characters.txt', 'r')
all_characters = file.readlines()
file.close()
all_characters = [line.replace('\n', '') for line in all_characters]

# create original embedding model

model = Model(all_characters)
model.generate_token()
demo_model = model.create_model()

if __name__ == "__main__":

    # create siamese embeddings

    siamese_embeddings = []
    siamese_embed_keys = []

    with torch.no_grad():
        for c in model.transform_into_characters():
            c.generate_component_embeddings(demo_model)
            torch_embedding = torch.tensor(c.character_embedding, dtype=torch.float32)
            siamese_embedding = net.forward_once(torch_embedding)
            siamese_embeddings.append(siamese_embedding)
            siamese_embed_keys.append(c.representation)

    # dimensionality reduction

    all_character_embeddings = np.array(siamese_embeddings)
    reducer = umap.UMAP(n_components=2, min_dist=3, spread=3.0)
    embeddings_2d = reducer.fit_transform(siamese_embeddings)

    # create embedding dictionary

    siamese_embed_dict = {}

    for i in range(len(siamese_embed_keys)):
        siamese_embed_dict[siamese_embed_keys[i]] = tuple(embeddings_2d[i])

    # visualization

    for character in siamese_embed_keys:
        coordinates = siamese_embed_dict[character]
        plt.scatter(coordinates[0], coordinates[1])
        plt.annotate(character, (coordinates[0], coordinates[1]), fontsize=12,
                    fontfamily='Songti SC',
                    xytext=(5, 5),
                    textcoords='offset points')
        
    plt.title("Chinese Character Embedding Visualization With Training", fontsize=16, fontweight='bold')
        
    plt.grid(True)
    plt.tight_layout()
    plt.show()

