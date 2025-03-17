import vwe
import umap
import random
import numpy as np
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

# set random seeds
np.random.seed(1)
random.seed(1)

# load all the demo characters

file = open('100_unique_chinese_characters.txt', 'r')
all_characters = file.readlines()
file.close()
all_characters = [line.replace('\n', '') for line in all_characters]

if __name__ == "__main__":
    # intialize
    components_dict = vwe.sequences(all_characters)
    token_list = vwe.generate_token(components_dict)

    # train model
    model = Word2Vec(sentences=token_list, vector_size=100, window=5, min_count=1, workers=4, sg=1, seed=1)

    # generate character embeddings
    all_character_embeddings = []
    character_embed_keys = []

    for c, c_components in components_dict.items():
        components = components_dict[c]
        character_embedding = vwe.single_character_embeddings(c, components, model)
        all_character_embeddings.append(character_embedding)
        character_embed_keys.append(c)
    
    # convert to 2-d
    all_character_embeddings = np.array(all_character_embeddings)
    reducer = umap.UMAP(n_components=2, random_state=1)
    embeddings_2d = reducer.fit_transform(all_character_embeddings)

    character_embed_dict = {}

    for i in range(len(character_embed_keys)):
        character_embed_dict[character_embed_keys[i]] = tuple(embeddings_2d[i])

    # visualization
    for character in character_embed_keys:
        coordinates = character_embed_dict[character]
        plt.scatter(coordinates[0], coordinates[1])
        plt.annotate(character, (coordinates[0], coordinates[1]), fontsize=12,
                     fontfamily='Songti SC',
                     xytext=(5, 5),
                     textcoords='offset points')
    
        plt.title("Chinese Character Embedding Visualization", fontsize=16, fontweight='bold')
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()