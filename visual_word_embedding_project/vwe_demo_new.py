import umap
import numpy as np
import matplotlib.pyplot as plt
import vwe_new

# load all the demo characters

file = open('100_unique_chinese_characters.txt', 'r')
all_characters = file.readlines()
file.close()
all_characters = [line.replace('\n', '') for line in all_characters]

if __name__ == "__main__":
    model = vwe_new.Model(all_characters)
    model.generate_token()
    demo_model = model.create_model()

    all_character_embeddings = []
    character_embed_keys = []

    for character in model.transform_into_characters():
        character.generate_component_embeddings(demo_model)
        all_character_embeddings.append(character.character_embedding)
        character_embed_keys.append(character.representation)
    
    all_character_embeddings = np.array(all_character_embeddings)
    reducer = umap.UMAP(n_components=2, min_dist=3, spread=3.0)
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