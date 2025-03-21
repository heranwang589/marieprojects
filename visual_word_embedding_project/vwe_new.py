import numpy as np
import pandas as pd
from hanzi_chaizi import HanziChaizi
from gensim.models import Word2Vec
import torch

# decompose all the characters

hc = HanziChaizi()

def decompose(c: str, visited=None) -> set[str]:
    """
    Decompose single chinese characters into its components.

    Args:
        c: the character to decompose
    
    Returns(set[str]): a set of all the decompositions of the character
    """

    if visited is None:
        visited = set()

    if c in visited:
        return set()

    visited.add(c)

    try:
        decomposition = hc.query(c)
    except TypeError:
        return {c}

    decompositions = set(decomposition)

    for part in decomposition:
        decompositions.update(decompose(part, visited))

    return decompositions

class Character:

    def __init__(self, character):
        self._character = character
        self._components = set()
        self._components_embeddings = []
    
    @property
    def representation(self) -> str:
        return self._character
    
    @property
    def components(self) -> list[str]:
        self._components = list(decompose(self._character))
        return self._components

    def generate_component_embeddings(self, model: Word2Vec) -> None:
        for component in self.components:
            self._components_embeddings.append(model.wv[component])
    
    @property
    def character_embedding(self):
        character_embedding = np.mean(self._components_embeddings, axis=0)
        return character_embedding
    
class Model:

    def __init__(self, c_list: list[str]) -> None:
        self._c_list = c_list
        self._character_list = []

    def transform_into_characters(self) -> None:
        for c in self._c_list:
            self._character_list.append(Character(c))
        return self._character_list

    def generate_token(self) -> list[list[str]]:
        token_list = []
        for character in self.transform_into_characters():
            token_list.append(character.components)
        return token_list
    
    def create_model(self) -> Word2Vec:
        return Word2Vec(sentences=self.generate_token(), vector_size=100, window=5, min_count=1, workers=4, sg=1)
