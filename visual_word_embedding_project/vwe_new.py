import numpy as np
from hanzi_chaizi import HanziChaizi
from gensim.models import Word2Vec
from strokes import strokes

# decompose all the characters

hc = HanziChaizi()

def is_character(c: str) -> bool:
    """
    check if a character is a real character or a stroke
    """
    return strokes(c) > 1

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

def clean_components(components: list[str]) -> list[str]:
    """
    retain only relavant components of a character
    """
    clean_components = set()
    for cp in components:
        if not is_character(cp):
            continue
        else:
            clean_components.add(cp)

    if clean_components == set():
        return components
    
    return clean_components

class Character:
    """
    a class for representating individual characters
    """
    _characters: str
    _components: list[str]
    _components_embeddings: list[np.ndarray]

    def __init__(self, character: str) -> None:
        self._character = character
        self._components = list(clean_components(decompose(character)))
        self._components_embeddings = []
    
    @property
    def representation(self) -> str:
        """
        return what the character is
        """
        return self._character
    
    @property
    def components(self) -> list[str]:
        """
        return the hanzi_chaizi components of the character
        """
        return self._components

    def generate_component_embeddings(self, model: Word2Vec) -> None:
        """
        generate embeddings for the components of the character
        """
        self._components_embeddings = []
        for component in self._components:
            self._components_embeddings.append(model.wv[component])
    
    @property
    def character_embedding(self) -> np.ndarray:
        """
        return the embedding for the character itself by averaging the component
        embeddings
        """
        character_embedding = np.mean(self._components_embeddings, axis=0)
        return character_embedding
    
class Model:
    """
    a class for representing a Word2Vec embedding model
    """
    _c_list: list[str]
    _character_list: list[Character]

    def __init__(self, c_list: list[str]) -> None:
        self._c_list = c_list
        self._character_list = []

    def transform_into_characters(self) -> None:
        """
        transform the strings in the list of strings into a characters in a list
        of characters
        """
        for c in self._c_list:
            self._character_list.append(Character(c))
        return self._character_list

    def generate_token(self) -> list[list[str]]:
        """
        tokenize the characters in the character list into their components
        """
        token_list = []
        for character in self.transform_into_characters():
            token_list.append(character.components)
        return token_list
    
    def create_model(self) -> Word2Vec:
        """
        create the word2Vec model
        """
        return Word2Vec(sentences=self.generate_token(), vector_size=100, window=5, min_count=1, workers=4, sg=1)
