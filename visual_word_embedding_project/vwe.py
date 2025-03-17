import numpy as np
from hanzi_chaizi import HanziChaizi
from gensim.models import Word2Vec

# decompose all the characters

hc = HanziChaizi()

def decompose(c: str, visited=None) -> set[str]:
    """
    Decompose single chinese characters into its components.

    Args:
        c: the character to decompose
    
    Returns(set[str]): a set of all the decompositions of the character
    """
    # possible problem: how decpomosed do you want it? right now it is giving
    # absolute most amount of possible decompositions

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

def sequences(c_list: list[str]) -> dict[str, set[str]]:
    """
    Decompose chinese characters into its components.

    Args:
        c_list(list[str]): list of all the characters
    
    Returns(dict[str, str]): a dictionary mapping the character to its minimum
    decomposed parts. e.g [努: 女、又、力]

    """
    all_decomps = {}

    for c in c_list:
        all_decomps[c] = decompose(c)
    
    return all_decomps

# tokenize decompositions

def generate_token(cd: dict[str, set[str]]) -> list[list[str]]:
    """
    Convert dictionary of characters and their decompositions into a list
    of lists for training word embeddings
    """
    token_list = []
    for decomposition in cd.values():
        token_list.append(list(decomposition))
    
    return token_list

# get embeddings for character based on visual structure

def single_character_embeddings(c: str, c_components: set[str], model: Word2Vec):
    all_comp_embeddings = []
    for component in c_components:
        all_comp_embeddings.append(model.wv[component])
    character_embedding = np.mean(all_comp_embeddings, axis=0)
    return character_embedding
