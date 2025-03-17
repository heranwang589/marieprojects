# The Visual Word Embedding Project

**Welcome!**

This is a Mandarin Chinese word embedding project created by Marie Wang. 

### Some Background
Traditionally, Word2vec are used to measure semantic similarity between words based on the contexts in which they appear in. However, this approach falls short when dealing with characters or words that are visually similar yet semantically unrelated. For example, in Mandarin Chinese, the characters "努" and “恕” looks similar but are used in entirely different contexts. A traditional Word2vec does not capture this visual similarity. Therefore, I created this word embedding project to measure the similarity in visual structure instead of similarity in semantic meaning in Mandarin Chinese characters. 

### Current Status
This project is still ***ongoing***. If you are a prospective employer or are interested, you can run a demo version that graphs 100 chinese characters based on their visual structure similarity. It should take a few seconds to run. 
Run the demo like this:
`cd visual_word_embedding_project`  
`python3 vwe_demo.py`  
If labels for nodes are not showing on the graph, it could be that you do not have the font Songti SC installed.

### How does this demo work?
The demo uses hanzi_chaizi from Xiaoquan Kong (howl-anderson) to decompose Mandarin Chinese characters into its sub-components. Word embeddings are then created for each sub-component through gensim Word2vec. Finally, word embeddings (more accurately character embeddings) are created for each character by taking an average of the embeddings of their sub-components. The resulting word embedding are then graphed, with the characters that appear similar(i.e has similar sub-components) near to each other. 

### To-Do
1. Fine-tune with supervised machine learning
2. Alternative model that generates embeddings from Image-based CNN to capture pixel-level similarity 