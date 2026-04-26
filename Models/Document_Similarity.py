from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

Document=[
     "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.",
    
    "Deep learning is a subset of machine learning that uses neural networks with many layers to analyze complex patterns.",
    
    "Natural language processing enables computers to understand, interpret, and generate human language.",
    
    "Computer vision allows machines to interpret and make decisions based on visual data such as images and videos.",
    
    "Reinforcement learning is a type of machine learning where an agent learns by interacting with an environment and receiving rewards."

]

query="What is Machine Learning"

model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

doc_emb=model.encode(Document)
query_emb=model.encode(query)

cosine_score=cosine_similarity([query_emb],doc_emb)[0]
index,score=sorted(list(enumerate(cosine_score)),key=lambda x:x[1],reverse=True)[0]

print(cosine_score)
print(Document[index])
print(score)




