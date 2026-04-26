from sentence_transformers import SentenceTransformer

sentence=["My Name is Nikhil"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding=model.encode(sentence)
print(embedding)