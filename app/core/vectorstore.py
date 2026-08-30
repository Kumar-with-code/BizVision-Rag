from langchain_community.vectorstores import FAISS

def vector_db(chunks, embeddings):
    db = FAISS.from_documents(chunks, embeddings)
    print("Vector Database Created Successfully!")
    return db

