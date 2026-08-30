from app.core.rag_chain import rag_chain

def user_query(question : str):
    result = rag_chain.invoke(question)
    return result