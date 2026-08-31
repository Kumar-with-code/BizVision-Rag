from app.core.rag_pipeline import rag_pipeline


def user_query(question : str):
    result = rag_pipeline.invoke(question)
    return result