from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()
from app.core.embedding import get_embedding_model
from app.core.vectorstore import vector_db
from app.core.doc_processor import load_and_split_documents



os.getenv('GROQ_API_KEY')


llm = ChatGroq(model='openai/gpt-oss-20b')

docs = load_and_split_documents("app/core/sample_Business_Vision.pdf")
embeddings = get_embedding_model()
db = vector_db(docs, embeddings)
retriever = db.as_retriever()

def rag_chain(question : str):

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context.
    Context: {context}
    Question: {input}
    """)

    docs_combining = create_stuff_documents_chain(
        llm=llm, 
        prompt=prompt
    )

    chain = create_retrieval_chain(
        retriever, 
        docs_combining
    )

    response = chain.invoke({
        'input' : question
    })

    return response['answer']

print("RAG Chain Created Successfully!")


if __name__ == '__main__':
    print(rag_chain('How can i enhance my skills on business development'))