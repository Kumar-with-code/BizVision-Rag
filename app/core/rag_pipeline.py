import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq

from app.core.embedding import get_embedding_model
from app.core.vectorstore import vector_db, img_vector_db
from app.core.doc_processor import load_and_split_documents
from app.core.image_processor import extract_images_from_pdf
from app.core.image_embedding import CLIPImageEmbedding

# Groq Model
os.getenv('GROQ_API_KEY')
llm = ChatGroq(model='qwen/qwen3.6-27b')


# Text Embeddings 
docs = load_and_split_documents("app/core/sample_Business_Vision.pdf")
embeddings = get_embedding_model()
db = vector_db(docs, embeddings)
retriever = db.as_retriever()


# Image Embeddings
img_docs = extract_images_from_pdf("app/core/sample_Business_Vision.pdf")
img_embedding_model = CLIPImageEmbedding()

img_embeddings = []
for image in img_docs:
    vector = img_embedding_model.embed_image(image["image"])
    img_embeddings.append(vector)
imgdb = img_vector_db(img_docs, img_embeddings)


# RAG Pipeline
def rag_pipeline(question: str):

    text_docs, image_docs = retrieve_multimodal(question)

    content = []

    # Question
    content.append({
        "type": "text",
        "text": f"Question: {question}"
    })

    # Retrieved text
    if text_docs:
        text_context = "\n\n".join(
            [
                f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
                for doc in text_docs
            ]
        )

        content.append({
            "type": "text",
            "text": f"Retrieved Text:\n{text_context}"
        })

    # Retrieved images
    for doc in image_docs:

        image_id = doc.metadata.get("image_id")

        # Find the actual image 
        image_data = next(
            (
                image
                for image in img_docs
                if image["image_id"] == image_id
            ),
            None
        )

        if image_data:
            content.append({
                "type": "text",
                "text": f"Image from page {image_data['page']}"
            })

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data['base64']}"
                }
            })

    content.append({
        "type": "text",
        "text": (
            "Answer the question using the retrieved text and images. "
            "Analyze the images when it contain relevant information."
        )
    })

    message = HumanMessage(content=content)

    response = llm.invoke([message])

    return response.content


def retrieve_multimodal(question):
    text_docs = retriever.invoke(question)
    query_vector = img_embedding_model.embed_text(question)
    image_docs = imgdb.similarity_search_by_vector(query_vector, k=3)
    return text_docs, image_docs


if __name__ == '__main__':
    print(rag_pipeline("Based on the document and its visual content, what are the key business development insights, and how do the charts or images support those insights?"))