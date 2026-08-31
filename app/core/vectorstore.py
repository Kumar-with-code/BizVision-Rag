from langchain_community.vectorstores import FAISS
from app.core.image_processor import extract_images_from_pdf
from app.core.image_embedding import CLIPImageEmbedding

def vector_db(chunks, embeddings):
    db = FAISS.from_documents(chunks, embeddings)
    print("Text Vector Database Created Successfully!")
    return db


def img_vector_db(images, image_embeddings):

    image_data = []
    metadatas = []

    for image, embedding in zip(images, image_embeddings):

        image_data.append((image["image_id"], embedding))
        metadatas.append({
            "image_id": image["image_id"],
            "page": image["page"],
            "type": "image"
        })

    db = FAISS.from_embeddings(text_embeddings=image_data, embedding=None, metadatas=metadatas)
    return db












# images = extract_images_from_pdf("app/core/sample_Business_Vision.pdf")
# clip = CLIPImageEmbedding()

# image_embeddings = []

# for image in images:
#     embedding = clip.embed_image(image["image"])
#     image_embeddings.append(embedding)

# print("Images:", len(images))
# print("Embeddings:", len(image_embeddings))
# print("Vector shape:", image_embeddings[0].shape)