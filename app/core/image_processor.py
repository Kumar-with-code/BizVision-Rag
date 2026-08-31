import io
import fitz
import base64
from PIL import Image


def extract_images_from_pdf(file_path : str):
    docs = fitz.open(file_path)

    all_images = []
    for page_num, page in enumerate(docs):

        images_list = page.get_images(full=True)

        for img_index, img_info in enumerate(images_list):
            xref = img_info[0]
            base_image = docs.extract_image(xref)
            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            img_buf = io.BytesIO()
            image.save(img_buf, format='PNG')
            img_buf.seek(0)

            base64_image = base64.b64encode(img_buf.getvalue()).decode('utf-8')

            all_images.append({
                "image_id": f"page_{page_num}_img_{img_index}",
                "page": page_num,
                "image" : image,
                "base64": base64_image
            })

    docs.close()
    return all_images