import os
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

class CLIPImageEmbedding:

    def __init__(self):

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()

    def embed_image(self, image: Image.Image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self.clip_processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = self.clip_model.get_image_features(**inputs)
            features = output.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)
            
            return features.squeeze().cpu().numpy()

    def embed_text(self, text: str):

        inputs = self.clip_processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )

        with torch.no_grad():
            output = self.clip_model.get_text_features(**inputs)
            features = output.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)

            return features.squeeze().cpu().numpy()