import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import torch


class RAGWithCLIP:

    def __init__(self, hf_model_name):
        self.model = CLIPModel.from_pretrained(hf_model_name).to("cuda")
        self.processor = CLIPProcessor.from_pretrained(hf_model_name)

        self.frame_features = None
        self.frames = None
        self.frame_paths = None

    @torch.no_grad()
    def get_image_features(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to("cuda")
        features = self.model.get_image_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True)
    
    @torch.no_grad()
    def get_text_features(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to("cuda")
        features = self.model.get_text_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True)
    
    def encode_frames(self, frames, batch_size=32, show_progress=True, load_frames_to_memory=True):
        self.frame_paths = frames
        frames = [Image.open(frame) for frame in frames]
        self.frames = frames if load_frames_to_memory else None

        features = []
        for i in tqdm(range(0, len(frames), batch_size), disable=not show_progress):
            batch_frames = frames[i:i + batch_size]
            features.append(self.get_image_features(batch_frames))
        self.frame_features = torch.cat(features)#.cpu().numpy()

        return self.frame_features
    
    def search(self, queries, n=1, do_visualization=False):
        text_features = self.get_text_features(queries)

        if self.frame_features is None:
            raise ValueError("Please encode the frames first using encode_frames method")
        
        scores = (self.frame_features @ text_features.T).T
        top_scores, top_indices = torch.topk(scores, n, dim=1)
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        results = []
        for i, query in enumerate(queries):
            result = []
            for j in range(n):
                result.append(self.frame_paths[top_indices[i, j]])
            results.append(result)
        
        if do_visualization:
            plt.figure(figsize=(20, 3))
            plt.plot(scores[0].cpu().numpy())

            for j in range(n):
                plt.scatter(top_indices[0, j], top_scores[0, j], color="red")
                plt.text(top_indices[0, j], top_scores[0, j], f"No. {top_indices[0, j]}")
            plt.xlabel("Frames")
            plt.ylabel("CLIP Similariy")
            plt.grid()
            plt.title(f"Query: {queries[0]}")
            plt.show()

            for i, query in enumerate(queries):
                plt.figure(figsize=(20, 3))
                for j in range(n):
                    plt.subplot(1, n, j + 1)
                    plt.imshow(self.frames[top_indices[i, j]])
                    plt.title(f"Top {j+1} frame (No. {top_indices[i, j]}), similarity {top_scores[i, j]:.3f}")
                    plt.axis("off")
                plt.tight_layout()
                plt.show()

        return results