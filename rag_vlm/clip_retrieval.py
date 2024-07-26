import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import torch
import open_clip

import numpy as np
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d


class RAGWithCLIP:

    def __init__(self, model_name, source='huggingface'):

        self.source = source

        if self.source == 'huggingface':
            self.model = CLIPModel.from_pretrained(model_name).to("cuda").half()
            self.processor = CLIPProcessor.from_pretrained(model_name)

        elif self.source == 'open_clip':
            model, pretrained = model_name.split("/")
            self.model, _, self.processor = open_clip.create_model_and_transforms(model, pretrained=pretrained)
            self.model = self.model.to("cuda").half().eval()
            self.tokenizer = open_clip.get_tokenizer(model)

        else:
            raise ValueError("Invalid source. Please choose either 'huggingface' or 'open_clip'")
        

        self.frame_features = None
        self.frames = None
        self.frame_paths = None


    @torch.no_grad()
    def get_image_features(self, image):

        if self.source == 'huggingface':
            inputs = self.processor(images=image, return_tensors="pt").to("cuda").half()
            features = self.model.get_image_features(**inputs)
            
        elif self.source == 'open_clip':
            inputs = torch.stack([self.processor(img).to("cuda").half() for img in image]).to("cuda").half()
            features = self.model.encode_image(inputs).float()

        return features / features.norm(dim=-1, keepdim=True)
        
    
    @torch.no_grad()
    def get_text_features(self, text):
        
        if self.source == 'huggingface':
            inputs = self.processor(text=text, return_tensors="pt", padding=True).to("cuda").half()
            features = self.model.get_text_features(**inputs)

        elif self.source == 'open_clip':
            inputs = self.tokenizer(text).to("cuda")
            features = self.model.encode_text(inputs).float()

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
    
    def search(self, query, top_k=1, window=0, sigma=0, do_visualization=False):
        text_feature = self.get_text_features([query])

        if self.frame_features is None:
            raise ValueError("Please encode the frames first using encode_frames method")
        
        scores = ((self.frame_features @ text_feature.T).T).cpu().numpy()[0]


        def smoothed_topk_indices(scores, topk, window, sigma):
            scores_smoothed = scores.copy()
            if window > 0:
                scores_smoothed = np.convolve(scores_smoothed, np.ones(window)/window, mode="same")
            if sigma > 0:
                scores_smoothed = gaussian_filter1d(scores_smoothed, sigma=2)
            
            # add zeros to the scores to make sure the local maxima are not at the edges
            scores_smoothed = np.concatenate([np.zeros(1), scores_smoothed, np.zeros(1)])
            local_maxima_indices = argrelextrema(scores_smoothed, np.greater)[0]
            local_maxima_scores = scores_smoothed[local_maxima_indices]

            if len(local_maxima_scores) < topk:
                top_indices = local_maxima_indices
            else:
                top_indices = local_maxima_indices[np.argsort(local_maxima_scores)[-topk:]]
            
            # large to small
            top_indices = np.sort(top_indices)[::-1] - 1
            scores_smoothed = scores_smoothed[1:-1]
            top_scores = scores_smoothed[top_indices]
            return top_indices, top_scores, scores_smoothed
        

        top_indices, top_scores, scores_smoothed = smoothed_topk_indices(
            scores, top_k, window=window, sigma=sigma)

        result = [self.frame_paths[i] for i in top_indices]
        
        if do_visualization:
            plt.figure(figsize=(20, 3))
            plt.plot(scores, color="blue", linewidth=1, alpha=0.5)
            plt.plot(scores_smoothed, color="green")

            for j in range(len(top_indices)):
                plt.scatter(top_indices[j], top_scores[j], color="red")
                plt.text(top_indices[j], top_scores[j]+0.005, f"No. {top_indices[j]}")
            plt.xlabel("Frames")
            plt.ylabel("CLIP Similariy")
            plt.grid()
            plt.title(f"Query: {query}")
            plt.show()

            plt.figure(figsize=(20, 3))
            for j in range(len(top_indices)):
                plt.subplot(1, top_k, j + 1)
                plt.imshow(self.frames[top_indices[j]])
                plt.title(f"Top {j+1} frame (No. {top_indices[j]}), similarity {top_scores[j]:.3f}")
                plt.axis("off")
            plt.tight_layout()
            plt.show()

        return result