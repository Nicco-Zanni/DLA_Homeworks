import argparse
import gradio as gr
import torch
from transformers import CLIPModel, CLIPProcessor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", type=str, default="data/dataset_features.pt",
        help="Path to the saved features file")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch16",
        help="Slug of huggingface Clip Model")
    return parser.parse_args()

class ImageRetrievalApp:
    def __init__(self, features_path: str, model_name: str):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        print(f"Loading data from {features_path}...")
        
        data = torch.load(features_path, map_location=self.device)
        self.image_embeddings = data["embeddings"].to(self.device)
        self.image_paths = data["paths"]

        print(f"Loading model {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def search(self, query: str, k: int):
        
        if not query.strip():
            return []

        inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs).pooler_output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
       
        scores = torch.matmul(self.image_embeddings, text_features.T).squeeze(1)
    
        topk_scores, topk_indices = torch.topk(scores, k=int(k))
        
        gallery_output = []
        for score, idx in zip(topk_scores, topk_indices):
            img_path = self.image_paths[idx.item()]
            caption = f"Score: {score.item():.4f}"
            gallery_output.append((img_path, caption))
            
        return gallery_output

    def build_ui(self):
        
        with gr.Blocks(title="CLIP Text-to-Image Retrieval") as demo:
            gr.Markdown("# 🔍 CLIP Text-to-Image Retrieval")
            gr.Markdown(f"Active Model in background: `{self.model_name}`")
            
            with gr.Column():
                with gr.Row():
                    query_input = gr.Textbox(
                        label="What are you searching?", 
                        placeholder="Es. A dog catching a frisbee in the air...",
                        lines=2,
                        scale=3
                    )
                    with gr.Column(scale=1):
                        k_input = gr.Slider(
                            minimum=1, maximum=20, step=1, value=10, 
                            label="Images to show (K)"
                        )
                        search_button = gr.Button("Search", variant="primary")
                    
                with gr.Row():
                    gallery_output = gr.Gallery(
                        label="Most similar images found", 
                        columns=3, rows=2, object_fit="contain", height="auto"
                    )
                    
            search_button.click(fn=self.search, inputs=[query_input, k_input], outputs=gallery_output)
            query_input.submit(fn=self.search, inputs=[query_input, k_input], outputs=gallery_output)
            
        return demo

if __name__ == "__main__":
    args = parse_args()
    
    app = ImageRetrievalApp(features_path=args.features_path, model_name=args.model)
    
    ui = app.build_ui()
    
    ui.launch(share=False)