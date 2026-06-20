from datasets import load_dataset, concatenate_datasets, get_dataset_split_names
from transformers import CLIPProcessor, CLIPModel
import torch
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="jxie/flickr8k",
        help="Slug of huggnface dataset")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch16",
        help="Slug of huggnface Clip Model")
    parser.add_argument("--batch-size", type=int, default=64,
        help="batch size used to compute image embeddings")
    
    args = parser.parse_args()

    return args

def compute_embeddings(ds, model, processor, device, batch_size= 64):

    os.makedirs("data/images", exist_ok=True)

    image_paths = []
    
    print("Saving Images")
    for i, sample in enumerate(ds):

        path = f"data/images/{i:06d}.png"

        sample["image"].save(path)

        image_paths.append(path)

    print("Dataset saved")

    model.eval()

    embeddings = []

    print("Computing embeddings")
    with torch.no_grad():

        for i in range(0, len(ds), batch_size):
            print(f'Images: {i}/{len(ds)}')
            batch = ds[i:i+batch_size]

            images = batch["image"]

            inputs = processor(images=images, return_tensors="pt",).to(device)

            feats = model.get_image_features(**inputs).pooler_output

            feats = feats / feats.norm(dim=-1, keepdim=True)

            embeddings.append(feats.cpu())

    embeddings = torch.cat(embeddings, dim=0)

    dataset_dict = { "embeddings": embeddings, "paths": image_paths}
    
    torch.save(dataset_dict, "data/dataset_features.pt")


def main():

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = get_dataset_split_names(args.dataset)
    dataset_list = [load_dataset(args.dataset, split=split) for split in splits]
    ds = concatenate_datasets(dataset_list)

    model = CLIPModel.from_pretrained(args.model)
    model.to(device)
    processor = CLIPProcessor.from_pretrained(args.model)

    compute_embeddings(ds, model, processor, device, args.batch_size)


if __name__ == "__main__":
    main()