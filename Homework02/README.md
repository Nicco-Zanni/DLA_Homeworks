# Homework02: The Transformative Transformer
This folder contains the files for the second laboaratory which focuses on the transformer architecture, DistilBert and CLIP.


## Main Files

### `DLA-Lab2.ipynb`

Main notebook of the laboratory containing data preparation, model configuration, training, evaluation, and result analysis.

### `precompute_embeddings.py`

Python scripts that downloads a dataset and a model, computes the images embeddings and save them togheter with the path to each image.

### `app.py`

This file contains the text-to-image retrieval Gradio application.


## Dependencies

Project dependencies are managed using **uv**.

Install the required packages with:

```bash
uv sync
```

## Using the Gradio App

Before launching the application, make sure the image embeddings have been computed and saved:

```bash
python compute_embeddings.py \
    --dataset jxie/flickr8k \
    --model openai/clip-vit-base-patch16
```

This command will:

1. Download the Flickr8k dataset.
2. Save all images inside `data/images/`.
3. Compute CLIP image embeddings.
4. Save the embeddings and image paths in `data/dataset_features.pt`.

### Launching the Application

Start the Gradio interface with:

```bash
python app.py \
    --features-path data/dataset_features.pt \
    --model openai/clip-vit-base-patch16
```

Once started, Gradio will display a local URL similar to:

```text
Running on local URL: http://127.0.0.1:7860
```

Open the URL in your browser to access the application.

## Information about the use of AI
I used LLM to develop this laboratory. In particular I used AI to help me code, debug and improve the models performance. I checked the results by looking at the documentation and trying the code in a notebook.
Here are some chat transcripts:  
https://chatgpt.com/share/6a3bb267-f658-83ed-b6e0-613c3090deb7   
https://gemini.google.com/share/db0b386e0cc4  

