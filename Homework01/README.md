# Homework01: From Pixels to Semantics

This folder contains the files for the first laboaratory which focuses on CNN, image classification and object detection.


## Main Files

### `DLA-Lab1.ipynb`

Main notebook of the laboratory containing data preparation, model configuration, training, evaluation, and result analysis.

### `utils.py`

Utility module containing classes and helper functions used throughout the notebook. In particular, it provides reusable components for fine-tuning and evaluating the models used in the notebook.

### `configResNet.yaml`

Configuration file containing the hyperparameters and settings used for the ResNet-based classification experiments.

### `FasterRCNN_config.yaml`

Configuration file containing the parameters used for the Faster R-CNN object detection experiments.

### `Results`

The `results` directory contains the outputs generated during the evaluation phase on the test set for the classification model finetuned using different training configurations.

## Dependencies

Project dependencies are managed using **uv**.

Install the required packages with:

```bash
uv sync
```

## Information about the use of AI
I used LLM to develop this laboratory. In particular I used AI to help me code, debug and improve the models performance. I checked the results by looking at the documentation and trying the code in a notebook.
Here are some chat transcripts:  
https://chatgpt.com/share/6a3ba8cd-3270-83eb-84a0-517807a1d364   
https://chatgpt.com/share/6a3baa58-67dc-83ed-9ae0-4a20fa12df5e  
https://claude.ai/share/62c3b726-2775-47be-84a9-70de960d32a8
https://claude.ai/share/306e2096-3467-4441-84c0-fefbb4bb1d23

