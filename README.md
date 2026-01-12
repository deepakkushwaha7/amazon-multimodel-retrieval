# Multimodal Representation Learning (Text + Image)

This repository contains a **multimodal representation learning pipeline** that learns joint embeddings from **text and images** using a **multi-tower neural network** and **contrastive learning**.

The project supports:

* Training a multimodal model
* Encoding queries and items into a shared embedding space
* Performing semantic similarity search
* Running an interactive demo UI

All components are modular and easy to extend.

---

## Project Structure

```
.
├── data_loader.py     # Dataset loader (CSV + image handling)
├── models.py          # Multi-tower neural network + loss
├── train.py           # Training script
├── demo.py            # Interactive search demo (Gradio)
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
└── .gitignore
```

---

## Architecture Overview

```
           Text Inputs                    Image Inputs
        ┌──────────────┐             ┌────────────────┐
        │ Query Text    │             │ Product Image  │
        └──────┬───────┘             └───────┬────────┘
               │                               │
               ▼                               ▼
     ┌──────────────────┐         ┌────────────────────┐
     │ Text Encoder     │         │ Vision Encoder     │
     │ (Transformer)   │         │ (Vision Transformer)│
     └────────┬─────────┘         └─────────┬──────────┘
              │                               │
              ├───────────┬───────────────────┤
              ▼           ▼                   ▼
        ┌────────────────────────────────────────┐
        │        Fusion + Projection Layer        │
        │        (Shared 512-dim space)           │
        └────────────────────────────────────────┘
                            │
                            ▼
               Contrastive Learning Objective
```

---

## Requirements

* Python **3.9+**
* CUDA GPU recommended (CPU works but slower)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Format

The dataset is expected to be a **CSV file** with (at minimum) the following columns:

| Column Name   | Description                |
| ------------- | -------------------------- |
| `title`       | Item title text            |
| `description` | Item description text      |
| `image_url`   | URL of an image (optional) |
| `id`          | Unique item identifier     |

Images are downloaded automatically and cached locally.

---

## Training the Model

Run the training script:

```bash
python train.py --data path/to/data.csv
```

### Optional Arguments

```bash
--epochs        Number of training epochs (default: 2)
--batch_size    Batch size (default: 8)
--max_samples   Limit number of samples (default: 2000)
```

Example:

```bash
python train.py --data data/products.csv --epochs 3 --batch_size 16
```

### Output

After training, the model weights are saved as:

```
model_weights.pt
```

---

## Running the Demo

The demo provides an **interactive semantic search interface**.

Start the demo:

```bash
python demo.py
```

This launches a **Gradio web UI** in your browser where you can:

* Enter a text query
* Retrieve the most similar items using multimodal embeddings

---

## How Search Works

1. All items are encoded once into embeddings
2. A query is encoded into the same embedding space
3. Cosine similarity is computed
4. Top-K most similar items are returned

This makes search **fast and scalable**.

---

## Model Details

* **Text Encoder**: Transformer-based encoder
* **Vision Encoder**: Vision Transformer
* **Embedding Size**: 512 dimensions
* **Training Objective**: Symmetric contrastive loss (NT-Xent style)

---

## Example Use Cases

* Semantic product search
* Multimodal retrieval systems
* Recommendation engines
* Research on text-image alignment
* Prototyping multimodal ML systems

---

## Notes

* Images are cached locally after first download
* Missing images are handled gracefully
* The architecture is modular and easy to customize

---

## Extending the Project

It can easily extend this project by:

* Adding a vector database (FAISS, MongoDB, etc.)
* Exposing the model via an API
* Adding more modalities (e.g., metadata, audio)
* Switching encoders
* Adding evaluation metrics

---

## Summary

This repository demonstrates a **clean, modular, and production-ready multimodal ML pipeline**, covering:

> Data loading → Model definition → Training → Inference → Interactive demo

It is suitable for **learning, research, and real-world prototyping**.

