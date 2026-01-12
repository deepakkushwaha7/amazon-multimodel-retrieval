import torch
import torch.nn.functional as F
import gradio as gr

# Internal modules
from dataset import ProductDataset
from model import MultiTowerEncoder


# =========================
# LOAD DATA
# =========================
DATA_SOURCE = "data/products.csv"  # local path or internal URI

dataset = ProductDataset(
    data_source=DATA_SOURCE,
    max_samples=300
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
model = MultiTowerEncoder(device=device).to(device)
model.load_state_dict(
    torch.load("model_weights.pt", map_location=device)
)
model.eval()

# =========================
# PRECOMPUTE PRODUCT EMBEDDINGS
# =========================
product_vectors = []
product_metadata = []

with torch.no_grad():
    batch_size = 32

    for start in range(0, len(dataset), batch_size):
        batch_items = dataset[start : start + batch_size]

        titles = [item["title"] for item in batch_items]
        descriptions = [item["description"] for item in batch_items]
        images = [item["image"] for item in batch_items]

        embeddings = model.encode_products(
            titles,
            descriptions,
            images
        )

        product_vectors.append(embeddings)
        product_metadata.extend(batch_items)

    product_vectors = torch.cat(product_vectors, dim=0)


# =========================
# SEARCH FUNCTION
# =========================
def search_items(query: str, top_k: int = 5):
    """
    Encode a query and retrieve the top-k most similar items.
    """
    with torch.no_grad():
        # Use a dummy product to enable full query fusion
        reference = dataset[0]

        batch = {
            "query_texts": [query],
            "titles": [reference["title"]],
            "descriptions": [reference["description"]],
            "images": [reference["image"]],
        }

        query_vector, _ = model(batch)

        query_norm = F.normalize(query_vector, dim=1)
        product_norm = F.normalize(product_vectors, dim=1)

        scores = torch.matmul(query_norm, product_norm.T).squeeze(0)
        values, indices = torch.topk(scores, top_k)

        results = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            item = product_metadata[idx]
            results.append({
                "title": item["title"],
                "id": item.get("id", ""),
                "score": f"{score:.3f}",
            })

        return results


# =========================
# GRADIO INTERFACE
# =========================
interface = gr.Interface(
    fn=search_items,
    inputs=[
        gr.Textbox(label="Search Query"),
        gr.Slider(
            minimum=1,
            maximum=10,
            step=1,
            value=5,
            label="Number of Results"
        ),
    ],
    outputs=gr.JSON(label="Results"),
    title="Multimodal Semantic Search Demo",
    description=(
        "Enter a text query to retrieve semantically similar items "
        "using a multimodal embedding model."
    ),
)

# =========================
# LAUNCH
# =========================
if __name__ == "__main__":
    interface.launch()
