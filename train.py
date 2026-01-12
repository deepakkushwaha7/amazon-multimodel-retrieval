import torch
from torch.utils.data import DataLoader

# Internal modules (generic names)
from dataset import ProductDataset
from model import MultiTowerModel, contrastive_loss


def run_training(
    data_source,
    device,
    epochs=2,
    batch_size=8,
    max_samples=2000,
):
    """
    Train a multi-tower representation model using contrastive learning.

    Args:
        data_source: Path or URI to input data
        device: torch.device("cuda") or torch.device("cpu")
        epochs: number of training epochs
        batch_size: batch size
        max_samples: cap on number of samples used
    """

    dataset = ProductDataset(
        data_source=data_source,
        max_samples=max_samples
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda x: x
    )

    model = MultiTowerModel(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for items in dataloader:
            batch = {
                "query_texts": [it["title"] for it in items],
                "titles": [it["title"] for it in items],
                "descriptions": [it["description"] for it in items],
                "images": [it["image"] for it in items],
            }

            optimizer.zero_grad()

            query_vecs, item_vecs = model(batch)
            loss = contrastive_loss(
                query_vecs,
                item_vecs,
                temperature=0.07
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "model_weights.pt")
    print("Training finished. Model saved to model_weights.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-modal representation training")
    parser.add_argument("--data", type=str, required=True, help="Path or URI to training data")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=2000)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_training(
        data_source=args.data,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
