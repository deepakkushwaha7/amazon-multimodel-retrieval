import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import clip


# =========================
# INTERNAL MODEL CONFIG
# =========================
TEXT_BACKBONE = "bert-base-uncased"
VISION_BACKBONE = "ViT-B/32"


class TextEncoder(nn.Module):
    """
    Transformer-based text encoder used for multiple text inputs.
    """
    def __init__(self, backbone=TEXT_BACKBONE, output_dim=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.output_dim = output_dim

    def forward(self, texts, device):
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        outputs = self.encoder(**tokens)
        return outputs.last_hidden_state[:, 0, :]  # CLS embedding


class VisionEncoder(nn.Module):
    """
    Vision encoder using a pretrained transformer-based image backbone.
    """
    def __init__(self, device, output_dim=512):
        super().__init__()
        self.device = device
        model, preprocess = clip.load(VISION_BACKBONE, device=device, jit=False)
        self.visual_encoder = model.visual
        self.preprocess = preprocess
        self.output_dim = output_dim

    def forward(self, images, device):
        processed = []
        for img in images:
            if img is not None:
                processed.append(self.preprocess(img))
            else:
                processed.append(torch.zeros(3, 224, 224))

        batch = torch.stack(processed).to(device)

        with torch.no_grad():
            features = self.visual_encoder(batch)

        return features


class MultiTowerEncoder(nn.Module):
    """
    Multi-tower encoder architecture:

    - Query text encoder
    - Title text encoder
    - Description text encoder
    - Image encoder

    Outputs embeddings in a shared 512-dim space.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.text_dim = 768
        self.image_dim = 512
        self.embed_dim = 512

        # Text towers
        self.query_encoder = TextEncoder()
        self.title_encoder = TextEncoder()
        self.desc_encoder = TextEncoder()

        # Vision tower
        self.image_encoder = VisionEncoder(device=device)

        # Projection layers
        self.product_proj = nn.Linear(
            2 * self.text_dim + self.image_dim,
            self.embed_dim
        )

        self.query_proj = nn.Linear(
            3 * self.text_dim + self.image_dim,
            self.embed_dim
        )

        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, batch):
        q_vec = self.query_encoder(batch["query_texts"], self.device)
        t_vec = self.title_encoder(batch["titles"], self.device)
        d_vec = self.desc_encoder(batch["descriptions"], self.device)
        i_vec = self.image_encoder(batch["images"], self.device)

        product_concat = torch.cat([t_vec, d_vec, i_vec], dim=1)
        product_emb = self.norm(self.product_proj(product_concat))

        query_concat = torch.cat([q_vec, t_vec, d_vec, i_vec], dim=1)
        query_emb = self.norm(self.query_proj(query_concat))

        return query_emb, product_emb

    def encode_queries(self, texts):
        return self.query_encoder(texts, self.device)

    def encode_products(self, titles, descriptions, images):
        t_vec = self.title_encoder(titles, self.device)
        d_vec = self.desc_encoder(descriptions, self.device)
        i_vec = self.image_encoder(images, self.device)

        concat = torch.cat([t_vec, d_vec, i_vec], dim=1)
        return self.norm(self.product_proj(concat))


# =========================
# CONTRASTIVE OBJECTIVE
# =========================
def contrastive_loss(a, b, temperature=0.07):
    """
    Symmetric contrastive loss between two embedding sets.

    a: (N, D)
    b: (N, D)
    """
    batch_size = a.size(0)
    device = a.device

    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)

    logits = torch.matmul(a, b.T) / temperature
    targets = torch.arange(batch_size, device=device)

    loss_ab = F.cross_entropy(logits, targets)
    loss_ba = F.cross_entropy(logits.T, targets)

    return 0.5 * (loss_ab + loss_ba)
