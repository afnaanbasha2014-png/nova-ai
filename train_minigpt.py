"""Train a tiny GPT-style character model from random weights.

Designed as a beginner learning exercise and for Google Colab.
Dataset: Tiny Shakespeare (public-domain Shakespeare text).
This is a text-continuation demo, not an instruction-following assistant.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, sequence_length, _ = x.shape
        # True values mask future positions, so tokens cannot look ahead.
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + self.dropout(attended)
        x = x + self.feed_forward(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        width: int = 128,
        heads: int = 4,
        layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(block_size, width)
        self.blocks = nn.Sequential(
            *[TransformerBlock(width, heads, dropout) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(width)
        self.language_head = nn.Linear(width, vocab_size)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length = token_ids.shape
        positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.blocks(x)
        logits = self.language_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, new_tokens: int, temperature: float = 0.8):
        self.eval()
        for _ in range(new_tokens):
            context = token_ids[:, -self.block_size :]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :] / max(temperature, 0.05)
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids


def download_dataset(path: Path) -> str:
    if not path.exists():
        print("Downloading the Tiny Shakespeare training text…")
        try:
            urllib.request.urlretrieve(DATA_URL, path)
        except Exception as exc:
            raise RuntimeError(
                "Could not download the training text. Check your internet connection "
                "and run the script again."
            ) from exc
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny GPT from random weights.")
    parser.add_argument("--steps", type=int, default=2500, help="Training steps (default: 2500).")
    parser.add_argument("--output", type=str, default="mini_gpt.pt", help="Checkpoint filename.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")
    if device == "cpu":
        print("Tip: in Google Colab, choose Runtime → Change runtime type → GPU for faster training.")

    text = download_dataset(Path("tinyshakespeare.txt"))
    characters = sorted(set(text))
    vocab_size = len(characters)
    char_to_id = {character: index for index, character in enumerate(characters)}
    id_to_char = {index: character for character, index in char_to_id.items()}

    all_ids = torch.tensor([char_to_id[character] for character in text], dtype=torch.long)
    split_at = int(0.9 * len(all_ids))
    train_data = all_ids[:split_at]
    validation_data = all_ids[split_at:]

    block_size = 128
    batch_size = args.batch_size
    width = 128
    heads = 4
    layers = 4
    dropout = 0.1

    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        source = train_data if split == "train" else validation_data
        starts = torch.randint(len(source) - block_size - 1, (batch_size,))
        x = torch.stack([source[start : start + block_size] for start in starts])
        y = torch.stack([source[start + 1 : start + block_size + 1] for start in starts])
        return x.to(device), y.to(device)

    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        width=width,
        heads=heads,
        layers=layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    @torch.no_grad()
    def estimate_loss() -> tuple[float, float]:
        model.eval()
        results: list[float] = []
        for split in ("train", "validation"):
            losses = []
            for _ in range(20):
                x_batch, y_batch = get_batch(split)
                _, loss = model(x_batch, y_batch)
                assert loss is not None
                losses.append(float(loss.item()))
            results.append(sum(losses) / len(losses))
        model.train()
        return results[0], results[1]

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Vocabulary: {vocab_size} characters")
    print(f"Model parameters: {total_parameters:,}")
    print(f"Training steps: {args.steps}")

    model.train()
    for step in range(args.steps + 1):
        if step % 250 == 0 or step == args.steps:
            train_loss, validation_loss = estimate_loss()
            print(
                f"step {step:5d} | train loss {train_loss:.3f} "
                f"| validation loss {validation_loss:.3f}"
            )

        x_batch, y_batch = get_batch("train")
        _, loss = model(x_batch, y_batch)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "characters": characters,
        "config": {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "width": width,
            "heads": heads,
            "layers": layers,
            "dropout": dropout,
        },
    }
    output_path = Path(args.output)
    torch.save(checkpoint, output_path)
    print(f"\nSaved your trained model to: {output_path.resolve()}")

    # Show a short sample generated by the newly trained model.
    model = model.to(device)
    prompt = "ROMEO:"
    prompt_ids = [char_to_id[ch] for ch in prompt if ch in char_to_id]
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(context, new_tokens=400, temperature=0.8)[0].tolist()
    print("\n--- SAMPLE TEXT FROM YOUR MODEL ---\n")
    print("".join(id_to_char[token] for token in generated))


if __name__ == "__main__":
    main()
