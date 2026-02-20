from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from lm import LM  # noqa: E402


def load_test_split(path: Path) -> Tuple[List[str], List[str]]:
    indices: List[str] = []
    codes: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            indices.append(record["index"])
            codes.append(record["code"])
    return indices, codes


@torch.no_grad()
def compute_embeddings(lm: LM, codes: List[str]) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    for i, code in enumerate(codes, start=1):
        emb = lm.embed_text(code).detach().float().cpu()
        embeddings.append(emb)
        if i % 100 == 0:
            print(f"Embedded {i}/{len(codes)}")
    return torch.stack(embeddings, dim=0)


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


@torch.no_grad()
def compute_embeddings_qwen3(
    model_id: str,
    codes: List[str],
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModel.from_pretrained(model_id, device_map="auto")
    model.eval()
    print(f"Loaded embedding model {model_id} on device {next(model.parameters()).device}")

    embeddings: List[torch.Tensor] = []
    for start in range(0, len(codes), batch_size):
        end = min(start + batch_size, len(codes))
        batch = tokenizer(
            codes[start:end],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        pooled = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.detach().float().cpu())
        print(f"Embedded {end}/{len(codes)}")

    return torch.cat(embeddings, dim=0)


@torch.no_grad()
def knn_by_angular_distance(
    embeddings: torch.Tensor,
    k: int,
    batch_size: int,
) -> torch.Tensor:
    """
    Compute KNN indices for each row in embeddings.

    Note: Angular distance is monotonically decreasing in cosine similarity,
    so nearest neighbors by angular distance are the same as by cosine similarity.
    """
    n, _ = embeddings.shape
    emb = torch.nn.functional.normalize(embeddings, dim=1)
    all_knn = torch.empty((n, k), dtype=torch.long)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        q = emb[start:end]
        sims = q @ emb.T  # (B, N)
        # Exclude self
        for i in range(start, end):
            sims[i - start, i] = float("-inf")
        topk = torch.topk(sims, k=k, dim=1, largest=True)
        all_knn[start:end] = topk.indices.cpu()
        print(f"KNN {end}/{n}")

    return all_knn


def write_predictions(
    out_path: Path,
    indices: List[str],
    knn_indices: torch.Tensor,
    id_map: List[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, neighbors in zip(indices, knn_indices.tolist()):
            answers = [id_map[n] for n in neighbors]
            f.write(json.dumps({"index": idx, "answers": answers}) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate KNN predictions for POJ-104 using middle-layer embeddings.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-Coder-0.5B",
        help="HuggingFace model id.",
    )
    parser.add_argument(
        "--use-qwen-embedding",
        action="store_true",
        help="Use Qwen/Qwen3-Embedding-0.6B embeddings instead of hidden-state embeddings.",
    )
    parser.add_argument(
        "--embedding-model-id",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model id (used with --use-qwen-embedding).",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help="Batch size for embedding model inference.",
    )
    parser.add_argument(
        "--embedding-max-length",
        type=int,
        default=8192,
        help="Max token length for embedding model inputs.",
    )
    parser.add_argument(
        "--test-path",
        default=str(REPO_ROOT / "Clone-detection-POJ-104/dataset/test.jsonl"),
        help="Path to test.jsonl.",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="Output predictions.jsonl path.",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors.")
    parser.add_argument(
        "--knn-batch-size",
        type=int,
        default=64,
        help="Batch size for KNN similarity computation.",
    )
    args = parser.parse_args()

    test_path = Path(args.test_path)
    indices, codes = load_test_split(test_path)
    if len(indices) != 12000:
        print(f"Warning: expected 12000 lines, got {len(indices)}")

    if args.out_path is None:
        default_dir = "qwen3_embedding" if args.use_qwen_embedding else "hs_mid"
        args.out_path = str(
            REPO_ROOT
            / f"Clone-detection-POJ-104/dataset/preds/{default_dir}/predictions.jsonl"
        )

    if args.use_qwen_embedding:
        embeddings = compute_embeddings_qwen3(
            args.embedding_model_id,
            codes,
            args.embedding_batch_size,
            args.embedding_max_length,
        )
    else:
        lm = LM(args.model_id)
        embeddings = compute_embeddings(lm, codes)
    knn_indices = knn_by_angular_distance(embeddings, args.k, args.knn_batch_size)
    write_predictions(Path(args.out_path), indices, knn_indices, indices)

    print(f"Wrote predictions to {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
