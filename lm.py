from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


class LM:
    """Small wrapper around AutoModelForCausalLM with hidden-state helpers."""

    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
        )
        self.model.eval()
        print(f"Loaded model {model_id} on device {next(self.model.parameters()).device}")

    def tokenize(self, text: str) -> Tensor:
        """Tokenize text into input_ids with shape (1, prompt_size)."""
        encoded = self.tokenizer(text, return_tensors="pt")
        return encoded["input_ids"]

    @torch.no_grad()
    def get_hidden_state(self, layer_index: int, prompt: Tensor) -> Tensor:
        """
        Return hidden state at `layer_index` for the last token in `prompt`.

        Args:
            layer_index: Index into `outputs.hidden_states`.
                Note: hidden_states[0] is the embedding output.
            prompt: Token ids, shape (1, prompt_size).

        Returns:
            Tensor of shape (d_model,) for the last token in the prompt.
        """
        if prompt.ndim != 2 or prompt.shape[0] != 1:
            raise ValueError("prompt must have shape (1, prompt_size)")

        device = next(self.model.parameters()).device
        prompt = prompt.to(device)

        outputs = self.model(prompt, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        if layer_index < 0 or layer_index >= len(hidden_states):
            raise IndexError(
                f"layer_index {layer_index} out of range (0..{len(hidden_states)-1})"
            )

        # hidden_states[layer_index]: (1, seq_len, d_model)
        last_token_state = hidden_states[layer_index][0, -1, :]
        return last_token_state

    @torch.no_grad()
    def get_hidden_state_all_prefixes(self, layer_index: int, prompt: Tensor) -> Tensor:
        """
        Return hidden states at `layer_index` for the last token of every prefix.

        Args:
            layer_index: Index into `outputs.hidden_states`.
                Note: hidden_states[0] is the embedding output.
            prompt: Token ids, shape (1, prompt_size).

        Returns:
            Tensor of shape (prompt_size, d_model).
            Row i corresponds to the last token of prefix ending at position i.
        """
        if prompt.ndim != 2 or prompt.shape[0] != 1:
            raise ValueError("prompt must have shape (1, prompt_size)")

        device = next(self.model.parameters()).device
        prompt = prompt.to(device)

        outputs = self.model(prompt, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        if layer_index < 0 or layer_index >= len(hidden_states):
            raise IndexError(
                f"layer_index {layer_index} out of range (0..{len(hidden_states)-1})"
            )

        # hidden_states[layer_index]: (1, seq_len, d_model)
        all_prefix_last_states = hidden_states[layer_index][0, :, :]
        return all_prefix_last_states

    @torch.no_grad()
    def get_hidden_state_all_layers(self, prompt: Tensor) -> Tensor:
        """
        Return hidden states for the last token across all layers.

        Args:
            prompt: Token ids, shape (1, prompt_size).

        Returns:
            Tensor of shape (num_layers, d_model).
            Note: includes the embedding output at index 0.
        """
        if prompt.ndim != 2 or prompt.shape[0] != 1:
            raise ValueError("prompt must have shape (1, prompt_size)")

        device = next(self.model.parameters()).device
        prompt = prompt.to(device)

        outputs = self.model(prompt, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")

        last_token_states = [h[0, -1, :] for h in hidden_states]
        return torch.stack(last_token_states, dim=0)

    def _middle_layer_index(self) -> int:
        """
        Return the hidden_states index for the middle transformer layer.

        Note: hidden_states[0] is the embedding output, so we offset by +1.
        """
        num_layers = int(self.model.config.num_hidden_layers)
        return (num_layers // 2) + 1

    @torch.no_grad()
    def embed_text(self, text: str) -> Tensor:
        """
        Embed text using the hidden state at the middle transformer layer.

        Returns:
            Tensor of shape (d_model,) for the last token in the prompt.
        """
        prompt = self.tokenize(text)
        layer_index = self._middle_layer_index()
        return self.get_hidden_state(layer_index, prompt)

    @torch.no_grad()
    def angular_distance(self, s: str, t: str) -> float:
        """
        Angular distance between embeddings of s and t.

        Returns:
            Normalized angular distance in [0, 1].
        """
        emb_s = self.embed_text(s)
        emb_t = self.embed_text(t)
        cos_sim = torch.nn.functional.cosine_similarity(
            emb_s.unsqueeze(0), emb_t.unsqueeze(0), dim=1
        )
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        return torch.acos(cos_sim).item() /  np.pi

    @torch.no_grad()
    def cosine_similarity(self, s: str, t: str) -> float:
        """
        Cosine similarity between embeddings of s and t.

        Returns:
            Cosine similarity in [-1, 1].
        """
        emb_s = self.embed_text(s)
        emb_t = self.embed_text(t)
        cos_sim = torch.nn.functional.cosine_similarity(
            emb_s.unsqueeze(0), emb_t.unsqueeze(0), dim=1
        )
        return cos_sim.item()

    @torch.no_grad()
    def _get_scores(self, p_tokens: Tensor, s_tokens: Tensor) -> Tensor:
        """
        Return the tensor of scores that the model + p_tokens gives to s_tokens.
        p_tokens and s_tokens are expected to be input_ids of shape (1, seq_len).

        The shape of this tensor should be (s_len, vocab_size).
        """
        ps_tokens = torch.cat([p_tokens, s_tokens], dim=1)

        device = next(self.model.parameters()).device
        ps_tokens = ps_tokens.to(device)

        outputs = self.model(ps_tokens, use_cache=False)
        # The scores for the i-th token of s_tokens are in the logits at
        # the (len(p_tokens) + i - 1)-th position.
        s_scores = outputs.logits[0, p_tokens.shape[1] - 1 : -1, :]
        return s_scores

    def _get_logprobs(self, scores: Tensor) -> Tensor:
        """
        Normalize scores to logprobs.

        Args:
            scores: Tensor of shape (s_len, vocab_size).

        Returns:
            Tensor of shape (s_len, vocab_size).
        """
        return torch.nn.functional.log_softmax(scores, dim=-1)

    @torch.no_grad()
    def _get_token_logprobs(self, p_tokens: Tensor, s_tokens: Tensor) -> list[float]:
        """
        Return per-token logprobs for s_tokens given p_tokens.

        Args:
            p_tokens: Tensor of shape (1, p_len).
            s_tokens: Tensor of shape (1, s_len).

        Returns:
            List of length s_len with logprob for each token in s_tokens.
        """
        scores = self._get_scores(p_tokens, s_tokens)
        logprobs = self._get_logprobs(scores)
        s_tokens_flat = s_tokens.flatten()
        return [logprobs[i, token_id].item() for i, token_id in enumerate(s_tokens_flat)]

    @torch.no_grad()
    def _get_total_logprob(self, p_tokens: Tensor, s_tokens: Tensor) -> float:
        """
        Compute the total logprob of the full string.
        """
        scores = self._get_scores(p_tokens, s_tokens)
        logprobs = self._get_logprobs(scores)
        
        # We need to get the log probability of each *actual* token in s_tokens.
        # The logprobs tensor has shape (s_len, vocab_size).
        # The s_tokens tensor has shape (1, s_len).
        # We want to select the logprob of the actual next token.
        s_tokens_flat = s_tokens.flatten()
        
        # The logprobs for the token at s_tokens[0, i] is at logprobs[i, s_tokens[0, i]]
        # But we need to select the logprob for the *next* token, so we use s_tokens[0, 1:]
        # and we take the logprobs from index 0 to -1
        
        # No, that's not right. The scores are already aligned with the s_tokens.
        # s_scores = outputs.logits[0, p_tokens.shape[1] - 1 : -1, :]
        # This means that s_scores[i] corresponds to the prediction for the token s_tokens[0, i].
        
        total_logprob = 0.0
        for i in range(s_tokens.shape[1]):
            token_id = s_tokens_flat[i]
            total_logprob += logprobs[i, token_id].item()
            
        return total_logprob


def _original_main() -> None:
    model_id = "Qwen/Qwen2.5-Coder-0.5B"
    # model_id = "Qwen/Qwen2.5-Coder-7B"
    lm = LM(model_id)
    print(f"Model: {model_id}")

    prompt_text = "def add(a, b):\n    return a + b\n"
    prompt = lm.tokenize(prompt_text)

    token_ids = prompt[0].tolist()
    tokens = lm.tokenizer.convert_ids_to_tokens(token_ids)
    tokens = [lm.tokenizer.convert_tokens_to_string([token]) for token in tokens]
    print(f"Prompt length: {len(token_ids)}")
    print(f"Token IDs: {token_ids}")
    print(f"Tokens: {tokens}")

    layer_index = 1
    last_state = lm.get_hidden_state(layer_index, prompt)
    all_prefix_states = lm.get_hidden_state_all_prefixes(layer_index, prompt)
    all_layer_states = lm.get_hidden_state_all_layers(prompt)

    print(f"Last state shape: {tuple(last_state.shape)}")
    print(f"All prefixes shape: {tuple(all_prefix_states.shape)}")
    print(f"All layers shape: {tuple(all_layer_states.shape)}")

    # s = "add two numbers"
    # t = "sum of two integers"
    # u = "sort a list in place"

    s, t, u = "hi", "hello", "goodbye"
    print(f"Angular distance(s, t): {lm.angular_distance(s, t):.4f}")
    print(f"Angular distance(s, u): {lm.angular_distance(s, u):.4f}")
    print(f"Cosine similarity(s, t): {lm.cosine_similarity(s, t):.4f}")
    print(f"Cosine similarity(s, u): {lm.cosine_similarity(s, u):.4f}")


def _main() -> None:
    model_id = "Qwen/Qwen2.5-Coder-0.5B"
    # model_id = "Qwen/Qwen2.5-Coder-7B"
    lm = LM(model_id)
    print(f"\n--- Testing Logprob Functions ---")

    prompt_add = "def add(a, b):\n    "
    # prompt_sub = "def subtract(a, b):\n    "
    prompt_sub = "def subtract(a, b):\n    "

    query_add = "return a + b\n"
    query_sub = "return a - b\n"

    prompts = [prompt_add, prompt_sub]
    queries = [query_add, query_sub]

    for q in queries:
        print(f"\nQuery:\n{q.strip()}")

        # Shared tokenization for display
        q_tokens = lm.tokenize(q)
        q_token_ids = q_tokens[0].tolist()
        q_tokens_str = lm.tokenizer.convert_ids_to_tokens(q_token_ids)
        q_tokens_str = [
            lm.tokenizer.convert_tokens_to_string([token]) for token in q_tokens_str
        ]

        per_prompt_logprobs: list[list[float]] = []
        per_prompt_totals: list[float] = []

        for p in prompts:
            p_tokens = lm.tokenize(p)
            q_token_logprobs = lm._get_token_logprobs(p_tokens, q_tokens)
            per_prompt_logprobs.append(q_token_logprobs)

            total_logprob = lm._get_total_logprob(p_tokens, q_tokens)
            per_prompt_totals.append(total_logprob)

        if len(prompts) >= 2:
            print(f"\nPrompt 1:\n{prompts[0].strip()}")
            print(f"Prompt 2:\n{prompts[1].strip()}")
            print("\nStep | Token | Token ID | P1 Logprob | P2 Logprob | Δ Logprob")
            print("-----|-------|----------|------------|------------|----------")
            for i, (token_str, token_id, lp1, lp2) in enumerate(
                zip(
                    q_tokens_str,
                    q_token_ids,
                    per_prompt_logprobs[0],
                    per_prompt_logprobs[1],
                ),
                start=1,
            ):
                print(
                    f"{i:>4} | {token_str!r} | {token_id:>8} | "
                    f"{lp1:>10.4f} | {lp2:>10.4f} | {(lp2 - lp1):>8.4f}"
                )

            print("-----|-------|----------|------------|------------|----------")
            print(
                f"{'TOTAL':>4} | {'':<5} | {'':<8} | {per_prompt_totals[0]:>10.4f} | "
                f"{per_prompt_totals[1]:>10.4f} | "
                f"{(per_prompt_totals[1] - per_prompt_totals[0]):>8.4f}"
            )

if __name__ == "__main__":
    _main()
