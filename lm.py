from __future__ import annotations

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from find_tokens import _display_token


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


def _main() -> None:
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


if __name__ == "__main__":
    _main()
