from __future__ import annotations

import torch
from torch import Tensor

from lm import LM


def logprob_analysis(lm: LM, prompts: list[Tensor], queries: list[Tensor]) -> None:
    assert len(prompts) == 2, "logprob_analysis expects exactly 2 prompts for now"

    for q_tokens in queries:
        q_token_ids = q_tokens[0].tolist()
        q_tokens_str = lm.tokenizer.convert_ids_to_tokens(q_token_ids)
        q_tokens_str = [
            lm.tokenizer.convert_tokens_to_string([token]) for token in q_tokens_str
        ]
        q_text = lm.tokenizer.decode(q_token_ids)
        print(f"\nQuery:\n{q_text.strip()}")

        per_prompt_logprobs: list[list[float]] = []
        per_prompt_totals: list[float] = []

        for p_tokens in prompts:
            q_token_logprobs = lm._get_token_logprobs(p_tokens, q_tokens)
            per_prompt_logprobs.append(q_token_logprobs)

            total_logprob = lm._get_total_logprob(p_tokens, q_tokens)
            per_prompt_totals.append(total_logprob)

        prompt_texts = [lm.tokenizer.decode(p[0].tolist()) for p in prompts]
        print(f"\nPrompt 1:\n{prompt_texts[0].strip()}")
        print(f"Prompt 2:\n{prompt_texts[1].strip()}")
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


@torch.no_grad()
def _generate_greedy_completion(
    lm: LM, prompt_tokens: Tensor, max_new_tokens: int
) -> Tensor:
    device = next(lm.model.parameters()).device
    prompt_tokens = prompt_tokens.to(device)
    outputs = lm.model.generate(
        input_ids=prompt_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=lm.tokenizer.eos_token_id,
    )
    completion_tokens = outputs[:, prompt_tokens.shape[1] :]
    return completion_tokens.cpu()


def compare_prompts(
    lm: LM, prompt_a: str, prompt_b: str, max_new_tokens: int = 256
) -> None:
    prompt_tokens = [lm.tokenize(prompt_a), lm.tokenize(prompt_b)]
    completion_tokens = [
        _generate_greedy_completion(lm, prompt_tokens[0], max_new_tokens),
        _generate_greedy_completion(lm, prompt_tokens[1], max_new_tokens),
    ]

    completion_texts = [
        lm.tokenizer.decode(tokens[0].tolist(), skip_special_tokens=False)
        for tokens in completion_tokens
    ]
    print("\n--- Greedy Completion 1 ---")
    print(completion_texts[0].rstrip())
    print("\n--- Greedy Completion 2 ---")
    print(completion_texts[1].rstrip())

    logprob_analysis(lm, prompt_tokens, completion_tokens)


def _main() -> None:
    # model_id = "Qwen/Qwen2.5-Coder-0.5B"
    model_id = "Qwen/Qwen2.5-Coder-7B"
    lm = LM(model_id)
    print(f"\n--- Comparing Prompts (Greedy Completion + Logprob Analysis) ---")

    with open("prompts/scc.txt", "r", encoding="utf-8") as f:
        prompt_a = f.read()
    with open("prompts/scc_types.txt", "r", encoding="utf-8") as f:
        prompt_b = f.read()

    compare_prompts(lm, prompt_a, prompt_b)


if __name__ == "__main__":
    _main()
