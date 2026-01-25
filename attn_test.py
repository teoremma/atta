import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

import datetime
import os

LOGIT_BIAS_VALUE = -1e6
LOGIT_BIAS_TOKEN_IDS = (
    2,
    442,
    671,
    3190,
    4210,
    7129,
    7704,
    11166,
    11456,
    12599,
)
LOGIT_BIAS = {token_id: LOGIT_BIAS_VALUE for token_id in LOGIT_BIAS_TOKEN_IDS}
ARROW_MUTATION_SCALE = 14

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class AttentionTest:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
        )
        self.model.eval()
        self.model.set_attn_implementation("eager")
        self.device = next(self.model.parameters()).device

    def get_last_n_tokens_str(self, prefix: tuple, n: int) -> str:
        last_n_tokens = prefix[-n:]
        return "".join(self.tokenizer.batch_decode(last_n_tokens)).replace("\n", "\\n")

    class TokenBiasLogitsProcessor(LogitsProcessor):
        def __init__(self, token_bias: dict[int, float]):
            self.token_bias = token_bias

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            for token_id, bias in self.token_bias.items():
                scores[:, token_id] += bias
            return scores

    def format_prefix_with_marker(self, prefix_ids: tuple[int, ...]) -> str:
        if not prefix_ids:
            return ""
        start_idx = 0
        for idx in range(len(prefix_ids) - 2, -1, -1):
            token_text = self.tokenizer.decode([prefix_ids[idx]], skip_special_tokens=False)
            if "\n" in token_text:
                start_idx = idx + 1
                break
        prev_ids = list(prefix_ids[start_idx:-1])
        prev_text = self.tokenizer.decode(prev_ids, skip_special_tokens=False) if prev_ids else ""
        last_token_text = self.tokenizer.decode([prefix_ids[-1]], skip_special_tokens=False)
        def escape_text(text: str) -> str:
            return text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
        line_prefix = prev_text.split("\n")[-1] if prev_text else ""
        return escape_text(line_prefix) + "|>" + escape_text(last_token_text) + "<|"

    def format_prefix_line_and_token(self, prefix_ids: tuple[int, ...]) -> tuple[str, str]:
        if not prefix_ids:
            return "", ""
        start_idx = 0
        for idx in range(len(prefix_ids) - 2, -1, -1):
            token_text = self.tokenizer.decode([prefix_ids[idx]], skip_special_tokens=False)
            if "\n" in token_text:
                start_idx = idx + 1
                break
        prev_ids = list(prefix_ids[start_idx:-1])
        prev_text = self.tokenizer.decode(prev_ids, skip_special_tokens=False) if prev_ids else ""
        last_token_text = self.tokenizer.decode([prefix_ids[-1]], skip_special_tokens=False)
        if last_token_text == "":
            last_token_text = self.tokenizer.convert_ids_to_tokens([prefix_ids[-1]])[0]

        def escape_text(text: str) -> str:
            return text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")

        line_prefix = prev_text.split("\n")[-1] if prev_text else ""
        return escape_text(line_prefix), escape_text(last_token_text)

    def text_width_points(self, text: str, fontsize: int, fontfamily: str) -> float:
        if not text:
            return 0.0
        try:
            from matplotlib.font_manager import FontProperties
            from matplotlib.textpath import TextPath
            font_prop = FontProperties(family=fontfamily, size=fontsize)
            text_path = TextPath((0, 0), text, prop=font_prop)
            return text_path.get_extents().width
        except Exception:
            return len(text) * fontsize * 0.6

    def monospace_width_points(self, text: str, fontsize: int) -> float:
        if not text:
            return 0.0
        return len(text) * fontsize * 0.6

    def get_hidden_states(self, prompt: str, n_completions: int, max_new_tokens: int = 32) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]

        logits_processor = LogitsProcessorList([
            self.TokenBiasLogitsProcessor(LOGIT_BIAS),
        ])

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            num_return_sequences=n_completions,
            # stop_strings=["\n"],
            stop_strings=["```"],
            return_dict_in_generate=True,
            output_hidden_states=True,
            # temperature=1.5,
        )

        outputs = self.model.generate(
            **inputs,
            generation_config=generation_config,
            tokenizer=self.tokenizer,
            logits_processor=logits_processor,
        )


        sequences = outputs.sequences
        # sequences has shape (n_completions, sequence_length)
        assert len(sequences.shape) == 2
        assert sequences.shape[0] == n_completions
        # sequences = sequences[:, len(inputs.input_ids[0]):]  # Remove the prompt tokens
        sequences = sequences[:, prompt_length:]  # Remove the prompt tokens
        assert sequences.shape[1] <= max_new_tokens
        max_generated_length = sequences.shape[1]

        hidden_states = outputs.hidden_states
        num_layers = self.model.config.num_hidden_layers
        hidden_size = self.model.config.hidden_size

        assert len(hidden_states) == max_generated_length
        hidden_states_list = []
        for step_idx, hidden_states_step in enumerate(hidden_states):
            assert len(hidden_states_step) == num_layers + 1 
            hidden_states_step_list = []
            for layer_idx, hidden_states_layer in enumerate(hidden_states_step):
                # print(f"Layer {layer_idx}")
                assert len(hidden_states_layer.shape) == 3
                assert hidden_states_layer.shape[0] == n_completions
                if step_idx == 0:
                    assert hidden_states_layer.shape[1] == prompt_length
                    # we want to keep only the hidden states corresponding to the gen token
                    hidden_states_layer = hidden_states_layer[:, prompt_length-1:, :]
                # else:
                #     assert layer_hidden_state.shape[1] == 1
                assert hidden_states_layer.shape[1] == 1
                assert hidden_states_layer.shape[2] == hidden_size
                hidden_states_step_list.append(hidden_states_layer)
            hidden_states_step_tensor = torch.cat(hidden_states_step_list, dim=1)
            assert hidden_states_step_tensor.shape == (n_completions, num_layers + 1, hidden_size)
            hidden_states_list.append(hidden_states_step_tensor)
        hidden_states_tensor = torch.stack(hidden_states_list, dim=0)
        assert hidden_states_tensor.shape == (max_generated_length, n_completions, num_layers + 1, hidden_size)
        # reorder to (n_completions, max_generated_length, num_layers + 1, hidden_size)
        hidden_states_tensor = hidden_states_tensor.permute(1, 0, 2, 3)
        assert hidden_states_tensor.shape == (n_completions, max_generated_length, num_layers + 1, hidden_size)

        print("stop")

        # return hidden_states_tensor
        return {
            "prompt_ids": inputs.input_ids,
            "completions_ids": sequences,
            "hidden_states": hidden_states_tensor,
        }

    def plot_embeddings(self, embeddings: dict, idx: int, plot_dir: str):
        # embeddings is a dictionary mapping from prefix (tuple of token ids) to hidden state vector (numpy array)
        # make sure the plot directory exists
        os.makedirs(f"{plot_dir}/tsne", exist_ok=True)

        hidden_state_vectors = np.array(list(embeddings.values()))
        prefixes = list(embeddings.keys())
        # we need to check if there are enough samples for the perplexity
        perplexity = min(10, len(prefixes) - 1)
        tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
        hidden_state_vectors_2d = tsne.fit_transform(hidden_state_vectors)
        plt.figure(figsize=(10, 10))
        for i, prefix in enumerate(prefixes):
            plt.scatter(hidden_state_vectors_2d[i, 0], hidden_state_vectors_2d[i, 1])
            # only show the last 5 tokens
            label = self.get_last_n_tokens_str(prefix, 5)
            plt.annotate(label, (hidden_state_vectors_2d[i, 0], hidden_state_vectors_2d[i, 1]))
        plt.title("t-SNE of Hidden State Vectors")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        # make sure the plot directory exists
        os.makedirs(f"{plot_dir}/tsne", exist_ok=True)
        plt.savefig(f"{plot_dir}/tsne/hidden_states_layer_{idx}.png", dpi=200)
        plt.close()
        
    def plot_sequence_tsne(
        self,
        hidden_states_tensor: torch.Tensor,
        completion_ids: torch.Tensor,
        idx: int,
        plot_dir: str,
        only_newline_tokens: bool = False,
    ):
        os.makedirs(f"{plot_dir}/tsne_sequences", exist_ok=True)
        n_completions, max_generated_length, _, hidden_size = hidden_states_tensor.shape
        eos_token_id = self.tokenizer.eos_token_id

        points = []
        completion_steps = []
        completion_token_steps = []
        completion_edge_flags = []
        for completion_idx in range(n_completions):
            step_indices = []
            token_steps = []
            edge_flags = []
            for step_idx in range(max_generated_length):
                token_id = completion_ids[completion_idx, step_idx].item()
                if eos_token_id is not None and token_id == eos_token_id:
                    break
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                is_edge = "\n" in token_text
                if only_newline_tokens:
                    if not is_edge:
                        continue
                vector = hidden_states_tensor[completion_idx, step_idx, idx, :].cpu().numpy()
                step_indices.append(len(points))
                token_steps.append(step_idx)
                edge_flags.append(is_edge)
                points.append(vector)
            completion_steps.append(step_indices)
            completion_token_steps.append(token_steps)
            completion_edge_flags.append(edge_flags)

        if not points:
            return

        points_array = np.array(points)
        perplexity = min(10, len(points_array) - 1)
        tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
        points_2d = tsne.fit_transform(points_array)

        plt.figure(figsize=(12, 10))
        cmap = plt.get_cmap("turbo")
        color_vals = np.linspace(0, 1, max(n_completions, 2))
        colors = [cmap(v) for v in color_vals]
        for completion_idx, step_indices in enumerate(completion_steps):
            if not step_indices:
                continue
            color = colors[completion_idx % len(colors)]
            coords = points_2d[step_indices]
            edge_flags = completion_edge_flags[completion_idx]
            edge_plot_indices = [
                i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge
            ]
            if len(coords):
                edge_plot_indices.extend([0, len(coords) - 1])
            edge_plot_indices = sorted(set(edge_plot_indices))
            edge_coords = coords[edge_plot_indices] if edge_plot_indices else np.array([])
            non_edge_coords = coords[
                [i for i in range(len(coords)) if i not in edge_plot_indices]
            ]
            if len(non_edge_coords):
                plt.scatter(
                    non_edge_coords[:, 0],
                    non_edge_coords[:, 1],
                    s=12,
                    facecolors="none",
                    edgecolors=color,
                    label=f"c{completion_idx}",
                )
            if len(edge_coords):
                plt.scatter(
                    edge_coords[:, 0],
                    edge_coords[:, 1],
                    s=28,
                    color=color,
                    label=None,
                )
            for i in range(1, len(coords)):
                start = coords[i - 1]
                end = coords[i]
                plt.annotate(
                    "",
                    xy=(end[0], end[1]),
                    xytext=(start[0], start[1]),
                    arrowprops={
                        "arrowstyle": "->",
                        "mutation_scale": ARROW_MUTATION_SCALE,
                        "linewidth": 0.8,
                        "color": color,
                        "linestyle": (0, (1, 3)),
                    },
                )
            plt.annotate(
                str(completion_idx),
                xy=(coords[0, 0], coords[0, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
            )
            plt.annotate(
                str(completion_idx),
                xy=(coords[-1, 0], coords[-1, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "black", "linewidth": 0.6},
            )
            edge_plot_indices = [
                i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge
            ]
            if len(edge_plot_indices) > 1:
                edge_coords = coords[edge_plot_indices]
                for i in range(len(edge_coords) - 1):
                    start = edge_coords[i]
                    end = edge_coords[i + 1]
                    plt.annotate(
                        "",
                        xy=(end[0], end[1]),
                        xytext=(start[0], start[1]),
                        arrowprops={
                            "arrowstyle": "->",
                            "mutation_scale": ARROW_MUTATION_SCALE,
                            "linewidth": 1.6,
                            "color": color,
                            "linestyle": "-",
                        },
                    )
            edge_only = [i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge]
            if edge_only and len(coords) > 1:
                first_edge = edge_only[0]
                last_edge = edge_only[-1]
                if first_edge != 0:
                    plt.annotate(
                        "",
                        xy=(coords[first_edge][0], coords[first_edge][1]),
                        xytext=(coords[0][0], coords[0][1]),
                        arrowprops={
                            "arrowstyle": "->",
                            "mutation_scale": ARROW_MUTATION_SCALE,
                            "linewidth": 1.6,
                            "color": color,
                            "linestyle": "-",
                        },
                    )
                if last_edge != len(coords) - 1:
                    plt.annotate(
                        "",
                        xy=(coords[len(coords) - 1][0], coords[len(coords) - 1][1]),
                        xytext=(coords[last_edge][0], coords[last_edge][1]),
                        arrowprops={
                            "arrowstyle": "->",
                            "mutation_scale": ARROW_MUTATION_SCALE,
                            "linewidth": 1.6,
                            "color": color,
                            "linestyle": "-",
                        },
                    )

        plt.title("t-SNE of Hidden State Vectors by Completion")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(title="Completion", ncol=2, fontsize=7, title_fontsize=8, loc="best")
        x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
        y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05
        xlim = (x_min - x_pad, x_max + x_pad)
        ylim = (y_min - y_pad, y_max + y_pad)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/tsne_sequences/hidden_states_tsne_layer_{idx}.png", dpi=200)
        plt.close()

        # Angle-colored arrows plot
        plt.figure(figsize=(12, 10))
        arrow_cmap = plt.get_cmap("hsv")
        for completion_idx, step_indices in enumerate(completion_steps):
            if not step_indices:
                continue
            color = colors[completion_idx % len(colors)]
            coords = points_2d[step_indices]
            edge_flags = completion_edge_flags[completion_idx]
            edge_plot_indices = [
                i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge
            ]
            if len(coords):
                edge_plot_indices.extend([0, len(coords) - 1])
            edge_plot_indices = sorted(set(edge_plot_indices))
            edge_coords = coords[edge_plot_indices] if edge_plot_indices else np.array([])
            non_edge_coords = coords[
                [i for i in range(len(coords)) if i not in edge_plot_indices]
            ]
            if len(non_edge_coords):
                plt.scatter(
                    non_edge_coords[:, 0],
                    non_edge_coords[:, 1],
                    s=12,
                    facecolors="none",
                    edgecolors=color,
                    label=f"c{completion_idx}",
                )
            if len(edge_coords):
                plt.scatter(
                    edge_coords[:, 0],
                    edge_coords[:, 1],
                    s=28,
                    color=color,
                    label=None,
                )
            for i in range(1, len(coords)):
                start = coords[i - 1]
                end = coords[i]
                angle = np.arctan2(end[1] - start[1], end[0] - start[0])
                hue = (angle + np.pi) / (2 * np.pi)
                arrow_color = arrow_cmap(hue)
                plt.annotate(
                    "",
                    xy=(end[0], end[1]),
                    xytext=(start[0], start[1]),
                    arrowprops={
                        "arrowstyle": "->",
                        "mutation_scale": ARROW_MUTATION_SCALE,
                        "linewidth": 0.8,
                        "color": arrow_color,
                        "linestyle": (0, (1, 3)),
                    },
                )
            edge_plot_indices = [
                i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge
            ]
            if len(edge_plot_indices) > 1:
                edge_coords = coords[edge_plot_indices]
                for i in range(len(edge_coords) - 1):
                    start = edge_coords[i]
                    end = edge_coords[i + 1]
                    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
                    hue = (angle + np.pi) / (2 * np.pi)
                    arrow_color = arrow_cmap(hue)
                    plt.annotate(
                        "",
                        xy=(end[0], end[1]),
                        xytext=(start[0], start[1]),
                        arrowprops={
                            "arrowstyle": "->",
                            "mutation_scale": ARROW_MUTATION_SCALE,
                            "linewidth": 1.6,
                            "color": arrow_color,
                            "linestyle": "-",
                        },
                    )
            edge_only = [i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge]
            if edge_only and len(coords) > 1:
                first_edge = edge_only[0]
                last_edge = edge_only[-1]
                if first_edge != 0:
                    start = coords[0]
                    end = coords[first_edge]
                    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
                    hue = (angle + np.pi) / (2 * np.pi)
                    arrow_color = arrow_cmap(hue)
                    plt.annotate(
                        "",
                        xy=(end[0], end[1]),
                        xytext=(start[0], start[1]),
                        arrowprops={
                            "arrowstyle": "->",
                            "mutation_scale": ARROW_MUTATION_SCALE,
                            "linewidth": 1.6,
                            "color": arrow_color,
                            "linestyle": "-",
                        },
                    )
                if last_edge != len(coords) - 1:
                    start = coords[last_edge]
                    end = coords[len(coords) - 1]
                    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
                    hue = (angle + np.pi) / (2 * np.pi)
                    arrow_color = arrow_cmap(hue)
                    plt.annotate(
                        "",
                        xy=(end[0], end[1]),
                        xytext=(start[0], start[1]),
                        arrowprops={
                            "arrowstyle": "->",
                            "mutation_scale": ARROW_MUTATION_SCALE,
                            "linewidth": 1.6,
                            "color": arrow_color,
                            "linestyle": "-",
                        },
                    )
            plt.annotate(
                str(completion_idx),
                xy=(coords[0, 0], coords[0, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
            )
            plt.annotate(
                str(completion_idx),
                xy=(coords[-1, 0], coords[-1, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "black", "linewidth": 0.6},
            )
        plt.title("t-SNE with Arrows Colored by Angle")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.legend(title="Completion", ncol=2, fontsize=7, title_fontsize=8, loc="best")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/tsne_sequences/hidden_states_tsne_layer_{idx}_angle.png", dpi=200)
        plt.close()

        # Angle-colored arrows plot without edge components
        plt.figure(figsize=(12, 10))
        for completion_idx, step_indices in enumerate(completion_steps):
            if not step_indices:
                continue
            color = colors[completion_idx % len(colors)]
            coords = points_2d[step_indices]
            edge_flags = completion_edge_flags[completion_idx]
            edge_plot_indices = [
                i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge
            ]
            if len(coords):
                edge_plot_indices.extend([0, len(coords) - 1])
            edge_plot_indices = sorted(set(edge_plot_indices))
            edge_coords = coords[edge_plot_indices] if edge_plot_indices else np.array([])
            non_edge_coords = coords[
                [i for i in range(len(coords)) if i not in edge_plot_indices]
            ]
            if len(non_edge_coords):
                plt.scatter(
                    non_edge_coords[:, 0],
                    non_edge_coords[:, 1],
                    s=12,
                    facecolors="none",
                    edgecolors=color,
                    label=f"c{completion_idx}",
                )
            if len(edge_coords):
                plt.scatter(
                    edge_coords[:, 0],
                    edge_coords[:, 1],
                    s=28,
                    color=color,
                    label=None,
                )
            for i in range(1, len(coords)):
                start = coords[i - 1]
                end = coords[i]
                angle = np.arctan2(end[1] - start[1], end[0] - start[0])
                hue = (angle + np.pi) / (2 * np.pi)
                arrow_color = arrow_cmap(hue)
                plt.annotate(
                    "",
                    xy=(end[0], end[1]),
                    xytext=(start[0], start[1]),
                    arrowprops={
                        "arrowstyle": "->",
                        "mutation_scale": ARROW_MUTATION_SCALE,
                        "linewidth": 0.8,
                        "color": arrow_color,
                        "linestyle": (0, (1, 3)),
                    },
                )
            plt.annotate(
                str(completion_idx),
                xy=(coords[0, 0], coords[0, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
            )
            plt.annotate(
                str(completion_idx),
                xy=(coords[-1, 0], coords[-1, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "black", "linewidth": 0.6},
            )
        plt.title("t-SNE with Arrows Colored by Angle (No Edge Components)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.legend(title="Completion", ncol=2, fontsize=7, title_fontsize=8, loc="best")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/tsne_sequences/hidden_states_tsne_layer_{idx}_angle_no_edges.png", dpi=200)
        plt.close()

        self.plot_sequence_tsne_gradient(
            points_2d,
            completion_steps,
            plot_dir=plot_dir,
            idx=idx,
        )

        self.plot_tsne_clusters(
            points_2d,
            points_array,
            completion_steps,
            completion_edge_flags,
            plot_dir=plot_dir,
            idx=idx,
            xlim=xlim,
            ylim=ylim,
            n_sequences=n_completions,
        )

        # Edge-only plot
        plt.figure(figsize=(12, 10))
        for completion_idx, step_indices in enumerate(completion_steps):
            if not step_indices:
                continue
            color = colors[completion_idx % len(colors)]
            coords = points_2d[step_indices]
            edge_flags = completion_edge_flags[completion_idx]
            edge_plot_indices = [
                i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge
            ]
            if len(coords):
                edge_plot_indices.extend([0, len(coords) - 1])
            edge_plot_indices = sorted(set(edge_plot_indices))
            if not edge_plot_indices:
                continue
            edge_coords = coords[edge_plot_indices]
            plt.scatter(
                edge_coords[:, 0],
                edge_coords[:, 1],
                s=28,
                color=color,
                label=f"c{completion_idx}",
            )
            for i in range(len(edge_coords) - 1):
                start = edge_coords[i]
                end = edge_coords[i + 1]
                plt.annotate(
                    "",
                    xy=(end[0], end[1]),
                    xytext=(start[0], start[1]),
                    arrowprops={
                        "arrowstyle": "->",
                        "mutation_scale": ARROW_MUTATION_SCALE,
                        "linewidth": 1.6,
                        "color": color,
                        "linestyle": "-",
                    },
                )
            plt.annotate(
                str(completion_idx),
                xy=(coords[0, 0], coords[0, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
            )
            plt.annotate(
                str(completion_idx),
                xy=(coords[-1, 0], coords[-1, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "black", "linewidth": 0.6},
            )
        plt.title("t-SNE Edge-Only by Completion")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(title="Completion", ncol=2, fontsize=7, title_fontsize=8, loc="best")
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/tsne_sequences/hidden_states_tsne_layer_{idx}_edge_only.png", dpi=200)
        plt.close()

        # Non-edge-only plot
        plt.figure(figsize=(12, 10))
        for completion_idx, step_indices in enumerate(completion_steps):
            if not step_indices:
                continue
            color = colors[completion_idx % len(colors)]
            coords = points_2d[step_indices]
            edge_flags = completion_edge_flags[completion_idx]
            edge_plot_indices = [
                i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge
            ]
            if len(coords):
                edge_plot_indices.extend([0, len(coords) - 1])
            edge_plot_indices = sorted(set(edge_plot_indices))
            non_edge_coords = coords[
                [i for i in range(len(coords)) if i not in edge_plot_indices]
            ]
            if len(non_edge_coords):
                plt.scatter(
                    non_edge_coords[:, 0],
                    non_edge_coords[:, 1],
                    s=12,
                    facecolors="none",
                    edgecolors=color,
                    label=f"c{completion_idx}",
                )
            for i in range(1, len(coords)):
                start = coords[i - 1]
                end = coords[i]
                plt.annotate(
                    "",
                    xy=(end[0], end[1]),
                    xytext=(start[0], start[1]),
                    arrowprops={
                        "arrowstyle": "->",
                        "mutation_scale": ARROW_MUTATION_SCALE,
                        "linewidth": 0.8,
                        "color": color,
                        "linestyle": (0, (1, 3)),
                    },
                )
            plt.annotate(
                str(completion_idx),
                xy=(coords[0, 0], coords[0, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
            )
            plt.annotate(
                str(completion_idx),
                xy=(coords[-1, 0], coords[-1, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "black", "linewidth": 0.6},
            )
        plt.title("t-SNE Without Edge Components")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(title="Completion", ncol=2, fontsize=7, title_fontsize=8, loc="best")
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/tsne_sequences/hidden_states_tsne_layer_{idx}_no_edges.png", dpi=200)
        plt.close()

        self.plot_tsne_density_kde(
            points_2d,
            plot_dir=plot_dir,
            idx=idx,
            xlim=xlim,
            ylim=ylim,
        )

        focus_dir = f"{plot_dir}/tsne_sequences_focus"
        os.makedirs(focus_dir, exist_ok=True)
        for focus_idx in range(n_completions):
            plt.figure(figsize=(12, 10))
            for completion_idx, step_indices in enumerate(completion_steps):
                if not step_indices:
                    continue
                color = colors[completion_idx % len(colors)]
                coords = points_2d[step_indices]
                is_focus = completion_idx == focus_idx
                alpha = 1.0 if is_focus else 0.15
                line_width = 0.8 if is_focus else 0.4
                point_size = 14 if is_focus else 10
                edge_flags = completion_edge_flags[completion_idx]
                edge_plot_indices = [
                    i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge
                ]
                if len(coords):
                    edge_plot_indices.extend([0, len(coords) - 1])
                edge_plot_indices = sorted(set(edge_plot_indices))
                edge_coords = coords[edge_plot_indices] if edge_plot_indices else np.array([])
                non_edge_coords = coords[
                    [i for i in range(len(coords)) if i not in edge_plot_indices]
                ]
                if len(non_edge_coords):
                    plt.scatter(
                        non_edge_coords[:, 0],
                        non_edge_coords[:, 1],
                        s=point_size,
                        facecolors="none",
                        edgecolors=color,
                        alpha=alpha,
                        label=f"c{completion_idx}",
                    )
                if len(edge_coords):
                    plt.scatter(
                        edge_coords[:, 0],
                        edge_coords[:, 1],
                        s=point_size + 10,
                        color=color,
                        alpha=alpha,
                        label=None,
                    )
                for i in range(1, len(coords)):
                    start = coords[i - 1]
                    end = coords[i]
                    plt.annotate(
                        "",
                        xy=(end[0], end[1]),
                        xytext=(start[0], start[1]),
                        arrowprops={
                        "arrowstyle": "->",
                        "mutation_scale": ARROW_MUTATION_SCALE,
                            "linewidth": 0.8,
                            "color": color,
                            "alpha": alpha,
                            "linestyle": (0, (1, 3)),
                        },
                    )
                if is_focus:
                    plt.annotate(
                        str(completion_idx),
                        xy=(coords[0, 0], coords[0, 1]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        color="black",
                        fontsize=9,
                        fontweight="bold",
                    )
                    plt.annotate(
                        str(completion_idx),
                        xy=(coords[-1, 0], coords[-1, 1]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        color="black",
                        fontsize=9,
                        fontweight="bold",
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "black", "linewidth": 0.6},
                    )
                    for point_idx in range(len(coords)):
                        plt.annotate(
                            str(point_idx),
                            xy=(coords[point_idx, 0], coords[point_idx, 1]),
                            xytext=(0, -7),
                            textcoords="offset points",
                            fontsize=5,
                            color="black",
                            ha="center",
                            va="top",
                        )
                    for local_idx in range(len(completion_token_steps[completion_idx]) - 1):
                        step_idx = completion_token_steps[completion_idx][local_idx]
                        token_text = self.tokenizer.decode(
                            [completion_ids[completion_idx, step_idx].item()],
                            skip_special_tokens=False,
                        )
                        prefix_ids = tuple(
                            completion_ids[completion_idx, : step_idx + 1].tolist()
                        )
                        prefix_text, token_text = self.format_prefix_line_and_token(prefix_ids)
                        fontsize = 6
                        fontfamily = "monospace"
                        prefix_width = self.monospace_width_points(prefix_text, fontsize)
                        token_width = self.monospace_width_points(token_text, fontsize)
                        total_width = prefix_width + token_width
                        prefix_offset = (-total_width / 2, 6)
                        token_offset = (-total_width / 2 + prefix_width, 6)
                        if prefix_text:
                            plt.annotate(
                                prefix_text,
                                xy=(coords[local_idx + 1, 0], coords[local_idx + 1, 1]),
                                xytext=prefix_offset,
                                textcoords="offset points",
                                fontsize=fontsize,
                                fontfamily=fontfamily,
                                color="black",
                                ha="left",
                                va="bottom",
                            )
                        plt.annotate(
                            token_text,
                            xy=(coords[local_idx + 1, 0], coords[local_idx + 1, 1]),
                            xytext=token_offset,
                            textcoords="offset points",
                            fontsize=fontsize,
                            fontfamily=fontfamily,
                            color="tab:red",
                            ha="left",
                            va="bottom",
                        )
                edge_plot_indices = [
                    i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge
                ]
                if len(edge_plot_indices) > 1:
                    edge_coords = coords[edge_plot_indices]
                    for i in range(len(edge_coords) - 1):
                        start = edge_coords[i]
                        end = edge_coords[i + 1]
                        plt.annotate(
                            "",
                            xy=(end[0], end[1]),
                            xytext=(start[0], start[1]),
                            arrowprops={
                                "arrowstyle": "->",
                                "mutation_scale": ARROW_MUTATION_SCALE,
                                "linewidth": line_width + 0.9,
                                "color": color,
                                "alpha": alpha,
                                "linestyle": "-",
                            },
                        )
                edge_only = [i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge]
                if edge_only and len(coords) > 1:
                    first_edge = edge_only[0]
                    last_edge = edge_only[-1]
                    if first_edge != 0:
                        plt.annotate(
                            "",
                            xy=(coords[first_edge][0], coords[first_edge][1]),
                            xytext=(coords[0][0], coords[0][1]),
                            arrowprops={
                                "arrowstyle": "->",
                                "mutation_scale": ARROW_MUTATION_SCALE,
                                "linewidth": line_width + 0.9,
                                "color": color,
                                "alpha": alpha,
                                "linestyle": "-",
                            },
                        )
                    if last_edge != len(coords) - 1:
                        plt.annotate(
                            "",
                            xy=(coords[len(coords) - 1][0], coords[len(coords) - 1][1]),
                            xytext=(coords[last_edge][0], coords[last_edge][1]),
                            arrowprops={
                                "arrowstyle": "->",
                                "mutation_scale": ARROW_MUTATION_SCALE,
                                "linewidth": line_width + 0.9,
                                "color": color,
                                "alpha": alpha,
                                "linestyle": "-",
                            },
                        )

            plt.title(f"t-SNE by Completion (focus c{focus_idx})")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.legend(title="Completion", ncol=2, fontsize=7, title_fontsize=8, loc="best")
            plt.tight_layout()
            plt.savefig(
                f"{focus_dir}/hidden_states_tsne_layer_{idx}_focus_c{focus_idx}.png",
                dpi=200,
            )
            plt.close()

    def plot_sequence_pca(
        self,
        points_2d: np.ndarray,
        completion_steps: list[list[int]],
        completion_edge_flags: list[list[bool]],
        plot_dir: str,
        idx: int,
    ):
        os.makedirs(f"{plot_dir}/pca_sequences", exist_ok=True)
        cmap = plt.get_cmap("turbo")
        color_vals = np.linspace(0, 1, max(len(completion_steps), 2))
        colors = [cmap(v) for v in color_vals]

        plt.figure(figsize=(12, 10))
        for completion_idx, step_indices in enumerate(completion_steps):
            if not step_indices:
                continue
            color = colors[completion_idx % len(colors)]
            coords = points_2d[step_indices]
            edge_flags = completion_edge_flags[completion_idx]
            edge_plot_indices = [
                i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge
            ]
            if len(coords):
                edge_plot_indices.extend([0, len(coords) - 1])
            edge_plot_indices = sorted(set(edge_plot_indices))
            edge_coords = coords[edge_plot_indices] if edge_plot_indices else np.array([])
            non_edge_coords = coords[
                [i for i in range(len(coords)) if i not in edge_plot_indices]
            ]
            if len(non_edge_coords):
                plt.scatter(
                    non_edge_coords[:, 0],
                    non_edge_coords[:, 1],
                    s=12,
                    facecolors="none",
                    edgecolors=color,
                    label=f"c{completion_idx}",
                )
            if len(edge_coords):
                plt.scatter(
                    edge_coords[:, 0],
                    edge_coords[:, 1],
                    s=28,
                    color=color,
                    label=None,
                )
            for i in range(1, len(coords)):
                start = coords[i - 1]
                end = coords[i]
                plt.annotate(
                    "",
                    xy=(end[0], end[1]),
                    xytext=(start[0], start[1]),
                    arrowprops={
                        "arrowstyle": "->",
                        "mutation_scale": ARROW_MUTATION_SCALE,
                        "linewidth": 0.8,
                        "color": color,
                        "linestyle": (0, (1, 3)),
                    },
                )
            plt.annotate(
                str(completion_idx),
                xy=(coords[0, 0], coords[0, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
            )
            plt.annotate(
                str(completion_idx),
                xy=(coords[-1, 0], coords[-1, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "black", "linewidth": 0.6},
            )
            edge_plot_indices = [
                i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge
            ]
            if len(edge_plot_indices) > 1:
                edge_coords = coords[edge_plot_indices]
                for i in range(len(edge_coords) - 1):
                    start = edge_coords[i]
                    end = edge_coords[i + 1]
                    plt.annotate(
                        "",
                        xy=(end[0], end[1]),
                        xytext=(start[0], start[1]),
                        arrowprops={
                            "arrowstyle": "->",
                            "mutation_scale": ARROW_MUTATION_SCALE,
                            "linewidth": 1.6,
                            "color": color,
                            "linestyle": "-",
                        },
                    )
            edge_only = [i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge]
            if edge_only and len(coords) > 1:
                first_edge = edge_only[0]
                last_edge = edge_only[-1]
                if first_edge != 0:
                    plt.annotate(
                        "",
                        xy=(coords[first_edge][0], coords[first_edge][1]),
                        xytext=(coords[0][0], coords[0][1]),
                        arrowprops={
                            "arrowstyle": "->",
                            "mutation_scale": ARROW_MUTATION_SCALE,
                            "linewidth": 1.6,
                            "color": color,
                            "linestyle": "-",
                        },
                    )
                if last_edge != len(coords) - 1:
                    plt.annotate(
                        "",
                        xy=(coords[len(coords) - 1][0], coords[len(coords) - 1][1]),
                        xytext=(coords[last_edge][0], coords[last_edge][1]),
                        arrowprops={
                            "arrowstyle": "->",
                            "mutation_scale": ARROW_MUTATION_SCALE,
                            "linewidth": 1.6,
                            "color": color,
                            "linestyle": "-",
                        },
                    )

        plt.title("PCA (2D) of Hidden State Vectors by Completion")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(title="Completion", ncol=2, fontsize=7, title_fontsize=8, loc="best")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/pca_sequences/hidden_states_pca_layer_{idx}.png", dpi=200)
        plt.close()

    def plot_pca_reconstruction_error(self, vectors: np.ndarray, plot_dir: str, idx: int):
        from sklearn.decomposition import PCA
        max_components = min(vectors.shape[0], vectors.shape[1])
        if max_components < 2:
            return
        pca = PCA(n_components=max_components, svd_solver="full")
        pca.fit(vectors)
        explained_ratio = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained_ratio)
        n90 = int(np.searchsorted(cumulative, 0.9, side="left") + 1)
        n99 = int(np.searchsorted(cumulative, 0.99, side="left") + 1)
        os.makedirs(f"{plot_dir}/pca", exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_components + 1), cumulative, marker="o", markersize=3)
        plt.axhline(0.9, color="gray", linestyle="--", linewidth=0.8)
        plt.axhline(0.99, color="gray", linestyle="--", linewidth=0.8)
        plt.axvline(n90, color="gray", linestyle="--", linewidth=0.8)
        plt.axvline(n99, color="gray", linestyle="--", linewidth=0.8)
        plt.annotate(
            f"{n90}",
            xy=(n90, 0.9),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=8,
            color="black",
        )
        plt.annotate(
            f"{n99}",
            xy=(n99, 0.99),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=8,
            color="black",
        )
        plt.title(
            f"PCA Cumulative Explained Variance (dim={vectors.shape[1]}, 90% @ {n90}, 99% @ {n99})"
        )
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance Ratio")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/pca/explained_variance_layer_{idx}.png", dpi=200)
        plt.close()

    def plot_sequence_tsne_gradient(
        self,
        points_2d: np.ndarray,
        completion_steps: list[list[int]],
        plot_dir: str,
        idx: int,
        cmap_name: str = "hsv",
    ):
        os.makedirs(f"{plot_dir}/tsne_sequences_gradient", exist_ok=True)
        cmap = plt.get_cmap(cmap_name)
        plt.figure(figsize=(12, 10))
        for completion_idx, step_indices in enumerate(completion_steps):
            if len(step_indices) < 2:
                continue
            coords = points_2d[step_indices]
            n_steps = len(coords)
            for i in range(1, n_steps):
                start = coords[i - 1]
                end = coords[i]
                color = cmap(i / (n_steps - 1))
                plt.annotate(
                    "",
                    xy=(end[0], end[1]),
                    xytext=(start[0], start[1]),
                    arrowprops={
                        "arrowstyle": "->",
                        "mutation_scale": ARROW_MUTATION_SCALE,
                        "linewidth": 0.6,
                        "color": color,
                    },
                )
            plt.scatter(
                coords[:, 0],
                coords[:, 1],
                s=8,
                color=[cmap(i / (n_steps - 1)) for i in range(n_steps)],
            )
            plt.annotate(
                "0",
                xy=(coords[0, 0], coords[0, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
                fontsize=8,
                fontweight="bold",
            )
            plt.annotate(
                str(completion_idx),
                xy=(coords[-1, 0], coords[-1, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
                fontsize=8,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "black", "linewidth": 0.6},
            )
        plt.title("t-SNE with Position Gradient")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.tight_layout()
        os.makedirs(f"{plot_dir}/tsne_sequences", exist_ok=True)
        plt.savefig(f"{plot_dir}/tsne_sequences/hidden_states_tsne_layer_{idx}_gradient.png", dpi=200)
        plt.close()

    def plot_tsne_clusters(
        self,
        points_2d: np.ndarray,
        vectors: np.ndarray,
        completion_steps: list[list[int]],
        completion_edge_flags: list[list[bool]],
        plot_dir: str,
        idx: int,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        n_sequences: int,
    ):
        from sklearn.cluster import AgglomerativeClustering

        n_clusters = max(2, int(round(len(vectors) / max(n_sequences, 1))) * 2)
        if len(vectors) < n_clusters:
            return

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(vectors)

        plt.figure(figsize=(12, 10))
        cmap = plt.get_cmap("tab10")
        colors = [cmap(label % 10) for label in labels]
        plt.scatter(points_2d[:, 0], points_2d[:, 1], s=12, c=colors)
        crosshair_span = 0.01 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        for cluster_id in range(n_clusters):
            idxs = np.where(labels == cluster_id)[0]
            if len(idxs) == 0:
                continue
            cluster_color = colors[idxs[0]]
            centroid = points_2d[idxs].mean(axis=0)
            plt.plot(
                [centroid[0] - crosshair_span, centroid[0] + crosshair_span],
                [centroid[1], centroid[1]],
                color=cluster_color,
                linewidth=0.8,
            )
            plt.plot(
                [centroid[0], centroid[0]],
                [centroid[1] - crosshair_span, centroid[1] + crosshair_span],
                color=cluster_color,
                linewidth=0.8,
            )
            plt.annotate(
                str(cluster_id),
                xy=(centroid[0], centroid[1]),
                xytext=(crosshair_span * 1.4, crosshair_span * 1.4),
                textcoords="offset points",
                color=cluster_color,
                fontsize=8,
                fontweight="bold",
            )

        # Draw arrows per sequence with cluster-based coloring.
        for completion_idx, step_indices in enumerate(completion_steps):
            if not step_indices:
                continue
            coords = points_2d[step_indices]
            edge_flags = completion_edge_flags[completion_idx]
            edge_plot_indices = [
                i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge
            ]
            if len(coords):
                edge_plot_indices.extend([0, len(coords) - 1])
            edge_plot_indices = sorted(set(edge_plot_indices))
            for i in range(1, len(coords)):
                start = coords[i - 1]
                end = coords[i]
                color = colors[step_indices[i]]
                plt.annotate(
                    "",
                    xy=(end[0], end[1]),
                    xytext=(start[0], start[1]),
                    arrowprops={
                        "arrowstyle": "->",
                        "mutation_scale": ARROW_MUTATION_SCALE,
                        "linewidth": 0.8,
                        "color": color,
                        "linestyle": (0, (1, 3)),
                    },
                )
            if len(edge_plot_indices) > 1:
                edge_coords = coords[edge_plot_indices]
                for i in range(len(edge_coords) - 1):
                    start = edge_coords[i]
                    end = edge_coords[i + 1]
                    color = colors[step_indices[edge_plot_indices[i + 1]]]
                    plt.annotate(
                        "",
                        xy=(end[0], end[1]),
                        xytext=(start[0], start[1]),
                        arrowprops={
                            "arrowstyle": "->",
                            "mutation_scale": ARROW_MUTATION_SCALE,
                            "linewidth": 1.6,
                            "color": color,
                            "linestyle": "-",
                        },
                    )
            edge_only = [i + 1 for i, is_edge in enumerate(edge_flags[:-1]) if is_edge]
            if edge_only and len(coords) > 1:
                first_edge = edge_only[0]
                last_edge = edge_only[-1]
                if first_edge != 0:
                    color = colors[step_indices[first_edge]]
                    plt.annotate(
                        "",
                        xy=(coords[first_edge][0], coords[first_edge][1]),
                        xytext=(coords[0][0], coords[0][1]),
                        arrowprops={
                            "arrowstyle": "->",
                            "mutation_scale": ARROW_MUTATION_SCALE,
                            "linewidth": 1.6,
                            "color": color,
                            "linestyle": "-",
                        },
                    )
                if last_edge != len(coords) - 1:
                    color = colors[step_indices[-1]]
                    plt.annotate(
                        "",
                        xy=(coords[-1][0], coords[-1][1]),
                        xytext=(coords[last_edge][0], coords[last_edge][1]),
                        arrowprops={
                            "arrowstyle": "->",
                            "mutation_scale": ARROW_MUTATION_SCALE,
                            "linewidth": 1.6,
                            "color": color,
                            "linestyle": "-",
                        },
                    )

        plt.title(f"t-SNE Colored by Clusters (k={n_clusters}, target size≈{n_sequences})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.tight_layout()
        os.makedirs(f"{plot_dir}/tsne_sequences", exist_ok=True)
        plt.savefig(
            f"{plot_dir}/tsne_sequences/hidden_states_tsne_layer_{idx}_clusters.png",
            dpi=200,
        )
        plt.close()

        # Cluster plot without edge components
        plt.figure(figsize=(12, 10))
        plt.scatter(points_2d[:, 0], points_2d[:, 1], s=12, c=colors)
        crosshair_span = 0.01 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        for cluster_id in range(n_clusters):
            idxs = np.where(labels == cluster_id)[0]
            if len(idxs) == 0:
                continue
            cluster_color = colors[idxs[0]]
            centroid = points_2d[idxs].mean(axis=0)
            plt.plot(
                [centroid[0] - crosshair_span, centroid[0] + crosshair_span],
                [centroid[1], centroid[1]],
                color=cluster_color,
                linewidth=0.8,
            )
            plt.plot(
                [centroid[0], centroid[0]],
                [centroid[1] - crosshair_span, centroid[1] + crosshair_span],
                color=cluster_color,
                linewidth=0.8,
            )
            plt.annotate(
                str(cluster_id),
                xy=(centroid[0], centroid[1]),
                xytext=(crosshair_span * 1.4, crosshair_span * 1.4),
                textcoords="offset points",
                color=cluster_color,
                fontsize=8,
                fontweight="bold",
            )

        for completion_idx, step_indices in enumerate(completion_steps):
            if not step_indices:
                continue
            coords = points_2d[step_indices]
            for i in range(1, len(coords)):
                start = coords[i - 1]
                end = coords[i]
                color = colors[step_indices[i]]
                plt.annotate(
                    "",
                    xy=(end[0], end[1]),
                    xytext=(start[0], start[1]),
                    arrowprops={
                        "arrowstyle": "->",
                        "mutation_scale": ARROW_MUTATION_SCALE,
                        "linewidth": 0.8,
                        "color": color,
                        "linestyle": (0, (1, 3)),
                    },
                )

        plt.title(f"t-SNE Colored by Clusters (k={n_clusters}, target size≈{n_sequences}) (No Edge Components)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.tight_layout()
        os.makedirs(f"{plot_dir}/tsne_sequences", exist_ok=True)
        plt.savefig(
            f"{plot_dir}/tsne_sequences/hidden_states_tsne_layer_{idx}_clusters_no_edges.png",
            dpi=200,
        )
        plt.close()

    def plot_tsne_density(
        self,
        points_2d: np.ndarray,
        plot_dir: str,
        idx: int,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        bins: int = 150,
    ):
        os.makedirs(f"{plot_dir}/tsne_sequences", exist_ok=True)
        plt.figure(figsize=(12, 10))
        plt.hist2d(points_2d[:, 0], points_2d[:, 1], bins=bins, cmap="inferno")
        plt.title("t-SNE Density (All Points)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.tight_layout()
        plt.savefig(
            f"{plot_dir}/tsne_sequences/hidden_states_tsne_layer_{idx}_density.png",
            dpi=200,
        )
        plt.close()

    def plot_tsne_density_kde(
        self,
        points_2d: np.ndarray,
        plot_dir: str,
        idx: int,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        grid_size: int = 800,
        bandwidth: float = 0.5,
    ):
        from sklearn.neighbors import KernelDensity

        os.makedirs(f"{plot_dir}/tsne_sequences", exist_ok=True)
        x_grid = np.linspace(xlim[0], xlim[1], grid_size)
        y_grid = np.linspace(ylim[0], ylim[1], grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)
        sample = np.column_stack([xx.ravel(), yy.ravel()])

        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(points_2d)
        log_density = kde.score_samples(sample)
        density = np.exp(log_density).reshape(grid_size, grid_size)
        if density.max() > 0:
            density = density / density.max()
            density = density ** 0.5

        plt.figure(figsize=(12, 10))
        plt.imshow(
            density,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            origin="lower",
            cmap="inferno",
            aspect="auto",
        )
        plt.title("t-SNE Density (KDE)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.tight_layout()
        plt.savefig(
            f"{plot_dir}/tsne_sequences/hidden_states_tsne_layer_{idx}_density_kde.png",
            dpi=200,
        )
        plt.close()

    def plot_sequence_umap(
        self,
        hidden_states_tensor: torch.Tensor,
        completion_ids: torch.Tensor,
        idx: int,
        plot_dir: str,
        only_newline_tokens: bool = False,
    ):
        try:
            import umap
        except ImportError:
            print("UMAP not available; install `umap-learn` to enable UMAP plots.")
            return

        os.makedirs(f"{plot_dir}/umap_sequences", exist_ok=True)
        n_completions, max_generated_length, _, hidden_size = hidden_states_tensor.shape
        eos_token_id = self.tokenizer.eos_token_id

        points = []
        completion_steps = []
        completion_token_steps = []
        for completion_idx in range(n_completions):
            step_indices = []
            token_steps = []
            for step_idx in range(max_generated_length):
                token_id = completion_ids[completion_idx, step_idx].item()
                if eos_token_id is not None and token_id == eos_token_id:
                    break
                if only_newline_tokens:
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    if "\n" not in token_text:
                        continue
                vector = hidden_states_tensor[completion_idx, step_idx, idx, :].cpu().numpy()
                step_indices.append(len(points))
                token_steps.append(step_idx)
                points.append(vector)
            completion_steps.append(step_indices)
            completion_token_steps.append(token_steps)

        if not points:
            return

        points_array = np.array(points)
        reducer = umap.UMAP(n_components=2, random_state=0)
        points_2d = reducer.fit_transform(points_array)

        plt.figure(figsize=(12, 10))
        cmap = plt.get_cmap("turbo")
        color_vals = np.linspace(0, 1, max(n_completions, 2))
        colors = [cmap(v) for v in color_vals]
        for completion_idx, step_indices in enumerate(completion_steps):
            if not step_indices:
                continue
            color = colors[completion_idx % len(colors)]
            coords = points_2d[step_indices]
            plt.scatter(coords[:, 0], coords[:, 1], s=12, color=color, label=f"c{completion_idx}")
            plt.annotate(
                str(completion_idx),
                xy=(coords[0, 0], coords[0, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
            )
            plt.annotate(
                str(completion_idx),
                xy=(coords[-1, 0], coords[-1, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "black", "linewidth": 0.6},
            )
            for i in range(len(step_indices) - 1):
                start = coords[i]
                end = coords[i + 1]
                plt.annotate(
                    "",
                    xy=(end[0], end[1]),
                    xytext=(start[0], start[1]),
                    arrowprops={"arrowstyle": "->", "mutation_scale": ARROW_MUTATION_SCALE, "linewidth": 0.6, "color": color},
                )

        plt.title("UMAP of Hidden State Vectors by Completion")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(title="Completion", ncol=2, fontsize=7, title_fontsize=8, loc="best")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/umap_sequences/hidden_states_umap_layer_{idx}.png", dpi=200)
        plt.close()

        focus_dir = f"{plot_dir}/umap_sequences_focus"
        os.makedirs(focus_dir, exist_ok=True)
        for focus_idx in range(n_completions):
            plt.figure(figsize=(12, 10))
            for completion_idx, step_indices in enumerate(completion_steps):
                if not step_indices:
                    continue
                color = colors[completion_idx % len(colors)]
                coords = points_2d[step_indices]
                is_focus = completion_idx == focus_idx
                alpha = 1.0 if is_focus else 0.15
                line_width = 0.8 if is_focus else 0.4
                point_size = 14 if is_focus else 10
                plt.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    s=point_size,
                    color=color,
                    alpha=alpha,
                    label=f"c{completion_idx}",
                )
                if is_focus:
                    plt.annotate(
                        str(completion_idx),
                        xy=(coords[0, 0], coords[0, 1]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        color="black",
                        fontsize=9,
                        fontweight="bold",
                    )
                    plt.annotate(
                        str(completion_idx),
                        xy=(coords[-1, 0], coords[-1, 1]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        color="black",
                        fontsize=9,
                        fontweight="bold",
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "black", "linewidth": 0.6},
                    )
                    for point_idx in range(len(coords)):
                        plt.annotate(
                            str(point_idx),
                            xy=(coords[point_idx, 0], coords[point_idx, 1]),
                            xytext=(0, -7),
                            textcoords="offset points",
                            fontsize=5,
                            color="black",
                            ha="center",
                            va="top",
                        )
                    for local_idx in range(len(completion_token_steps[completion_idx]) - 1):
                        step_idx = completion_token_steps[completion_idx][local_idx]
                        token_text = self.tokenizer.decode(
                            [completion_ids[completion_idx, step_idx].item()],
                            skip_special_tokens=False,
                        )
                        if "\n" not in token_text:
                            continue
                        prefix_ids = tuple(
                            completion_ids[completion_idx, : step_idx + 1].tolist()
                        )
                        prefix_text, token_text = self.format_prefix_line_and_token(prefix_ids)
                        fontsize = 6
                        fontfamily = "monospace"
                        prefix_width = self.monospace_width_points(prefix_text, fontsize)
                        token_width = self.monospace_width_points(token_text, fontsize)
                        total_width = prefix_width + token_width
                        prefix_offset = (-total_width / 2, 6)
                        token_offset = (-total_width / 2 + prefix_width, 6)
                        if prefix_text:
                            plt.annotate(
                                prefix_text,
                                xy=(coords[local_idx + 1, 0], coords[local_idx + 1, 1]),
                                xytext=prefix_offset,
                                textcoords="offset points",
                                fontsize=fontsize,
                                fontfamily=fontfamily,
                                color="black",
                                ha="left",
                                va="bottom",
                            )
                        plt.annotate(
                            token_text,
                            xy=(coords[local_idx + 1, 0], coords[local_idx + 1, 1]),
                            xytext=token_offset,
                            textcoords="offset points",
                            fontsize=fontsize,
                            fontfamily=fontfamily,
                            color="tab:red",
                            ha="left",
                            va="bottom",
                        )
                for i in range(len(step_indices) - 1):
                    start = coords[i]
                    end = coords[i + 1]
                    plt.annotate(
                        "",
                        xy=(end[0], end[1]),
                        xytext=(start[0], start[1]),
                        arrowprops={
                            "arrowstyle": "->",
                            "mutation_scale": ARROW_MUTATION_SCALE,
                            "linewidth": line_width,
                            "color": color,
                            "alpha": alpha,
                        },
                    )

            plt.title(f"UMAP by Completion (focus c{focus_idx})")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.legend(title="Completion", ncol=2, fontsize=7, title_fontsize=8, loc="best")
            plt.tight_layout()
            plt.savefig(
                f"{focus_dir}/hidden_states_umap_layer_{idx}_focus_c{focus_idx}.png",
                dpi=200,
            )
            plt.close()


    def cluster_embeddings(self, embeddings: dict, n_clusters: int, idx: int, plot_dir: str, metric: str = 'cosine'):
        hidden_state_vectors = np.array(list(embeddings.values()))
        prefixes = list(embeddings.keys())

        linked = linkage(
            hidden_state_vectors, 
            # method='single', 
            # method='complete', 
            # method='centroid', 
            method='average', 
            metric=metric,
            optimal_ordering=True
        )
        labelList = [self.get_last_n_tokens_str(prefix, 5) for prefix in prefixes]
        plt.figure(figsize=(20, 14))
        dendrogram(linked,
            orientation='left',
            labels=labelList,
            distance_sort=True,
            show_leaf_counts=True
        )
        plt.title("Hierarchical Clustering of Hidden State Vectors")
        plt.xlabel("Prefix")
        plt.ylabel("Distance")
        # make sure there's enough space to show the labels
        plt.tight_layout()
        os.makedirs(f"{plot_dir}/dendrogram", exist_ok=True)
        plt.savefig(f"{plot_dir}/dendrogram/hidden_states_dendrogram_{idx}_{metric}.png", dpi=200)
        plt.close()
    
    def find_nearest_neighbors(
        self,
        embeddings: dict,
        prefix_meta: dict,
        n_completions: int,
        n_neighbors: int,
        index: int,
        plot_dir: str,
        metric: str = 'cosine',
    ):
        # for each hidden state vector, find the n_neighbors nearest neighbors in the embedding space and print their corresponding prefixes
        from sklearn.neighbors import NearestNeighbors
        hidden_state_vectors = np.array(list(embeddings.values()))
        prefixes = list(embeddings.keys())
        max_neighbors = min(len(hidden_state_vectors), n_completions * n_neighbors)
        nbrs = NearestNeighbors(n_neighbors=max_neighbors, algorithm='auto', metric=metric).fit(hidden_state_vectors)
        distances, indices = nbrs.kneighbors(hidden_state_vectors)
        # make sure the plot directory exists
        os.makedirs(f"{plot_dir}/nn", exist_ok=True)
        # save nearest neighbors to file
        nn_file = f"{plot_dir}/nn/nearest_neighbors_layer_{index}_{metric}.txt"
        with open(nn_file, "w") as f:
            for i, prefix in enumerate(prefixes):
                prefix_marker = self.format_prefix_with_marker(prefix)
                completion_idx, step_idx = prefix_meta.get(prefix, (-1, -1))
                f.write(f"Prefix: ({completion_idx}, {step_idx}) `{prefix_marker}`\n")
                f.write("Nearest neighbors:\n")
                added = 0
                for j in range(1, max_neighbors):
                    neighbor_idx = indices[i, j]
                    neighbor_prefix = prefixes[neighbor_idx]
                    n_completion_idx, n_step_idx = prefix_meta.get(neighbor_prefix, (-1, -1))
                    if n_completion_idx == completion_idx:
                        continue
                    neighbor_marker = self.format_prefix_with_marker(neighbor_prefix)
                    f.write(
                        f"  d:{distances[i, j]:.4f} ({n_completion_idx}, {n_step_idx}) `{neighbor_marker}`\n"
                    )
                    added += 1
                    if added >= n_neighbors:
                        break
                f.write("\n\n")

    def print_completion_neighbors(
        self,
        completion_ids: torch.Tensor,
        hidden_states_tensor: torch.Tensor,
        plot_dir: str,
        n_neighbors: int = 5,
        metric: str = "cosine",
    ):
        from sklearn.neighbors import NearestNeighbors

        n_completions, max_generated_length = completion_ids.shape
        num_layers = hidden_states_tensor.shape[2]
        layer_idx = 14
        if layer_idx >= num_layers:
            raise ValueError(f"Requested layer {layer_idx}, but model has {num_layers} layers.")
        eos_token_id = self.tokenizer.eos_token_id

        completion_0_ids = []
        for step_idx in range(max_generated_length):
            token_id = completion_ids[0, step_idx].item()
            if eos_token_id is not None and token_id == eos_token_id:
                break
            completion_0_ids.append(token_id)

        completion_0_text = self.tokenizer.decode(completion_0_ids, skip_special_tokens=False)
        completion_0_tokens = self.tokenizer.convert_ids_to_tokens(completion_0_ids)
        completion_0_pairs = list(zip(completion_0_ids, completion_0_tokens))

        nn_dir = f"{plot_dir}/nn_completion"
        os.makedirs(nn_dir, exist_ok=True)
        nn_file = f"{nn_dir}/completion_0_neighbors_{metric}.txt"

        candidate_vectors = []
        candidate_meta = []
        for completion_idx in range(1, n_completions):
            for step_idx in range(max_generated_length):
                token_id = completion_ids[completion_idx, step_idx].item()
                if eos_token_id is not None and token_id == eos_token_id:
                    break
                vector = hidden_states_tensor[completion_idx, step_idx, layer_idx, :].cpu().numpy()
                token_str = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                candidate_vectors.append(vector)
                candidate_meta.append((completion_idx, step_idx, token_id, token_str))

        if not candidate_vectors:
            with open(nn_file, "w") as f:
                f.write("No candidate tokens from other completions to compare against.\n")
            return

        k = min(n_neighbors, len(candidate_vectors))
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric=metric).fit(candidate_vectors)

        with open(nn_file, "w") as f:
            f.write("completion_0:\n" + completion_0_text + "\n")
            f.write(f"completion_0_token_pairs: {completion_0_pairs}\n\n")
            for step_idx, token_id in enumerate(completion_0_ids):
                query_vector = hidden_states_tensor[0, step_idx, layer_idx, :].cpu().numpy().reshape(1, -1)
                distances, indices = nbrs.kneighbors(query_vector)
                token_str = completion_0_tokens[step_idx]
                f.write(f"step {step_idx}: {token_id} {token_str}\n")
                for dist, cand_idx in zip(distances[0], indices[0]):
                    c_idx, s_idx, c_token_id, c_token_str = candidate_meta[cand_idx]
                    f.write(f"  nn: c{c_idx} s{s_idx} {c_token_id} {c_token_str} d:{dist}\n")
                f.write("\n")
        

    def plot_hidden_states(
        self,
        text_file: str,
        n_completions: int,
        metric: str = 'cosine',
        max_new_tokens: int = 32,
        only_newline_tokens: bool = False,
    ):
        with open(text_file, "r") as f:
            prompt = f.read()
        
        prompt_name = text_file.split("/")[-1].split(".")[0]

        hidden_states_dict = self.get_hidden_states(prompt, n_completions, max_new_tokens=max_new_tokens)
        prompt_ids = hidden_states_dict["prompt_ids"]
        print(self.tokenizer.batch_decode(prompt_ids[0]))
        completion_ids = hidden_states_dict["completions_ids"]
        hidden_states_tensor = hidden_states_dict["hidden_states"]

        # for each layer, we want to get the hidden state for each completion and each generation step,
        # that is, for each layer, we want to get n_completions x max_generated_length vectors of size hidden_size
        # we consider these as embeddings and we want to cluster and plot them
        
        max_generated_length = hidden_states_tensor.shape[1]
        num_layers = hidden_states_tensor.shape[2]
        hidden_size = hidden_states_tensor.shape[3]

        # create a subdir in `plots` to save the plots for this test
        tms = timestamp()
        plot_dir = f"plots/{tms}_{prompt_name}"
        os.makedirs(plot_dir, exist_ok=True)

        # save completions
        completions_dir = f"{plot_dir}/completions"
        os.makedirs(completions_dir, exist_ok=True)
        all_completions_content = []
        for i in range(n_completions):
            completion = self.tokenizer.decode(completion_ids[i])
            with open(f"{completions_dir}/completion_{i}.txt", "w") as f:
                f.write(completion)
            all_completions_content.append(f"--- COMPLETION {i} ---\n{completion}\n")
        
        with open(f"{completions_dir}/all_completions.txt", "w") as f:
            f.write("\n".join(all_completions_content))

        self.print_completion_neighbors(completion_ids, hidden_states_tensor, plot_dir=plot_dir, n_neighbors=5, metric=metric)

        target_layer_idx = 14
        if target_layer_idx >= num_layers:
            raise ValueError(f"Requested layer {target_layer_idx}, but model has {num_layers} layers.")

        # PCA analysis using all tokens at target layer
        all_vectors = []
        completion_steps = []
        completion_edge_flags = []
        for completion_idx in range(n_completions):
            step_indices = []
            edge_flags = []
            for step_idx in range(max_generated_length):
                token_id = completion_ids[completion_idx, step_idx].item()
                if self.tokenizer.eos_token_id is not None and token_id == self.tokenizer.eos_token_id:
                    break
                vector = hidden_states_tensor[completion_idx, step_idx, target_layer_idx, :].cpu().numpy()
                step_indices.append(len(all_vectors))
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                edge_flags.append("\n" in token_text)
                all_vectors.append(vector)
            completion_steps.append(step_indices)
            completion_edge_flags.append(edge_flags)

        run_pca = False
        if run_pca and all_vectors:
            vectors = np.array(all_vectors)
            self.plot_pca_reconstruction_error(vectors, plot_dir=plot_dir, idx=target_layer_idx)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, svd_solver="full")
            points_2d = pca.fit_transform(vectors)
            self.plot_sequence_pca(
                points_2d,
                completion_steps,
                completion_edge_flags,
                plot_dir=plot_dir,
                idx=target_layer_idx,
            )

        for layer_idx in [target_layer_idx]:
            print("Processing layer", layer_idx)
            embeddings = {}
            prefix_meta = {}
            layer_hidden_states = hidden_states_tensor[:, :, layer_idx, :]
            # layer_hidden_states has shape (n_completions, max_generated_length, hidden_size)
            # we want to reshape it to (n_completions * max_generated_length, hidden_size)
            for completion_idx in range(n_completions):
                for step_idx in range(max_generated_length):
                    # if last token is eos, skip
                    if completion_ids[completion_idx, step_idx] == self.tokenizer.eos_token_id:
                        continue
                    # # if last token does not contain a newline, skip
                    # if "\n" not in self.tokenizer.decode(completion_ids[completion_idx, step_idx]):
                    #     continue
                    completion_prefix_ids = completion_ids[completion_idx, :step_idx+1]
                    # print(self.tokenizer.batch_decode(completion_prefix_ids))
                    assert completion_prefix_ids.shape == (step_idx + 1,)
                    hidden_state_vector = layer_hidden_states[completion_idx, step_idx, :]
                    assert hidden_state_vector.shape == (hidden_size,)
                    # convert the prefix ids to a tuple of ints to use as a key in the dictionary
                    prefix_key = tuple(completion_prefix_ids.cpu().numpy())
                    if prefix_key not in embeddings:
                        embeddings[prefix_key] = hidden_state_vector.cpu().numpy()
                        prefix_meta[prefix_key] = (completion_idx, step_idx)
                    else:
                        # make sure the hidden state vector is the same as the one already in the dictionary
                        existing_vector = embeddings[prefix_key]
                        assert np.allclose(existing_vector, hidden_state_vector.cpu().numpy()), f"Hidden state vector for prefix {prefix_key} is different from the existing vector in the dictionary"
            # print(f"Layer {layer_idx}: {len(embeddings)} unique prefixes")

            self.plot_embeddings(embeddings, layer_idx, plot_dir=plot_dir)
            self.plot_sequence_tsne(
                hidden_states_tensor,
                completion_ids,
                layer_idx,
                plot_dir=plot_dir,
                only_newline_tokens=False,
            )
            run_umap = False
            if run_umap:
                self.plot_sequence_umap(
                    hidden_states_tensor,
                    completion_ids,
                    layer_idx,
                    plot_dir=plot_dir,
                    only_newline_tokens=only_newline_tokens,
                )
            run_dendrogram = False
            if run_dendrogram:
                self.cluster_embeddings(embeddings, n_clusters=5, idx=layer_idx, plot_dir=plot_dir, metric=metric)
            self.find_nearest_neighbors(
                embeddings,
                prefix_meta,
                n_completions,
                n_neighbors=5,
                index=layer_idx,
                plot_dir=plot_dir,
                metric=metric,
            )
        

def test_attention(
    model_id: str = None,
    text_file: str = None,
    n_completions: int = 10,
    metric: str = "cosine",
    max_new_tokens: int = 32,
    only_newline_tokens: bool = False,
):
    # # text_file = "test.txt"
    # # text_file = "prompts/dijkstra.txt"
    # # text_file = "prompts/dijkstra2.txt"
    # # text_file = "prompts/gcd.txt"
    # # text_file = "prompts/gcd2.txt"
    # # text_file = "prompts/hash.txt"
    # text_file = "prompts/duplicate.txt"

    # # with open(text_file, "r") as f:
    # #     text = f.read()

    # # model_id = "Qwen/Qwen2.5-Coder-0.5B"
    # model_id = "Qwen/Qwen2.5-Coder-3B"
    # # model_id = "Qwen/Qwen2.5-Coder-7B"

    # # n_completions = 14
    # # n_completions = 35
    # n_completions = 49

    attention_test = AttentionTest(model_id)
    # attention_test.generate(text)
    # attention_test.get_hidden_states(text, n_completions=n_completions)
    attention_test.plot_hidden_states(
        text_file,
        n_completions=n_completions,
        metric=metric,
        max_new_tokens=max_new_tokens,
        only_newline_tokens=only_newline_tokens,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--text_file", type=str)
    parser.add_argument("--n_completions", type=int, default=10)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--only_newline_tokens", action="store_true")
    args = parser.parse_args()
    test_attention(
        model_id=args.model_id,
        text_file=args.text_file,
        n_completions=args.n_completions,
        metric=args.metric,
        max_new_tokens=args.max_new_tokens,
        only_newline_tokens=args.only_newline_tokens,
    )
