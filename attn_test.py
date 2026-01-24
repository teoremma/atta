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
    671,
    3190,
    4210,
    7129,
    11166,
    11456,
    12599,
)
LOGIT_BIAS = {token_id: LOGIT_BIAS_VALUE for token_id in LOGIT_BIAS_TOKEN_IDS}

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
    
    def find_nearest_neighbors(self, embeddings: dict, n_neighbors: int, index: int, plot_dir: str, metric: str = 'cosine'):
        # for each hidden state vector, find the n_neighbors nearest neighbors in the embedding space and print their corresponding prefixes
        from sklearn.neighbors import NearestNeighbors
        hidden_state_vectors = np.array(list(embeddings.values()))
        prefixes = list(embeddings.keys())
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto', metric=metric).fit(hidden_state_vectors)
        distances, indices = nbrs.kneighbors(hidden_state_vectors)
        # make sure the plot directory exists
        os.makedirs(f"{plot_dir}/nn", exist_ok=True)
        # save nearest neighbors to file
        nn_file = f"{plot_dir}/nn/nearest_neighbors_layer_{index}_{metric}.txt"
        with open(nn_file, "w") as f:
            for i, prefix in enumerate(prefixes):
                prefix_toks = self.tokenizer.batch_decode(prefix)
                prefix_str = self.get_last_n_tokens_str(prefix, 5)
                f.write(f"Prefix: {prefix_str}    <{prefix_toks}>\n")
                f.write(f"Nearest neighbors:\n")
                for j in range(1, n_neighbors + 1):
                    neighbor_idx = indices[i, j]
                    neighbor_prefix = prefixes[neighbor_idx]
                    neighbor_toks = self.tokenizer.batch_decode(neighbor_prefix)
                    neighbor_str = self.get_last_n_tokens_str(neighbor_prefix, 5)
                    f.write(f"  {neighbor_str}    <{neighbor_toks}>    d:{distances[i, j]}\n")
                f.write("\n\n")
        

    def plot_hidden_states(self, text_file: str, n_completions: int, metric: str = 'cosine', max_new_tokens: int = 32):
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

        for layer_idx in range(num_layers):            
            print("Processing layer", layer_idx)
            embeddings = {}
            layer_hidden_states = hidden_states_tensor[:, :, layer_idx, :]
            # layer_hidden_states has shape (n_completions, max_generated_length, hidden_size)
            # we want to reshape it to (n_completions * max_generated_length, hidden_size)
            for completion_idx in range(n_completions):
                for step_idx in range(max_generated_length):
                    # if last token is eos, skip
                    if completion_ids[completion_idx, step_idx] == self.tokenizer.eos_token_id:
                        continue
                    # if last token does not contain a newline, skip
                    if "\n" not in self.tokenizer.decode(completion_ids[completion_idx, step_idx]):
                        continue
                    completion_prefix_ids = completion_ids[completion_idx, :step_idx+1]
                    # print(self.tokenizer.batch_decode(completion_prefix_ids))
                    assert completion_prefix_ids.shape == (step_idx + 1,)
                    hidden_state_vector = layer_hidden_states[completion_idx, step_idx, :]
                    assert hidden_state_vector.shape == (hidden_size,)
                    # convert the prefix ids to a tuple of ints to use as a key in the dictionary
                    prefix_key = tuple(completion_prefix_ids.cpu().numpy())
                    if prefix_key not in embeddings:
                        embeddings[prefix_key] = hidden_state_vector.cpu().numpy()
                    else:
                        # make sure the hidden state vector is the same as the one already in the dictionary
                        existing_vector = embeddings[prefix_key]
                        assert np.allclose(existing_vector, hidden_state_vector.cpu().numpy()), f"Hidden state vector for prefix {prefix_key} is different from the existing vector in the dictionary"
            # print(f"Layer {layer_idx}: {len(embeddings)} unique prefixes")

            self.plot_embeddings(embeddings, layer_idx, plot_dir=plot_dir)
            self.cluster_embeddings(embeddings, n_clusters=5, idx=layer_idx, plot_dir=plot_dir, metric=metric)
            self.find_nearest_neighbors(embeddings, n_neighbors=5, index=layer_idx, plot_dir=plot_dir, metric=metric)
        

def test_attention(model_id: str = None, text_file: str = None, n_completions: int = 10, metric: str = "cosine", max_new_tokens: int = 32):
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
    attention_test.plot_hidden_states(text_file, n_completions=n_completions, metric=metric, max_new_tokens=max_new_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--text_file", type=str)
    parser.add_argument("--n_completions", type=int, default=10)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()
    test_attention(model_id=args.model_id, text_file=args.text_file, n_completions=args.n_completions, metric=args.metric, max_new_tokens=args.max_new_tokens)
