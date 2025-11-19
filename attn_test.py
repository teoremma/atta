from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # print(inputs.shape)
        outputs = self.model.generate(
            **inputs,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            output_attentions=True,
            max_new_tokens=7,
            do_sample=True,
            num_return_sequences=10,
        )

        # assert isinstance(outputs, GenerateOutput) 

        # Print the generated tokens and their corresponding attention scores
        sequence = outputs.sequences[0]
        # tokens = self.tokenizer.batch_decode(sequence)
        scores = outputs.scores[0]
        hiddent_states = outputs.hidden_states[0]
        attentions = outputs.attentions[0]

        for sequence in outputs.sequences:
            tokens = self.tokenizer.batch_decode(sequence)
            print(tokens)
        pass

    def get_hidden_states(self, prompt: str, n_completions: int):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]

        max_new_tokens = 16

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            num_return_sequences=n_completions,
            stop_strings=["\n"],
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

        outputs = self.model.generate(
            **inputs,
            generation_config=generation_config,
            tokenizer=self.tokenizer,
        )


        sequences = outputs.sequences
        # sequences has shape (n_completions, sequence_length)
        assert len(sequences.shape) == 2, f"Expected sequences to have shape (n_completions, sequence_length), but got {sequences.shape}"
        assert sequences.shape[0] == n_completions, f"Expected {n_completions} completions, but got {sequences.shape[0]}"
        # sequences = sequences[:, len(inputs.input_ids[0]):]  # Remove the prompt tokens
        sequences = sequences[:, prompt_length:]  # Remove the prompt tokens
        assert sequences.shape[1] <= max_new_tokens, f"Expected generated sequence length to be at most {max_new_tokens}, but got {sequences.shape[1]}"
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

    def plot_embeddings(self, embeddings: dict, idx: int):
        # embeddings is a dictionary mapping from prefix (tuple of token ids) to hidden state vector (numpy array)

        # for prefix, hidden_state_vector in embeddings.items():
        #     print(f"Prefix: {self.tokenizer.batch_decode(prefix)}, Hidden state vector shape: {hidden_state_vector.shape}")

        # use t-sne to reduce the dimensionality of the hidden state vectors to 2D and plot them
        # make sure to label the points with the corresponding prefix
        hidden_state_vectors = np.array(list(embeddings.values()))
        prefixes = list(embeddings.keys())
        tsne = TSNE(n_components=2, random_state=0, perplexity=10)
        hidden_state_vectors_2d = tsne.fit_transform(hidden_state_vectors)
        plt.figure(figsize=(10, 10))
        for i, prefix in enumerate(prefixes):
            plt.scatter(hidden_state_vectors_2d[i, 0], hidden_state_vectors_2d[i, 1])
            plt.annotate("".join(self.tokenizer.batch_decode(prefix)).replace("\n", "\\n"), (hidden_state_vectors_2d[i, 0], hidden_state_vectors_2d[i, 1]))
        plt.title("t-SNE of Hidden State Vectors")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.savefig(f"plots/tsne/hidden_states_layer_{idx}.png")
        plt.close()
        

    def cluster_embeddings(self, embeddings: dict, n_clusters: int, idx: int):
        # use hierarchical clustering to cluster the hidden state vectors into n_clusters clusters
        from sklearn.cluster import AgglomerativeClustering
        hidden_state_vectors = np.array(list(embeddings.values()))
        prefixes = list(embeddings.keys())
        # clustering = AgglomerativeClustering(n_clusters=n_clusters)
        # cluster_labels = clustering.fit_predict(hidden_state_vectors)
        # clusters = {}
        # for i, label in enumerate(cluster_labels):
        #     if label not in clusters:
        #         clusters[label] = []
        #     clusters[label].append(prefixes[i])

        # now, plot a dendrogram of the clusters
        from scipy.cluster.hierarchy import dendrogram, linkage
        linked = linkage(
            hidden_state_vectors, 
            # method='single', 
            method='complete', 
            optimal_ordering=True
        )
        labelList = ["".join(self.tokenizer.batch_decode(prefix)).replace("\n", "\\n") for prefix in prefixes]
        plt.figure(figsize=(20, 14))
        dendrogram(linked,
            # orientation='top',
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
        plt.savefig(f"plots/dendrogram/hidden_states_dendrogram_{idx}.png")
        plt.close()
    
    def find_nearest_neighbors(self, embeddings: dict, n_neighbors: int, index: int):
        # for each hidden state vector, find the n_neighbors nearest neighbors in the embedding space and print their corresponding prefixes
        from sklearn.neighbors import NearestNeighbors
        hidden_state_vectors = np.array(list(embeddings.values()))
        prefixes = list(embeddings.keys())
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(hidden_state_vectors)
        distances, indices = nbrs.kneighbors(hidden_state_vectors)
        # save nearest neighbors to file
        nn_file = f"plots/nn/nearest_neighbors_layer_{index}.txt"
        with open(nn_file, "w") as f:
            for i, prefix in enumerate(prefixes):
                prefix_toks = self.tokenizer.batch_decode(prefix)
                prefix_str = "".join(prefix_toks).replace("\n", "\\n")
                f.write(f"Prefix: {prefix_str}    <{prefix_toks}>\n")
                f.write(f"Nearest neighbors:\n")
                for j in range(1, n_neighbors + 1):
                    neighbor_idx = indices[i, j]
                    neighbor_prefix = prefixes[neighbor_idx]
                    neighbor_toks = self.tokenizer.batch_decode(neighbor_prefix)
                    neighbor_str = "".join(neighbor_toks).replace("\n", "\\n")
                    f.write(f"  {neighbor_str}    <{neighbor_toks}>    d:{distances[i, j]}\n")
                f.write("\n\n")
        

    def plot_hidden_states(self, prompt: str, n_completions: int):
        hidden_states_dict = self.get_hidden_states(prompt, n_completions)
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

            self.plot_embeddings(embeddings, layer_idx)
            self.cluster_embeddings(embeddings, n_clusters=5, idx=layer_idx)
            self.find_nearest_neighbors(embeddings, n_neighbors=5, index=layer_idx)
        

def test_attention():
    # text_file = "test.txt"
    text_file = "prompts/dijkstra.txt"
    with open(text_file, "r") as f:
        text = f.read()

    # model_id = "Qwen/Qwen2.5-Coder-0.5B"
    # model_id = "Qwen/Qwen2.5-Coder-3B"
    model_id = "Qwen/Qwen2.5-Coder-7B"

    # n_completions = 14
    # n_completions = 35
    n_completions = 49

    attention_test = AttentionTest(model_id)
    # attention_test.generate(text)
    # attention_test.get_hidden_states(text, n_completions=n_completions)
    attention_test.plot_hidden_states(text, n_completions=n_completions)

if __name__ == "__main__":
    test_attention()