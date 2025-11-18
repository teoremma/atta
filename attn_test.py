from transformers import AutoTokenizer, AutoModelForCausalLM

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

def test_attention():
    # text_file = "test.txt"
    text_file = "prompts/dijkstra.txt"
    with open(text_file, "r") as f:
        text = f.read()

    model_id = "Qwen/Qwen2.5-Coder-0.5B"

    attention_test = AttentionTest(model_id)
    attention_test.generate(text)

if __name__ == "__main__":
    test_attention()