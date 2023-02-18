import dataclasses
import time
import typing as tp
import random

import onnxruntime
from matplotlib import pyplot as plt
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from tqdm import notebook

@dataclasses.dataclass
class ModelTester:
    model_name: str
    model_func: callable
    times: tp.List[float] = dataclasses.field(default_factory=list)
    outputs: tp.List[str] = dataclasses.field(default_factory=list)
    perplexities: tp.List[float] = dataclasses.field(default_factory=list)

    @staticmethod
    def perplexity(targets, seq_logits):
        """Perplexity for a fixed length sequence given targets and logits.

        Taken from: https://huggingface.co/docs/transformers/perplexity#perplexity-of-fixedlength-models.

        Args:
            targets: _description_
            seq_logits: _description_
        """
        log_prob_sum = 0
        for target, logits in zip(targets, seq_logits):
            log_prob_sum += np.log2(torch.softmax(logits[0], dim=-1)[:, target])
        return np.exp(-1 * (1/len(targets)) * log_prob_sum).item()

    @staticmethod
    def generate_random_input(length: int, tokenizer: PreTrainedTokenizer) -> str:
        """Generate a random input for prompting during the speed test.
        
        This seems to be a common procedure in other benchmarking code.
        
        
        Args:
            length: The length of the input *in tokens*.
            tokenizer: The tokenizer that will be used to encode the input prior to inference."""

        return tokenizer.decode([random.randint(0, tokenizer.vocab_size-1) for _ in range(length)])

    def run_perplexity_test(self, test_sequences, tokenizer: PreTrainedTokenizer):
        for sequence in notebook.tqdm(test_sequences):
            tokenized_sequence = tokenizer(sequence)['input_ids']
            seq_logits = []
            targets = tokenized_sequence[1:]
            for i in range(1, len(tokenized_sequence)):
                ipt = tokenizer.decode(tokenized_sequence[:i])
                _, logits = self.model_func(ipt, max_len=1, perf=True)
                seq_logits.append(logits)

        self.perplexities.append(self.perplexity(targets, seq_logits))

    def run_speed_test(self, n_tests: int, prompt_length: int, max_generation_length: int, tokenizer: PreTrainedTokenizer) -> None:
        for _ in notebook.tqdm(range(n_tests)):
            prompt = self.generate_random_input(length=prompt_length, tokenizer=tokenizer)
            start = time.time()
            
            out = self.model_func(prompt, max_generation_length)
            t = time.time() - start
            self.times.append(t)
            self.outputs.append(out)

def sample_dataset(
    dataset: any,
    n_articles: int,
    min_length_chars: int,
    max_length_chars: int,
) -> tp.List[str]:

    article_indexes = list(range(len(dataset['train'])))
    articles = []
    for _ in range(n_articles):
        while True and article_indexes:
            i = article_indexes.pop(random.randint(0, len(article_indexes) - 1))
            if len(dataset["train"][i]['text']) > min_length_chars:
                articles.append(dataset['train'][i]['text'][:max_length_chars])
                break
 
    
    if len(articles) != n_articles:
        print(f"WARNING: Only found {len(articles)} articles greater than {min_length_chars} characters.")
    
    return articles

def plot_test_results(model_testers: tp.List[ModelTester], result_type: str) -> None:

    # Initialize graph information.
    fig, ax = plt.subplots()
    x_pos = np.arange(len(model_testers))

    # Calculate summary data for error plotting.
    means = []
    errors = []
    for model_tester in model_testers:
        results = model_tester.__dict__[result_type]
        # Take the mean of all reported inference times.
        means.append(np.mean(results))
        # Error is represented as standard error of the mean.
        errors.append(np.std(results) / np.sqrt(len(results)))

    # Plot and label
    ax.bar(x_pos, means, yerr=errors, align="center")
    ax.set_xticks(
        x_pos, labels=[model_tester.model_name for model_tester in model_testers]
    )
    ax.set_ylabel(f"Inference {result_type}")
    ax.set_title(f"{result_type} comparison of GPT2 inference implementations.")

    plt.show()

   


# TODO: Does preloading the models into the function have a tangible performance benefit?
def hf_lm_pipe(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
) -> callable:
    def generate(
        prompt: str, max_len: int, perf: bool = False
    ) -> tp.Tuple[str, tp.List[torch.tensor]]:
        out = model.generate(
            tokenizer(prompt, return_tensors="pt")["input_ids"],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_len,
            return_dict_in_generate=perf,
            output_scores=perf,
        )
        # print(out.sequences)
        if perf:
            return tokenizer.decode(out.sequences[0]), out.scores
        else:
            return tokenizer.decode(out[0])

    return generate


def nanoGPT_pipe(tokenizer: any, model: any) -> callable:
    model = torch.compile(model)
    model.eval()

    def generate(prompt: str, max_len: int) -> str:
        ipt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)[None, ...]
        out = model.generate(ipt_tokens, max_len)
        return tokenizer.decode(out[0].tolist())

    return generate


def onnx_custom_pipe(
    tokenizer: PreTrainedTokenizer,
    model_path: str,
    n_attention_heads: int,
    hidden_size: int,
    n_layers: int,
) -> callable:
    """Initialize a custom greedy sampling generation pipeline for a onnx compiled model.

    Args:
        tokenizer: The tokenizer to use.
        model_path: The path to the .onnx model.
        n_attention_heads: The number of heads in the model.
        hidden_size: The size of each hidden layer in the model.
        n_layers: The total number of model layers.

    Returns:
        callable: A `generate` function for model generative inference.
    """
    # Initialize runtime session.
    ort_session = onnxruntime.InferenceSession(model_path)
    # For all tests we are committing to single batch inference.
    batch_size = 1

    def init_input(
        prompt: str,
    ) -> tp.Tuple[torch.tensor, torch.tensor, torch.tensor, list]:
        """Tokenize prompt and initialize empty past state for initial model input.

        NB: We elect not to use any early stopping logic here to ensure a fair speed comparison.

        Args:
            prompt: Prompt to tokenize.

        Returns:
            input_ids,
        """
        encodings_dict = tokenizer(prompt)

        input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int64)
        attention_mask = torch.tensor(
            encodings_dict["attention_mask"], dtype=torch.int64
        )
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(position_ids < 0, 0)
        position_ids = position_ids.to(torch.int64)

        # Initialize empty past state, since we have not done any decoding yet.
        empty_past = []
        past_shape = [
            2,
            batch_size,
            n_attention_heads,
            0,
            hidden_size // n_attention_heads,
        ]
        for i in range(n_layers):
            empty_past.append(torch.empty(past_shape).type(torch.float32).to("cpu"))

        return input_ids, attention_mask, position_ids, empty_past

    def generate(
        prompt: str, max_len: int, perf: bool = False
    ) -> tp.Tuple[str, tp.List[torch.tensor]]:
        """MAYBE THIS SHOULD BE AN ABSTRACT FUNCTION?

        Args:
            prompt (str): _description_
            max_len (int): _description_
            perf: Do we want to output logit information so that we can analyse LM performance?

        Returns:
            str: _description_
        """
        logits = []

        # Compiled model expects a batched input, so we nest it before tokenizing.
        input_ids, attention_mask, position_ids, past = init_input(prompt=[prompt])

        all_token_ids = input_ids.clone()

        for _ in range(max_len):
            # Prepare inputs just how onnx runtime likes them.
            ort_inputs = {
                "input_ids": np.ascontiguousarray(input_ids.cpu().numpy()),
                "attention_mask": np.ascontiguousarray(attention_mask.cpu().numpy()),
                "position_ids": np.ascontiguousarray(position_ids.cpu().numpy()),
            }
            for i, past_i in enumerate(past):
                ort_inputs[f"past_{i}"] = np.ascontiguousarray(past_i.cpu().numpy())

            # Run onnx runtime model inference.
            outputs = ort_session.run(None, ort_inputs)

            # Extract token logits from output and use fast greedy sampling to get next token.
            next_token_logits = torch.from_numpy(outputs[0][:, -1, :])
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            if perf:
                logits.append(next_token_logits)
            all_token_ids = torch.cat(
                [all_token_ids, next_tokens.unsqueeze(-1)], dim=-1
            )

            # Update input_ids, attention_mask, position_ids and past given new states and output.
            input_ids = next_tokens.clone().detach().reshape([batch_size, 1]).to("cpu")
            position_ids = (position_ids[:, -1] + 1).reshape(batch_size, 1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones([batch_size, 1]).type_as(attention_mask)], 1
            ).to("cpu")

            # Update past with decoder layer states so that they do not need to be recomputed.
            past = []
            for i in range(n_layers):
                past_i = torch.from_numpy(outputs[i + 1])

                past.append(past_i.to("cpu"))

        return tokenizer.decode(all_token_ids[0], skip_special_tokens=True), logits

    return generate




            