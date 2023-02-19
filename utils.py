import abc
from collections import defaultdict
import dataclasses
import time
import typing as tp
import random

from matplotlib import pyplot as plt
import numpy as np
from transformers import PreTrainedTokenizer
import torch
from tqdm import notebook


@dataclasses.dataclass
class ModelTester:
    """A wrapper class to test the performance of a specific inference function and store the results.

    Attributes:
        model_name: A identifier given the model for use in the display of results.
        model_func: The inference function to be tested. Documented below.
        times: An array of times (in seconds) corresponding to a speed tests following `run_speed_test`.
        outputs: An array of generations corresponding to a set of speed tests following `run_speed_test`.
        perplexities: An array of perplexity values corresponding to a set of perplexity tests following `run_perplexity_test`.
    """

    model_name: str
    model_func: callable
    times: tp.List[float] = dataclasses.field(default_factory=list)
    outputs: tp.List[str] = dataclasses.field(default_factory=list)
    perplexities: tp.List[float] = dataclasses.field(default_factory=list)

    @abc.abstractstaticmethod
    def model_func(
        prompt: str, generation_length: int, return_logits: bool = False
    ) -> tp.Tuple[str, tp.List[torch.tensor]]:
        """The inference method to be overridden.

        Args:
            prompt: The context to encode prior to inference.
            generation_length: The number of tokens for the model to generate.
            return_logits: A flag indicating whether or not the model should return logits for analysis.

        Returns:
            A tuple containing the outputted string and the logits (if requested, otherwise just an empty list).
        """
        pass

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
            # Record information in bits.
            log_prob_sum += np.log2(torch.softmax(logits[0], dim=-1)[:, target])
        return np.exp(-1 * (1 / len(targets)) * log_prob_sum).item()

    @staticmethod
    def generate_random_input(length: int, tokenizer: PreTrainedTokenizer) -> str:
        """Generate a random input for prompting during the speed test.

        This seems to be a common procedure in other benchmarking code.

        Args:
            length: The length of the input *in tokens*.
            tokenizer: The tokenizer that will be used to encode the input prior to inference."""

        return tokenizer.decode(
            [random.randint(0, tokenizer.vocab_size - 1) for _ in range(length)]
        )

    def run_perplexity_test(
        self, test_sequences: tp.List[str], tokenizer: PreTrainedTokenizer
    ) -> None:
        """Analyse and record the perplexity for each sequence in `test_sequences`.

        Appends perplexities to `self.perplexities` and *does not* return them.

        ! Slightly inefficient in terms of having to tokenize and re-decode due to tech debt.

        Args:
            test_sequences: A set of sequences to analyse. Ideally taken from an established database and of reasonable length.
            tokenizer: The tokenize that the model will use. Used to feed the input token-wise.
        """
        for sequence in notebook.tqdm(test_sequences):
            tokenized_sequence = tokenizer(sequence)["input_ids"]
            seq_logits = []
            # Predict each token after the first.
            targets = tokenized_sequence[1:]
            for i in range(1, len(tokenized_sequence)):
                # Get the first `i` tokens in string format.
                ipt = tokenizer.decode(tokenized_sequence[:i])
                # Run inference and collect logits.
                _, logits = self.model_func(
                    ipt, generation_length=1, return_logits=True
                )
                seq_logits.append(logits)
            # Calculate perplexity for each sequence seperately.
            self.perplexities.append(self.perplexity(targets, seq_logits))

    def run_speed_test(
        self,
        n_tests: int,
        prompt_length: int,
        max_generation_length: int,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        """Analyse and record the inference time for a set of random prompts.

        Appends times to `self.perplexities` and *does not* return them.

        Args:
            test_sequences: A set of sequences to analyse. Ideally taken from an established database and of reasonable length.
            tokenizer: The tokenize that the model will use. Used to feed the input token-wise.
        """
        for _ in notebook.tqdm(range(n_tests)):
            # Generate a random prompt for each test.
            prompt = self.generate_random_input(
                length=prompt_length, tokenizer=tokenizer
            )
            # Record time in seconds of the full text generation procedure and store.
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
    """Take `n_articles` random samples from the `train` section of a dataset and return them in a list.

    Ensure that each article is greater than `min_length_chars` and crop them to be <= `max_len_chars`.

    Args:
        dataset: A datasets.Dataset to sample from.
        n_articles: The number of articles to return.
        min_length_chars: The minimum number of characters an article must have to be returned.
        max_length_chars: The maximum number of characters an article in the list will be. If the article is larger, it will be clipped.

    Returns:
        A list of articles.
    """
    article_indexes = list(range(len(dataset["train"])))
    articles = []
    for _ in range(n_articles):
        # Ensure there are still articles to sample from.
        while True and article_indexes:
            # Select a random article.
            i = article_indexes.pop(random.randint(0, len(article_indexes) - 1))
            # If it is longer than `min_length_chars`, clip and add it, break for next article.
            if len(dataset["train"][i]["text"]) > min_length_chars:
                articles.append(dataset["train"][i]["text"][:max_length_chars])
                break

    if len(articles) != n_articles:
        print(
            f"WARNING: Only found {len(articles)} articles greater than {min_length_chars} characters."
        )

    return articles


def standard_error_mean(results: tp.List[float]) -> float:
    """Returns the SEM for an array of results."""
    return np.std(results) / np.sqrt(len(results))


def plot_test_results(model_testers: tp.List[ModelTester], result_type: str) -> None:
    """Generates a bar graph with SEM error for the desired result attribute in a given `ModelTester`.

    Args:
        model_testers: A list of `ModelTester's` to include in the graph.
        result_type: The `ModelTester` attribute to take the results from.
    """
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
        errors.append(standard_error_mean(results))

    # Plot and label
    ax.bar(x_pos, means, yerr=errors, align="center")
    ax.set_xticks(
        x_pos, labels=[model_tester.model_name for model_tester in model_testers]
    )
    ax.set_ylabel(f"Inference {result_type}")
    ax.set_title(f"{result_type} comparison of GPT2 inference implementations.")

    plt.show()
    fig.savefig(f"results/{result_type}_comparison_figure.png")


def plot_varied_speed_test(
    tester_results: tp.Dict[str, tp.List[tp.List[float]]], generation_lengths: list
) -> None:
    """Plot results from inference time testing over input and generation length.

    Args:
        tester_results: Dictionary containing results from the test:
        {   # Key for each inference function - prompt length pair.
            <test label> :
            [ # Array for each different generation length.
                [# Float for each test repetition within generation length.
                    0.1,
                    ...
                ],
                ...
            ],
            ...
        }
        generation_lengths: List of generation lengths to label the x axis.
    """
    fig, ax = plt.subplots()

    results_means, results_errors = {}, {}

    # Calculate the mean and SEM for each set of tests for a specific func-prompt_length-generation_length set.
    results_per_test = 0
    for label, results in tester_results.items():
        results_means[label] = [np.mean(result) for result in results]
        results_errors[label] = [standard_error_mean(result) for result in results]

        if not results_per_test:
            results_per_test = len(results_means[label])

    # Plot as a line for each func-prompt_length set.
    x = np.arange(results_per_test)
    for name in tester_results.keys():

        ax.errorbar(x, results_means[name], yerr=results_errors[name], label=name)

    # Do some house keeping.
    ax.legend(loc="upper left")
    ax.set_ylabel(f"Inference times (s)")
    ax.set_xticks(x, labels=generation_lengths)
    ax.set_xlabel("Generation Length (tokens)")
    ax.set_title(
        "Inference times for model and prompt length as a function of generated tokens"
    )
    plt.show()
    fig.savefig(f"results/varied_speed_comparison_figure.png")
