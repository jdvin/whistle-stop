import typing as tp

import onnxruntime
import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


# TODO: Does preloading the models into the function have a tangible performance benefit?
def huggingface_pipe(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
) -> callable:
    def generate(
        prompt: str, generation_length: int, return_logits: bool = False
    ) -> tp.Tuple[str, tp.List[torch.tensor]]:
        out = model.generate(
            tokenizer(prompt, return_tensors="pt")["input_ids"],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=generation_length,
            return_dict_in_generate=return_logits,
            output_scores=return_logits,
        )

        if return_logits:
            return tokenizer.decode(out.sequences[0]), out.scores
        else:
            return tokenizer.decode(out[0]), []

    return generate


# def nanoGPT_pipe(tokenizer: any, model: any) -> callable:
#     model = torch.compile(model)
#     model.eval()

#     def generate(prompt: str, max_len: int) -> str:
#         ipt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)[None, ...]
#         out = model.generate(ipt_tokens, max_len)
#         return tokenizer.decode(out[0].tolist())

#     return generate


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
        prompt: str, generation_length: int, return_logits: bool = False
    ) -> tp.Tuple[str, tp.List[torch.tensor]]:
        """Custom state passing generation pipeline adapted from: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/Inference_GPT2_with_OnnxRuntime_on_CPU.ipynb."""

        logits = []

        # Compiled model expects a batched input, so we nest it before tokenizing.
        input_ids, attention_mask, position_ids, past = init_input(prompt=[prompt])

        all_token_ids = input_ids.clone()

        for _ in range(generation_length):
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
            if return_logits:
                logits.append(next_token_logits)
            # Insert next token into all tokens.
            all_token_ids = torch.cat(
                [all_token_ids, next_tokens.unsqueeze(-1)], dim=-1
            )

            # Set generated token to be inputted into the model.
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




            