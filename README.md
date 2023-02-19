# whistle-stop

Experimenting with fast GPT inference.

# Notes.

The goal here was to create a function that runs generation on GPT2 as fast as possible. 

After building an automated testing framework, the Huggingface Transformers GPT2 model was used as a reference. Next, nanoGPT was assessed. In contrast to my assumptions, nanoGPT ran far slower than HF. One obvious optimisation was that nanoGPT is using stochastic token sampling, as opposed to greedy sampling like the HF pipeline. But this was definitely not the whole story as this problem appears to go all the way down to single forward passes of both the models; nanoGPT was taking twice as long! After checking and double checking the obvious things (e.g., maybe HF was using GPU [it wasn't], maybe the inputs were different [they weren't], maybe the configs were different [they didn't seem to be]), it was decided to move on to _actually_ completing the task, potentially circling back for this peculiarity later.

The rest of my time was spent digging into the onnx toolkit and extending my testing framework as necessary. In hindsight, more time could've been spent on the former task, and less on the latter, but it is fun to build tools. Where I landed for the weekends work was the following:

A HF GPT2 model compiled with past state passing through onnx with post compilation optimisations and quantisation led to the best performance by time. The different models I tried and the resulting inference time for a 50 token prompt and 100 generated tokens are displayed in the following graph*:

![image](https://user-images.githubusercontent.com/34499901/219914822-b4eda896-3cd5-4a5c-8769-33010bbf5cba.png)

Given the static generation parameters, the fastest function displayed a >4x speed up over the standard HF model.
To get a rough estimate of relative model performance, fixed-length sequence perplexity was also calculated for each model:

![image](https://user-images.githubusercontent.com/34499901/219915172-d93abd13-49a6-48e4-a2c4-505959a1ad52.png)

Thus this result came at a substantial hit to generation performance, due to quantisation. If the goal were to balance speed and performance, the onnx base compiled model with state passing presents an excellent compromise. This may be further improved through less aggressive quantisation. Interestingly the post compilation optimisations did not seem to decrease speed unless quantisation was also used.

As the goal here was specifically to be _as fast as possible_, additional analyses were conducted on the fasted model. The following graph displays the impact of prompt length and generation length on model performance:

![image](https://user-images.githubusercontent.com/34499901/219918131-f4a01a82-f474-4cf9-a7d9-12fecc6ee216.png)

The generation speed of the onnx model is far more robust to increases in prompt length and generation length than the base HF model. Therefore, the speed gains resulting from the onnx model compound as generation length increases. Additionally, the positive interaction between prompt length and generation length in the HF model (i.e., that differences in generation time between prompt lengths are larger for larger generation lengths) appears attenuated in the onnx model. One curious observation is that the generation speed is comparable between the HF and onnx models when doing inference on a single token. One potential explanation for this is that the initial work done to set up the data structures required for inference and state passing in the onnx model cancels out the gains due to faster inference. Definitely something to look into in future.

\* NB: For all graphs, error bars refer to standard error of the mean.

# Future Questions.

- Why _exactly_ is single token inference so slow for the onnx models?
- How would less aggressive quantisation impact speed, and what would be gained in terms of performance?
- What was making nanoGPT so slow?
- How much work is state passing doing here?
- Can we do better?

# To Run.

After activating your virtual environment...

```
# Install requirements
pip instal -r requirements.txt

# Compile the base past state onnx model.
python -m onnxruntime.transformers.models.gpt2.convert_to_onnx -m gpt2 --output gpt2_past.onnx -p fp32
```

You should then just be able to run the notebook from the beginning.

# Log [WARNING: stream of consciousness].

    - project started
    - creating basic automated testing framework
        - potentially add accuracy evaluation if there is time
    - start with baseline hf implementation
        - not using default generation pipe to suppress annoying warning message
    - testing framework is operational
    - Aiden strongly hinted to look at Karpathy's nanoGPT, so thats next
        - will try base nanoGPT first, then probably try:
            1. Compiling
            2. Quantisation
            3. Both
            4. ???
    - an initial test shows nanoGPT taking 2.5x times as long as HF?!?!
        - definitely nothing to do with the tokenizer
        - compiling doesnt seem to help
        - going to need to dig into the inference logic of both implementations (or perhaps configs)
            - if nothing obvious comes up then probably go straight to quantising and maybe compiling the hf model
        - Comparing across generation lengths:
                    hf base                 nanoGPT
            1:    [0.034813547134399415, 0.0639218807220459]
            10:   [0.21567120552062988,  0.60211181640625]
            100:  [2.1581297397613524,   6.610726928710937]
            1000: [26.426644372940064,   291.933207321167]
        - large interaction between model and tokens generated for inference time
            - the more tokens to be generated, the larger the discrepancy between models
                - something is compounding in inference
                    - should try to just run straight forward (not including tokenization and sampling) on both models to compare)
                        - if nanoGPT is still slower than cut losses and try to make HF faster
                            - after making sure that it is not a config/hardware thing (make sure hf isnt running mps, make sure nothing is moving around devices)
                        - single inference time for
                                nanoGPT             HFGPT
                            0.05437493324279785 0.028493165969848633
                            - this result is highly repeatable, and it is not obvious why it is occurring
                                - both models are operating on cpu, with the same input, and seemingly the same config.
                                - for now I think the best thing to do is to turn my attention to making hf faster,  then maybe coming back and having another pass as nano
                        - one of the problems seems to be that nanoGPT is using nucleus sampling whilst hf seems to be using argmax
                            - but general inference is still slower so...
    - going to try onnx first: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/Inference_GPT2_with_OnnxRuntime_on_CPU.ipynb
        - `python -m onnxruntime.transformers.models.gpt2.convert_to_onnx -m models/huggingface_GPT2.pt --output huggingface_gpt2.onnx -p fp32`
        - model exported but failed to optimise
            - exported onnx model is still 2x faster than base hf for just model.forward() :D
                - this is even before quantisation!!
            - faster without IOBinding, presumably because we are doing CPU inference?
        - model sampling appears faster too
            - evaluating all models on greedy search because we are optimising for speed and reproducibility for the sake of testing
        - confirmed onnx custom pipeline beats standard huggingface by >2x
            - what next?
                - quantisation
                    - but first, want to see if there is some kind of system to be built that will assess performance of models so that the tradeoff between full precision and quantisation is explicit
                - quantisation improves performance

            - plot performance as a function of generation length and context size
            - potentially variety in contexts aswell?
            - cleaning up obviously
                - line everything up so it runs in one go (all model compiling included)
                - document and check
                - write tests?
                - requirements file
                - something may be wrong with the absolute perplexity calculation, but relatively they are where i expect them to be
            - swapping in fast tokenizers?!
                - apparently gains in fast tokenizer are only realised when you are tokenising large chunks of text in parallel, so probably not worth it.
     - single token inference appears to be slower for onnx models?
        - something to look into...
