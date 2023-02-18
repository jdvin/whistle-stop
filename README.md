# whistle-stop

Experimenting with fast GPT inference.

# Log.

    - project started
    - creating basic automated testing framework
        - potentially add accuracy evaluation if there is time
    - start with baseline hf implementation
        - not using default generation pipe to suppress annoying warning message
    - testing framework is operational
    - Aiden strongly hinted to look at karpathys nanoGPT, so thats next
        - will try base nanoGPT first, then probably try:
            1. Compiling
            2. Quantisation
            3. Both
            4. ???
            5. Profit
    - need pytorch > 2.0 for flash attention
        - will try first without and then see if it makes an impact
    - an initial test shows nanoGPT taking 2.5x times as long as HF?!?!
        - definitely nothing to do with the tokenizer
        - compiling doesnt seem to help
        - going to need to dig into the inference logic of both implementations (or perhaps configs)
            - if nothing obvious comes up then probably go straight to quantising and maybe compiling the hf model (¯\_(ツ)_/¯)
        - Comparing across generation lengths:
                    hf base                 nanoGPT
            1:    [0.034813547134399415, 0.0639218807220459]
            10:   [0.21567120552062988,  0.60211181640625]
            100:  [2.1581297397613524,   6.610726928710937]
            1000: [26.426644372940064,   291.933207321167]
        - large interation between model and tokens generated for inference time
            - the more tokens to be genereated, the larger the discrepancy between models
                - something is compounding in inference
                    - should try to just run straight forward (not including tokenization and sampling) on both models to compare)
                        - if nanoGPT is still slower than cut losses and try to make HF faster
                            - after making sure that it is not a config/hardware thing (make sure hf isnt running mps, make sure nothing is moving around devices)
                        - single inference time for
                                nanoGPT             HFGPT
                            0.05437493324279785 0.028493165969848633
                            - this result is highly repeatable, and it is not obvious why it is occuring
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
            - evaluating all models on greedy search because we are optimising for speed and reproducibily for the sake of testing
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
            - swapping in fast tokenizers?! >:)
