# UniAudio 1.5
This Repository provides an LLM-driven audio codec model, which can be used to build multi-modal LLMs (text and audio modalities).
More details will be introduced as soon as.
You can find the paper from https://arxiv.org/pdf/2406.10056

## Introduction
The Large Language models (LLMs) have demonstrated supreme capabilities in text understanding and generation, but cannot be directly applied to cross-modal tasks without fine-tuning. This paper proposes a cross-modal in-context learning approach, empowering the frozen LLMs to achieve multiple audio tasks in a few-shot style without any parameter update. Specifically, we propose a novel and LLMs-driven audio codec model, LLM-Codec, to transfer the audio modality into the textual space, \textit{i.e.} representing audio tokens with words or sub-words in the vocabulary of LLMs, while keeping high audio reconstruction quality. The key idea is to reduce the modality heterogeneity between text and audio by compressing the audio modality into a well-trained LLMs token space. Thus, the audio representation can be viewed as a new \textit{foreign language}, and LLMs can learn the new \textit{foreign language} with several demonstrations. In experiments, we investigate the performance of the proposed approach across multiple audio understanding and generation tasks, \textit{e.g.} speech emotion classification, audio classification, text-to-speech generation, speech enhancement, etc. The experimental results demonstrate that the LLMs equipped with the proposed LLM-Codec, named as UniAudio 1.5, prompted by only a few examples, can achieve the expected functions in simple scenarios. It validates the feasibility and effectiveness of the proposed cross-modal in-context learning approach. To facilitate research on few-shot audio task learning and multi-modal LLMs, we have open-sourced the LLM-Codec model. 


## How to use LLM-Codec?
step 1:
```
download the checkpoint (wget https://huggingface.co/Dongchao/UniAudio/resolve/main/llm3_codec_uni.pth)
```
Step 2: Download LLAMA 2 7B based on https://github.com/meta-llama/llama-recipes/tree/main <br>
Step 3: refer to infer.py
```
python infer.py
```

## How to use LLM-Code and LLAMA 2 (UniAudio 1.5)
In the following, we give a simple demonstration to use it.
```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port=10645 infer_code/eval_accent_understanding_v2.py \
            --batch_size 1 \
            --max_seq_len 2048 \
            --num_workers 0 \
            --output_type "next_token_prediction" \
            --audio_path "the path of audio folder" \
            --file_path tsv/acc_9way_1_shot.scp \
            --vq_config_path config.yaml \
            --output_dir log_eval_few_shot/7B_output \
            --llama_model_path llama_inference/llama-2-7b \
            --induction 1 \
            --codec_ckpt "llm-codec.pth" \

```

## Demos
Please refer to demos folder to listen the generated audio.


### Acknowledgements
https://github.com/descriptinc/descript-audio-codec 
https://github.com/yangdongchao/AcademiCodec
https://github.com/hubertsiuzdak/snac
https://github.com/Meta-Llama/llama-recipes

