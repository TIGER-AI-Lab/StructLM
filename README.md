# StructLM

This is the repository for our COLM-2024 paper [StructLM: Towards Building Generalist Models for Structured Knowledge Grounding](https://arxiv.org/abs/2402.16671).

You can use this repository to evaluate the models. To reproduce the models, use [SKGInstruct](https://huggingface.co/datasets/TIGER-Lab/SKGInstruct) in your preferred finetuning framework. The checkpoitns are being released on [Huggingface](https://huggingface.co/collections/TIGER-Lab/structlm-65dcab5a183c499cc365fafc).

The processed test data is already provided, but the prompts used for training and testing can be found in `/prompts`

## Table of Contents
  * [Links](#links)
  * [Install Requirements](#install-requirements)
  * [Download files](#download-files)
  * [Run evaluation](#run-evaluation)
  * [Acknowledgements](#acknowledgements)
  * [Cite](#cite)

## Links
- Arxiv Link: https://arxiv.org/abs/2402.16671
- Website: https://tiger-ai-lab.github.io/StructLM/

## Training

Easy reproduction can be done with the [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory).

1. Follow the [data preparation steps](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md) on their repo to add one of the StructLM datasets from huggingface
2. use the parameters in the bash script `StructLM_finetune.yaml`, as a reference replacing the parametres in block quotes [] with your paths. Then start the training like
   `llamafactory-cli train StructLM_finetuning.yaml`, as [such](https://github.com/hiyouga/LLaMA-Factory/tree/b2fc7aeb03fbb40e9beb27e9958c958ee48e23cf?tab=readme-ov-file#quickstart)

## Evaluate StructLM-7B

### Install Requirements

Requirements:
- Python 3.10
- Linux
- support for CUDA 11.8

`pip install -r requirements.txt`

### Download files

`./download.sh`

this will download
1. StructLM-7B-Mistral
2. The raw data required for executing evaluation
3. The processed test data splits ready for evaluation

### Run evaluation

#### For StructLM-7B/13B/34B
You can download these models seperately with
```
huggingface-cli download --repo-type=model --local-dir=models/ckpts/StructLM-7B TIGER-Lab/StructLM-7B
huggingface-cli download --repo-type=model --local-dir=models/ckpts/StructLM-13B TIGER-Lab/StructLM-13B
huggingface-cli download --repo-type=model --local-dir=models/ckpts/StructLM-34B TIGER-Lab/StructLM-34B
```

Then, you can run the inference on the downloaded checkpoints.
```
./run_test_eval.sh StructLM-7B
./run_test_eval.sh StructLM-13B
./run_test_eval.sh StructLM-34B
```

#### For StructLM-7B-Mistral
You can download these models seperately with
```
huggingface-cli download --repo-type=model --local-dir=models/ckpts/StructLM-7B-Mistral TIGER-Lab/StructLM-7B-Mistral
```

We can run the inference on the donwloaded checkpoint.
```
python mistral-fix-data.py
./run_test_eval.sh StructLM-7B-Mistral
```

These evaluation will generate the results in `outputs/StructLM-*/`

## Acknowledgements

The evaluation metrics in this repository were adapted and modified from the evaluation files found in https://github.com/HKUNLP/UnifiedSKG

## Cite
```
@misc{zhuang2024structlm,
    title={StructLM: Towards Building Generalist Models for Structured Knowledge Grounding},
    author={Alex Zhuang and Ge Zhang and Tianyu Zheng and Xinrun Du and Junjie Wang and Weiming Ren and Stephen W. Huang and Jie Fu and Xiang Yue and Wenhu Chen},
    year={2024},
    eprint={2402.16671},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
