# T-CotMechanism

This repository contains code for the paper： 

Unveiling the Mechanisms of Explicit CoT Training: How CoT Enhances Reasoning Generalization

## 🎯 Introduction

The integration of explicit Chain-of-Thought (CoT) reasoning into training large language models (LLMs) has advanced their reasoning capabilities, yet the mechanisms by which CoT enhances generalization remain poorly understood:

- **(Q1)** how the integration of step-by-step rationales during explicit CoT training reshapes internal model representations?

- **(Q2)** how these structural changes contribute to improved reasoning generalization across both in-distribution (ID) and out-of-distribution (OOD) scenarios?

## 🎉key Insights

In this paper, we investigate zero-shot CoT data directly and formalize how step-wise reasoning emerges to support both ID and OOD generalization in transformer-based CoT optimization. The analysis is carried out through a **comparison study** of training paradigms with and without CoT. (See main text for more details!)

- **Structural Advantage** (for Q1):

CoT training internalizes reasoning into a two-stage generalizing circuit, where the number of stages corresponds to the explicit reasoning steps during training. *Notably, CoT-trained models resolve intermediate results at shallower layers compared to non-CoT counterparts, freeing up deeper layers to specialize in subsequent reasoning steps.*

<img src="Pictures\circuit_change.jpg" alt="circuit_change" style="zoom:87%;" />

- **Theoretical Analysis** (for Q2):

the information-theoretic generalization bounds via distributional divergence can be decomposed into ID and OOD components. While ID error diminishes with sufficient training regardless of CoT, OOD error critically depends on CoT: Non-CoT training fails to generalize to OOD samples due to unseen reasoning patterns, whereas CoT training achieves near-perfect OOD generalization by mastering subtasks and reasoning compositions during training.

<img src="Pictures\theorem1.jpg" alt="theorem1" style="zoom:80%;" />

<img src="Pictures\theorem2.jpg" alt="theorem2" style="zoom:80%;" />


## ⚙️ Controlled Experiments

1. To install the experiment, please install the pip file.

```setup
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

cd transformers
pip install -e .
cd ..

cd simpletransformers
pip install -e .
cd ..
```

2. You can run the following example  to generate data.

```data
TCotMechanism/ControlledExperiments/composition.ipynb
```
>📋 You can adjust the following hyperparameters.
>
>NUM_ENTITY_IN = 2000  #  $|\mathcal{E}|$
>
>NUM_RELATION = 200  #  $|\mathcal{R}|$
>
>OOD_ratio = 0.05  # $|S_{\text{OOD}}|: |S_{\text{ID}}|=5\%:95\%$. 
>
>lambda_noise = 0.4 $ noise ratio
>
>phi=7.2  # $|S_{\text{ID}_{train}}^{(2)}|/|S_{\text{ID}}|$

3. Training.

```train example
chmod +x TCotMechanism/ControlledExperiments/run.sh  #ensure execute permissions
TCotMechanism/ControlledExperiments/run.sh
```

4. Evaluation.

```test example
TCotMechanism/ControlledExperiments/eval_qa.py --dir <path_to_saved_checkpoints>
```

5.  Two-stage Generalizing Circuit.

```test example
TCotMechanism/ControlledExperiments/tracing_composition.py  #check the first stage
TCotMechanism/ControlledExperiments/tracing_composition1.py  #check the second stage
```

## 🚀 Realistic Data Verification

1. Datasets we used are publicly available at [Dataset](https://huggingface.co/datasets/fxmeng/pissa-dataset).

```bash
/TCotMechanism/RealisticDataVerification/dataset/metamath
```

- Quick start

2. Install via pip:

```
conda install nvidia/label/cuda-12.1.0::cuda-toolkit
conda install pytorch==2.4.0 torchvision=0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r /TCotMechanism/RealisticDataVerification/requirements.txt
pip install flash-attn --no-build-isolation
```

3. Training.

```
/TCotMechanism/RealisticDataVerification/run_lora.sh
```

4. Testing

```
python utils/gen_vllm.py --model $OUTPUT_PATH --sub_task metamath --output_file $...$
python utils/test_acc.py --input_file $...$
```

5. (Optional): introduce noise.

```
TCotMechanism/RealisticDataVerification/data_process.py
```

## 😊Other Results

CoT training accelerates convergence and enhances generalization from ID to both ID and OOD scenarios while maintaining robust performance even with tolerable noise $\xi$. These findings are further validated on complex real-world datasets.

<img src="Pictures\noise_only_t.jpg" alt="noise_only_t" style="zoom:50%;" />

<img src="Pictures\real1.jpg" alt="real1" style="zoom:60%;" />
