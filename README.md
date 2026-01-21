# VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG

![Framework](/Figure/framework.png)

Abstract: *Scaling multimodal large language models (MLLMs) to long videos is constrained by limited context windows. While retrieval-augmented generation (RAG) is a promising remedy by organizing query-relevant visual evidence into a compact context, most existing methods (i) flatten videos into independent segments, breaking their inherent spatio-temporal structure, and (ii) depend on explicit semantic matching, which can miss cues that are implicitly relevant to the query’s intent. To overcome these limitations, we propose VideoStir, a structured and intent-aware long-video RAG framework. It firstly structures a video as a spatio-temporal graph at clip level, and then performs multi-hop retrieval to aggregate evidence across distant yet contextually related events. Furthermore, it introduces an MLLM-backed intent-relevance scorer that retrieves frames based on their alignment with the query’s reasoning intent. To support this capability, we curate IR-600K, a large-scale dataset tailored for learning frame–query intent alignment. Experiments show that VideoStir outperforms state-of-the-art baselines in most cases, highlighting the promise of shifting long-video RAG from flattened semantic matching to structured, intent-aware reasoning.*


# Getting Started
We recommend installing 64-bit Python 3.12.12 and PyTorch 2.9.0. On a CUDA GPU machine, the following will do the trick:

**Clone the repository:**
```
git clone https://github.com/RomGai/VideoStir
cd VideoStir
```

# Inference

```
cd inference
```

**Install the required dependencies:**

```
pip install -r requirements_inference.txt
```

**Download checkpoint:**

Download the pretrained weights for the Intent-Relevance Scorer via [this link](https://drive.google.com/file/d/1CUC1i7zstZktWDp30Pts4s8E7QULDgns/view?usp=drive_link). Put it in the current directory and extract the checkpoint.

```
unzip checkpoint.zip
```

**Download sample data:**

Download the sample data (for demonstrating the file structure and input format) from [this link](https://drive.google.com/file/d/1Jqhtt8dClorfYuPoeTXQd_GSt-iYPHlc/view?usp=drive_link). The subtitle files are optional.

```
unzip sample_data.zip
```

**Run the RAG pipeline:**

```
python pipeline.py
```

**Perform downstream reasoning:**

Take LLaVA-Video-7B and LongVideoBench as an example:

1. Install dependencies for LLaVA-Video:

```
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
pip install flash-attn --no-build-isolation
```

2. Run downstream reasoning:

```
python run_llava_video_samples.py
python evaluate_llava.py
```

Tips: You may encounter an error when using the installed llava package as-is. To fix this, edit "llava/model/\_\_init\_\_.py" and replace

```
```

with

```
from .model.language_model.llava_llama import LlavaLlamaForCausalLM
```

Then, edit "llava/model/multimodal_resampler/qformer.py" and rplace

```
```

with

```
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from transformers.modeling_utils import (
    PreTrainedModel,
```



# Train

```
cd train
```

**Install the required dependencies:**

```
pip install -r requirements_train.txt
```

**Prepare IR-600K dataset:**

1. In the “Missing Videos” section on the [official ActivityNet page](http://activity-net.org/download.html), submit the application form to request access to the dataset. Approval may take some time. Once you receive the dataset’s Google Drive link, download "\[Update\]_Anet_videos_15fps_short256.zip". Put it in the current directory. Then extract the annotation files and sample target frames from the ActivityNet dataset.
   
```
unzip -o labels.zip
python Anet_data_sampling.py
```

2. Download the remaining parts of the IR-600K via [this link](https://drive.google.com/file/d/1D15nd3dzqiTiP6ufTxCOY9lxdQmxk8dK/view?usp=sharing). Put it in the current directory and extract the remaining files.

```
unzip -n IR-600K.zip
```

**Train the model**

On multiple GPUs:

```
accelerate launch --config_file muti_gpu.yaml train.py --config mmkd_black_box_multi.json
```

On a single GPU:

```
python train_lora.py --config mmkd_black_box_lora_single.json
```

