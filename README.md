# VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG

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

**Extract sample data:**

```
unzip sample_data.zip
```

**Run the RAG pipeline:**

```
python pipeline.py
```

**Perform downstream reasoning:**

Take LLaVA-Video-7B and LongVideoBench as an example:

```
python run_llava_video_samples.py
python evaluate_llava.py
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

1. In the “Missing Videos” section on the [official ActivityNet page](http://activity-net.org/download.html), submit the application form to request access to the dataset. Approval may take some time. Once you receive the dataset’s Google Drive link, download "\[Update\]_Anet_videos_15fps_short256.zip". Put it in the current directory.

2. Extract the annotation files and then sample target frames from the ActivityNet dataset.
   
```
unzip -o labels.zip
python Anet_data_sampling.py
```

(Optional) Delete the raw files to save disk space.

```
rm -rf \[Update\]_Anet_videos_15fps_short256.zip
```

3. Download the remaining parts of the IR-600K via [this link](https://drive.google.com/file/d/1D15nd3dzqiTiP6ufTxCOY9lxdQmxk8dK/view?usp=sharing). Put it in the current directory and extract the remaining files.

```
unzip -n IR-600K.zip
```

(Optional) Delete the raw files to save disk space.

```
rm -rf IR-600K.zip
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

