[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/IBJk4qkU)
# 🛠️ PACE 2025 – Final Model Containerization Submission  

📅 **Final Submission Deadline: 24th August 2025 (23:59 GMT)**  

This repository contains the **final submission guidelines** for the PACE 2025 Challenge.  
Participants are required to submit their trained models as a **Docker container** to ensure reproducibility.  

---
## 📜 Submission Policy  

Each team is allowed **only one submission**. Please ensure you submit your **best model**.  

Using GitHub for submission may prevent you from uploading large files. If you encounter this issue, please follow the official Git LFS guide:  
👉 [Managing Large Files with Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)  

If you still face challenges uploading your model, contact the organizers through the **official communication channel** **before the specified deadline**, or you will be disqualified. An alternative submission method will be provided.

You are allowed to include a **README.md** file explaining the flow of your code and model.

---

## PACE2025 Docker Guide

This guide explains how to build and run the inference for the **ultrasound multi-task segmentation and classification model** in a Docker environment with GPU/cpu support.

---

## 📌 Project Overview

This Docker container runs a multi-task deep learning model for ultrasound image analysis that can perform:

- **Segmentation** – Generates binary masks for ultrasound images  
- **Classification** – Classifies ultrasound images into predefined categories  
---

## 📂 Project Structure
```
├── Dockerfile
├── requirements.txt
├── main.py                 # Main inference script
├── model.py               # Model architecture
├── checkpoints/           # Model weights directory
│   └── best_model.pth    # Trained model checkpoint
└── tools/                # Preprocessing and postprocessing utilities
    ├── preprocess.py
    └── postprocess.py
```

## 1. Install Docker
First, install Docker Desktop (available for Windows, macOS, and Linux):

- Download: https://www.docker.com/get-started/

After installation, verify Docker is available:

```sh
docker --version
```
If it shows a version number, Docker is installed correctly.

## 2. Build the Docker Image
Assuming your project code and Dockerfile are in the same directory:

```sh
cd /path/to/docker
docker build -f Dockerfile -t [image_name] .
```

Parameters:
- `-f Dockerfile` — specify the Dockerfile to use.
- `-t [image_name]` — name the image, e.g., uusic.
- `.` — use the current directory as the build context.

Example:
```sh
docker build -f Dockerfile -t PACE2025 .
```

## 3. Run the Docker Container
### For Segmentation Task

To run the **segmentation task** using the Docker container with GPU support:

```sh
docker run --gpus all --rm \
  -v [/path/to/input/images]:/input:ro \
  -v [/path/to/output]:/output \
  -it [image_name] python main.py -i /input -o /output -t seg -d gpu
```
### For Classification Task

To run the **classification task** using the Docker container with GPU support:

```sh
docker run --gpus all --rm \
  -v [/path/to/input/images]:/input:ro \
  -v [/path/to/output]:/output \
  -it [image_name] python main.py -i /input -o /output -t cls -d gpu
```

## Docker Parameters

- `--gpus all` — enable all available GPUs inside the container  
- `--rm` — remove the container automatically after it stops  
- `-v /host/path:/container/path` — mount a local directory into the container:  
  - `/input:ro` — input image directory (read-only)  
  - `/output` — output results directory  
- `-it` — interactive mode  
- `[image_name]` — the Docker image name  

---

## Command-Line Arguments

- `-i, --input` — Path to input directory containing PNG ultrasound images  
- `-o, --output` — Path to output directory  
- `-t, --task` — Task to perform: `seg` for segmentation, `cls` for classification  
- `-d, --device` — Device to use: `cpu` or `gpu` (default: `gpu`)  

## Output Structure

### Segmentation Task Output
```
output_directory/
└── segmentation/
    ├── PACE_00001_000_BUS_mask.png
    ├── PACE_00002_001_BRE_mask.png
    └── ...
```

### Classification Task Output
```
output_directory/
└── classification/
    └── predictions.csv    # Contains image_id, label columns
```
#### predictions.csv
```
image_id,label
PACE_00001_000_BUS,Normal
PACE_00002_001_BRE,Benign
```