# MedGPT: Resources for Multimodal Medical Insights

This repository contains resources for MedGPT, a project aimed at leveraging multimodal Large Language Models (LLMs) to interpret medical images and clinical text, providing insights for healthcare professionals.

**Current State of the Repository:**

*   **Fine-tuned BLIP VQA Model Adapters:** LoRA adapters for a fine-tuned BLIP VQA model are available in the `Model/blip-saved-model/` directory. You can perform inference with these adapters using the `src/blip/infer_blip.py` script.
*   **LLaVA Fine-tuning and Inference Scripts:** This repository includes scripts for preparing data (`src/llava/prepare_llava_data.py`), fine-tuning LLaVA (`src/llava/train_llava.py`), and performing inference (`src/llava/infer_llava.py`).
    *   **Note:** A pre-fine-tuned LLaVA model is *not* included. Users need to prepare datasets (e.g., from [PMC-VQA](https://huggingface.co/datasets/xmcmic/PMC-VQA) or custom data) and run the fine-tuning process using the provided scripts.
    *   The `src/llava/infer_llava.py` script can be used with official LLaVA models from Hugging Face or with user-fine-tuned adapters.

The overall goal is to facilitate better interactions between healthcare professionals and medical data, ultimately improving diagnostic accuracy and patient care.

---

## 🚀 Features

This repository provides resources that support or demonstrate the following capabilities:

- **BLIP VQA Model (Adapters Available):**
    - Enables Visual Question Answering on medical images using fine-tuned LoRA adapters.
    - Fine-tuning was performed using PEFT (LoRA) for efficient adaptation.
    - Inference can be run using the `src/blip/infer_blip.py` script with the provided adapters.

- **LLaVA (Large Language and Vision Assistant) (Scripts for DIY Fine-tuning):**
    - **Multimodal Understanding (Potential):**
        - Can be fine-tuned to extract insights from both medical images (e.g., MRIs, X-rays) and clinical text data.
        - Aims to provide ChatGPT-like outputs tailored for medical diagnostics.
    - **Sophisticated Model Engineering (Scripts Provided):**
        - The `src/llava/train_llava.py` script demonstrates fine-tuning LLaVA using Parameter Efficient Fine-Tuning (PEFT) with LoRA.
        - This approach optimizes model performance without extensively modifying pre-trained weights. Data preparation for this script is handled by `src/llava/prepare_llava_data.py`.
    - **Potential Applications (If fine-tuned by user):**
        - Medical image analysis with text-based reporting.
        - Clinical text comprehension and summarization.
        - Assisting doctors with diagnostics and medical decision-making.

---

## 📊 Dataset

- **Primary Dataset for Fine-tuning Examples:** The fine-tuning examples in this repository were originally based on concepts from the [PMC-VQA dataset](https://huggingface.co/datasets/xmcmic/PMC-VQA).
  - PMC-VQA is a comprehensive dataset designed for vision-based question answering in medical contexts.
  - It includes annotated data combining medical images and text, suitable for training models like BLIP and LLaVA.
  - Users will need to source this or their own datasets for the fine-tuning scripts (`src/blip/train_blip.py`, `src/llava/prepare_llava_data.py`, and `src/llava/train_llava.py`).

---

## 🛠️ Tech Stack

- **Visual Question Answering (VQA) Models:**
    - **BLIP:** Fine-tuned LoRA adapters provided.
    - **LLaVA (Large Language and Vision Assistant):** Scripts provided for user fine-tuning.
- **Fine-Tuning Approaches:**
    - Parameter Efficient Fine-Tuning (PEFT) with LoRA.
- **Framework:** PyTorch
- **Key Libraries & Tools:** Hugging Face Transformers, PEFT, Accelerate, BitsAndBytes, LLaVA (custom install from source).

---

## Repository Structure

*   `README.md`: This file.
*   `requirements.txt`: A file listing Python dependencies for the project.
*   `src/`: Directory containing all core Python scripts.
    *   `src/__init__.py`
    *   `src/blip/`: Contains scripts related to the BLIP model.
        *   `src/blip/__init__.py`
        *   `src/blip/blip_dataset.py`: Utility script for BLIP VQA dataset handling.
        *   `src/blip/train_blip.py`: Script for fine-tuning the BLIP VQA model with LoRA.
        *   `src/blip/infer_blip.py`: Script for performing inference with the fine-tuned BLIP LoRA adapters.
    *   `src/llava/`: Contains scripts related to the LLaVA model.
        *   `src/llava/__init__.py`
        *   `src/llava/llava_dataset.py`: Utility script for LLaVA dataset preparation (tokenization, prompt formatting).
        *   `src/llava/prepare_llava_data.py`: Script to process raw LLaVA conversational datasets.
        *   `src/llava/train_llava.py`: Script for fine-tuning the LLaVA model with LoRA.
        *   `src/llava/infer_llava.py`: Script for performing inference with LLaVA models.
*   `Model/blip-saved-model/`: Directory containing the fine-tuned BLIP VQA LoRA adapters and configuration.
*   `HIPAA_GDPR_Compliance_MedGPT.pdf`: A document discussing data privacy, security, and compliance considerations.
*   `Model.zip`: An archive containing a copy of the `Model/blip-saved-model/` directory.

---

## Compliance & Data Security (HIPAA & GDPR)

- We take data privacy and protection seriously. MedGPT is designed to be compliant with both HIPAA (for U.S. healthcare data) and GDPR (for EU personal data) standards.
- The `HIPAA_GDPR_Compliance_MedGPT.pdf` document in this repository provides an overview of key considerations. (Note: The automated summarization of this PDF was not possible due to its format; users should refer to the PDF directly.)
- To support compliance, any infrastructure code for setting up secure cloud environments (if provided separately) would typically include:
    *   Scripts to provision encrypted storage, IAM roles with restricted access (MFA), cloud logging/monitoring, security alerts, and threat detection.
    *   Tutorials for de-identifying medical data, converting images for safe testing, implementing breach notifications, audit logging, and enforcing access control.
- **User Responsibility:** Users are responsible for ensuring their use of these tools and any data complies with all applicable regulations, including de-identifying data before use.

---

## ⚙️ Setup Guide

This guide provides instructions to set up your environment to run the scripts in this repository.

### 1. Prerequisites

*   **Python:** Version 3.8 or higher is recommended.
*   **CUDA:** If you plan to use GPU acceleration for model training or inference (highly recommended), ensure you have a compatible NVIDIA GPU and have installed the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and corresponding drivers.
*   **Git:** For cloning the repository and the LLaVA library.

### 2. Environment Setup

It is highly recommended to use a virtual environment to manage dependencies.

**Using `venv` (standard Python):**
1.  Create: `python3 -m venv medgpt-env`
2.  Activate:
    *   macOS/Linux: `source medgpt-env/bin/activate`
    *   Windows: `medgpt-env\Scripts\activate`

**Using `conda`:**
1.  Create: `conda create -n medgpt-env python=3.9` (or your preferred Python 3.8+ version)
2.  Activate: `conda activate medgpt-env`

### 3. Clone the Repository
```bash
git clone https://github.com/pratikjadahv2726/MedGPT.git
cd MedGPT
```

### 4. Install Python Libraries

**a. Install Core Libraries from `requirements.txt`:**
Install the core Python dependencies using pip:
```bash
pip install -r requirements.txt
```
This file includes libraries such as `torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`, `datasets`, `Pillow`, and `tqdm`.

**b. LLaVA Library Installation (Required for LLaVA scripts):**
The LLaVA scripts (`src/llava/train_llava.py`, `src/llava/infer_llava.py`, `src/llava/llava_dataset.py`, `src/llava/prepare_llava_data.py`) require the LLaVA package to be installed from its source repository.
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
cd .. # Return to the MedGPT repository root
```
**Note:** If you encounter issues, refer to the [official LLaVA repository](https://github.com/haotian-liu/LLaVA) for the latest installation instructions.

**c. For BLIP model scripts:**
The libraries installed via `requirements.txt` are sufficient for the BLIP scripts (`src/blip/train_blip.py`, `src/blip/infer_blip.py`, `src/blip/blip_dataset.py`).

### 5. Running the Scripts

Python scripts can be run directly from your activated terminal. Use the `--help` flag with any script to see all available command-line options (e.g., `python src/blip/train_blip.py --help`).

*   **Prepare LLaVA data:**
    ```bash
    python src/llava/prepare_llava_data.py --input_file path/to/your/filtered_dataset.json --output_file path/to/your/final_data.json --format captions
    ```
*   **Train BLIP model:**
    ```bash
    python src/blip/train_blip.py --data_path path/to/your/train_2.csv --image_dir path/to/your/blip_images --output_dir Model/blip-saved-model-custom
    ```
*   **Train LLaVA model:**
    ```bash
    python src/llava/train_llava.py --data_path path/to/your/final_data.json --image_dir path/to/your/llava_images --output_dir Model/llava-custom-adapters
    ```
*   **Run BLIP inference:**
    ```bash
    python src/blip/infer_blip.py --model_path Model/blip-saved-model --image_path path/to/your/test_image.jpg --question "What does this image show?"
    ```
*   **Run LLaVA inference (with fine-tuned adapters):**
    ```bash
    python src/llava/infer_llava.py --base_model_id liuhaotian/llava-v1.6-mistral-7b --adapter_path Model/llava-custom-adapters --image_file path/to/your/test_image.jpg --prompt "Describe this image in detail."
    ```
*   **Run LLaVA inference (with base model):**
    ```bash
    python src/llava/infer_llava.py --model_path_or_id liuhaotian/llava-v1.6-mistral-7b --image_file path/to/your/test_image.jpg --prompt "Describe this image in detail."
    ```

**Important Data Dependencies:**
The fine-tuning scripts require specific data files and image directories that are **not** included in this repository. The "💾 Data Preparation Guide" below provides details on the expected data formats.

---

## 💾 Data Preparation Guide

This section outlines the structure and content of the data files required to run the fine-tuning scripts for both the LLaVA and BLIP models. As these specific data files are not included in the repository, you will need to source or prepare them according to these guidelines.

### 1. LLaVA Workflow Data

The LLaVA fine-tuning process (`src/llava/train_llava.py`) relies on data generated by the `src/llava/prepare_llava_data.py` script.

**a. `filtered_dataset.json` (Input for `src/llava/prepare_llava_data.py`)**

*   **Origin:** This file is the initial input for the LLaVA data processing pipeline. You would typically need to obtain a dataset like [PMC-VQA](https://huggingface.co/datasets/xmcmic/PMC-VQA) (or a similar conversational/VQA dataset) and reformat it into this structure if it's not already.
*   **Structure:** A JSON file containing a list of objects. Each object represents a data instance.
    *   `image`: (string) The filename of the image (e.g., "PMC12345_figure1.jpg").
    *   `conversations`: (list of objects) Each object represents a turn in a conversation.
        *   `from`: (string) "human" for user turns, "gpt" for assistant turns.
        *   `value`: (string) The text content of the turn.
*   **Example:** (Refer to `src/llava/prepare_llava_data.py` docstring for a conceptual example)

**b. `final_data.json` (Output of `src/llava/prepare_llava_data.py`, Input for `src/llava/train_llava.py`)**

*   **Origin:** Generated by `src/llava/prepare_llava_data.py` (typically using the `--format captions` option).
*   **Structure:** A JSON file containing a list of objects.
    *   `id`: (integer or string) A unique identifier.
    *   `image`: (string) The image filename. Images are expected in the directory specified by `--image_dir` in `train_llava.py`.
    *   `caption`: (string) The target text/description for the LLaVA model to learn.
*   **Example:** (Refer to `src/llava/prepare_llava_data.py` docstring for a conceptual example)

**c. `images/` Directory (for LLaVA)**

*   **Content:** Contains all image files referenced in `final_data.json`.
*   **Location:** The path to this directory is specified via the `--image_dir` argument to `src/llava/train_llava.py` and `src/llava/infer_llava.py`.

### 2. BLIP Workflow Data (`src/blip/train_blip.py`)

The BLIP fine-tuning (`src/blip/train_blip.py`) uses `src/blip/blip_dataset.py` and requires:

**a. CSV Data File (e.g., `train_2.csv`)**

*   **Origin:** This CSV file is specified via the `--data_path` argument to `src/blip/train_blip.py`.
*   **Structure:** CSV with a header. Expected columns by `src/blip/blip_dataset.py`:
    *   `Figure_path`: Filename/path of the image.
    *   `Caption`: Image caption.
    *   `Question`: Question about the image.
    *   `Answer`: Single character label (e.g., 'A').
    *   `Choice A`, `Choice B`, etc.: Text for multiple-choice options.
*   **Example Row:** (Refer to `src/blip/blip_dataset.py` docstring for a conceptual example)

**b. `figures/` Directory (for BLIP)**

*   **Content:** Contains all image files referenced in the CSV's `Figure_path` column.
*   **Location:** Path specified via the `--image_dir` argument to `src/blip/train_blip.py`.

### 3. General Advice

*   **Data Sourcing:** You are responsible for obtaining/creating these data files.
*   **Path Adjustments:** Use the command-line arguments of the scripts to point to your data.
*   **Start Small:** Test with a small subset of data first.

---

## 🔬 BLIP Model - Fine-tuning and Inference Guide

This section provides step-by-step instructions for fine-tuning the BLIP model using `src/blip/train_blip.py` and then performing inference using `src/blip/infer_blip.py`.

### I. Fine-tuning the BLIP Model (`src/blip/train_blip.py`)

This script adapts a pre-trained BLIP model for a specific medical VQA dataset using PEFT LoRA.

**1. Prerequisites:**
*   Follow the "⚙️ Setup Guide".
*   Prepare your data as per the "💾 Data Preparation Guide" (CSV file and `figures/` image directory).

**2. Run the Training Script:**
*   Execute `src/blip/train_blip.py` from your terminal.
*   **Example Command:**
    ```bash
    python src/blip/train_blip.py \
        --data_path path/to/your/train_2.csv \
        --image_dir path/to/your/figures \
        --output_dir Model/blip-saved-model-custom \
        --epochs 20 \
        --batch_size 8 \
        --learning_rate 4e-5
    ```
*   Use `python src/blip/train_blip.py --help` for all options.
*   **GPU Required.**

**3. Output:**
*   Fine-tuned LoRA adapters and configuration are saved to the specified `--output_dir`.

**4. Troubleshooting:**
*   **File Not Found:** Check `--data_path` and `--image_dir`.
*   **CUDA/GPU Issues:** Ensure correct setup; try reducing `--batch_size`.
*   **Import Errors:** Verify `requirements.txt` installation.

### II. Performing Inference with the Fine-tuned BLIP Model (`src/blip/infer_blip.py`)

**1. Prerequisites:**
*   Successful BLIP fine-tuning (`src/blip/train_blip.py`).
*   Fine-tuned adapter files in the output directory from training.

**2. Run the Inference Script:**
*   **Example Command:**
    ```bash
    python src/blip/infer_blip.py \
        --model_path Model/blip-saved-model \
        --image_path path/to/your/test_image.jpg \
        --question "What are the key findings in this image?"
    ```
*   Use `python src/blip/infer_blip.py --help` for options.

**3. Output:**
*   The model's generated answer is printed to the console.

---

## 🌋 LLaVA Model - Fine-tuning and Inference Guide

This section provides step-by-step instructions for preparing data, fine-tuning the LLaVA model, and performing inference.

### I. Data Preparation for LLaVA (`src/llava/prepare_llava_data.py`)

This script processes an input JSON file (e.g., `filtered_dataset.json`) into `final_data.json` for training.

**1. Prerequisites:**
*   Input data file (e.g., `filtered_dataset.json`) as described in the "💾 Data Preparation Guide".

**2. Run the Data Preparation Script:**
*   **Example Command (for captioning format):**
    ```bash
    python src/llava/prepare_llava_data.py \
        --input_file path/to/your/filtered_dataset.json \
        --output_file path/to/your/final_data.json \
        --format captions
    ```
*   Use `python src/llava/prepare_llava_data.py --help` for options.

**3. Output:**
*   `final_data.json` (or specified name) with `image` and `caption` pairs.

### II. Fine-tuning the LLaVA Model (`src/llava/train_llava.py`)

This script fine-tunes a LLaVA model (e.g., `liuhaotian/llava-v1.6-mistral-7b`) on your prepared custom dataset using PEFT LoRA.

**1. Prerequisites:**
*   Follow the "⚙️ Setup Guide" (including LLaVA source install).
*   Prepared `final_data.json` (from `src/llava/prepare_llava_data.py`).
*   An `images/` directory (or path specified by `--image_dir`) with all images referenced in `final_data.json`.

**2. Run the Training Script:**
*   **Example Command:**
    ```bash
    python src/llava/train_llava.py \
        --model_id liuhaotian/llava-v1.6-mistral-7b \
        --data_path path/to/your/final_data.json \
        --image_dir path/to/your/llava_images \
        --output_dir Model/llava-v1.6-mistral-7b-finetuned-custom \
        --epochs 3 \
        --batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-5 
    ```
    (See `--help` for LoRA and other arguments).
*   **Resource Intensive:** Requires a powerful GPU.

**3. Output:**
*   Fine-tuned LLaVA LoRA adapters saved to the `--output_dir`.

**4. Troubleshooting:**
*   **File Not Found:** Check `--data_path` and `--image_dir`.
*   **CUDA/GPU Issues:** Reduce `--batch_size`; ensure correct setup.
*   **Import Errors for `llava`:** Verify LLaVA source installation.
*   **Dataset Format:** Check `final_data.json` structure for `src/llava/llava_dataset.py`.

### III. Performing Inference with LLaVA Models (`src/llava/infer_llava.py`)

**1. Prerequisites:**
    *   Completed the "⚙️ Setup Guide".
    *   If using fine-tuned adapters, ensure training was successful and you have the adapter path.

**2. Run the Inference Script:**
*   **Using a Base Pre-trained LLaVA Model:**
    ```bash
    python src/llava/infer_llava.py \
        --model_path_or_id liuhaotian/llava-v1.6-mistral-7b \
        --image_file path/to/your/test_image.jpg \
        --prompt "Describe this image in detail."
    ```
*   **Using Fine-tuned LLaVA LoRA Adapters:**
    ```bash
    python src/llava/infer_llava.py \
        --base_model_id liuhaotian/llava-v1.6-mistral-7b \
        --adapter_path path/to/your/llava-adapters \
        --image_file path/to/your/test_image.jpg \
        --prompt "What are the key findings in this medical image?"
    ```
*   Use `python src/llava/infer_llava.py --help` for all options.

**3. Output:**
*   The script prints the model's generated text to the console.

---

## 💡 Example Use Cases

This section provides illustrative examples of how the MedGPT repository's components can be applied. These are conceptual demonstrations; users would adapt these workflows to their specific medical images, questions, and datasets.

### 1. BLIP Model (Visual Question Answering)

**Scenario:** A healthcare professional reviewing a chest X-ray.
*   **Input:** Image (`chest_xray_study_01.png`), Question ("Are there any visible signs of pleural effusion...?").
*   **Process:**
    1.  The user runs the `src/blip/infer_blip.py` script, providing the path to their fine-tuned model adapters (via `--model_path`), the path to the image (`chest_xray_study_01.png` via `--image_path`), and the question ("Are there any visible signs of pleural effusion in the lower right lung field?" via `--question`).
*   **Conceptual Output:** "Based on the visual information, there appears to be [...] in the lower right lung field..."
*   **❗ Disclaimer:** *Model outputs are for research/informational purposes and not a substitute for professional medical advice.*

### 2. LLaVA Model (Image Captioning / Visual Question Answering)

**Scenario 1: Generating a Descriptive Caption for a Pathology Slide**
*   **Input:** Image (`pathology_slide_03.tiff`), Prompt ("Describe this microscopic image in detail.").
*   **Process:**
    1.  The user employs their fine-tuned LLaVA model by using the `src/llava/infer_llava.py` script with the `--adapter_path` argument pointing to their fine-tuned LLaVA adapters, along with the image path and prompt.
*   **Conceptual Output:** "This microscopic image displays [detailed description of cellular structures]..."

**Scenario 2: Answering Questions about a Medical Diagram**
*   **Input:** Image (`anatomical_diagram.svg`), Question ("Which nerve roots form the superior trunk...?").
*   **Process:**
    *   Similar to the captioning scenario, the user provides the diagram image and the question to their fine-tuned LLaVA model using the `src/llava/infer_llava.py` script.
*   **Conceptual Output:** "The superior trunk... is formed by the C5 and C6 nerve roots."
*   **❗ Disclaimer:** *Model outputs are for research/informational purposes and not a substitute for professional medical advice.*

### 3. General Encouragement
Explore these tools with your own (appropriately de-identified) medical data. Always adhere to data privacy and ethical guidelines.

---

## 🚀 Getting Started & Next Steps

You've reached the end of the main README! To get started with the code:

1.  **Set up your environment:** Follow the "⚙️ Setup Guide" for detailed instructions on prerequisites, environment creation, and library installation.
2.  **Prepare your data:** Consult the "💾 Data Preparation Guide" to understand the data requirements for the BLIP and LLaVA models and how to structure your datasets.
3.  **Explore the models:**
    *   For **BLIP fine-tuning and inference**, refer to the "🔬 BLIP Model - Fine-tuning and Inference Guide".
    *   For **LLaVA fine-tuning and inference**, refer to the "🌋 LLaVA Model - Fine-tuning and Inference Guide".
4.  **Review example use cases:** The "💡 Example Use Cases" section provides conceptual ideas on how these models can be applied.

For detailed explanations of each script's functionality and command-line arguments, refer to the respective script's `--help` option and the docstrings within the Python files.
