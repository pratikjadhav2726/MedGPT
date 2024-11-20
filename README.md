# MedGPT: Multimodal Large Language Model for Medical Insights

MedGPT is an advanced multimodal system combining the power of Large Language and Vision Models (LLMs) to interpret medical images and clinical text. This system provides ChatGPT-like responses tailored for medical diagnostics by extracting meaningful insights from complex data. It aims to facilitate better interactions between healthcare professionals and medical data, ultimately improving diagnostic accuracy and patient care.

---

## 🚀 Features

- **Multimodal Understanding:**
  - Extracts insights from both medical images (e.g., MRIs, X-rays) and clinical text data.
  - Provides ChatGPT-like outputs tailored specifically for medical diagnostics.

- **Sophisticated Model Engineering:**
  - Fine-tuned LLaVA (Large Language and Vision Assistant) using Parameter Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA).
  - Optimized model performance without losing pre-trained weights.

- **Applications:**
  - Medical image analysis with text-based reporting.
  - Clinical text comprehension and summarization.
  - Assists doctors with diagnostics and medical decision-making.

---

## 📊 Dataset

- **Dataset Used:** [PMC-VQA](https://huggingface.co/datasets/xmcmic/PMC-VQA)
  - A comprehensive dataset designed for vision-based question answering in medical contexts.
  - Includes annotated data combining medical images and text for effective model training.

---

## 🛠️ Tech Stack

- **Language Model:** LLaVA (Large Language and Vision Assistant)
- **Fine-Tuning Approach:** Parameter Efficient Fine-Tuning (PEFT) with LoRA
- **Framework:** PyTorch
- **Tools:** Hugging Face Transformers, PEFT, Vision Models

---


---

## 📋 Getting Started

### Prerequisites

- **Python:** v3.8+
- **CUDA:** For GPU acceleration
- **Libraries:**
  - Hugging Face Transformers
  - PEFT
  - PyTorch

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/pratikjadahv2726/MedGPT.git
   cd MedGPT


