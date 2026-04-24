# Visionary: Adaptive Federated Multimodal Learning

![Project Status](https://img.shields.io/badge/status-research--in--progress-orange)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/framework-FastAPI-green)
![Federated](https://img.shields.io/badge/federated-Flower-red)

AFSPL is a state-of-the-art research project focused on **Adaptive Federated Soft Prompt Learning**. It combines the power of large-scale multimodal models (CLIP & Flan-T5) with privacy-preserving federated learning and efficient parameter tuning via soft prompts.

## 🚀 Key Features

- **Federated Learning Architecture**: Built on top of the [Flower (flwr)](https://flower.dev/) framework for scalable, decentralized training.
- **Multimodal Core**: Integrates `CLIP` (Vision Encoder) and `Flan-T5` (Text Decoder) for advanced visual-linguistic tasks.
- **Adaptive Soft Prompting**: Implements dynamic fusion strategies and adaptive top-k token selection for efficient model adaptation.
- **Real-time Dashboard**: A modern React-based frontend for monitoring training rounds, loss curves, and system diagnostics.
- **FastAPI Backend**: High-performance API for handling inference, training control, and metric aggregation.

## 📁 Project Structure

```text
afspl/
├── backend/                # Python FastAPI Backend
│   ├── app/                # Application logic (routes, schemas, core)
│   ├── models/             # Multimodal model definitions (AFSPLModel)
│   ├── federated/          # Federated server and client logic
│   ├── training/           # Training loops, datasets, and losses
│   ├── scripts/            # Utility scripts for running the pipeline
│   ├── configs/            # YAML configuration files
│   └── checkpoints/        # Saved model weights (.pt)
├── frontend/               # React + Vite Dashboard
│   ├── src/                # Component logic and UI
│   └── ...                 # Vite configuration
└── data/                   # Dataset storage (MS-COCO, Flickr30k)
```

## 🛠️ Technology Stack

### Backend
- **Core**: Python 3.10+, PyTorch
- **Models**: HuggingFace Transformers, OpenCLIP
- **Federated**: Flower (flwr)
- **API**: FastAPI, Uvicorn
- **Utilities**: Accelerate, NLTK, Loguru

### Frontend
- **Framework**: React 18 (Vite)
- **Styling**: Tailwind CSS, Framer Motion
- **Charts**: Recharts
- **State Management**: Zustand
- **Icons**: Lucide-React

## 🚦 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Node.js & npm (for the frontend)
- (Optional) CUDA-enabled GPU for faster training

### 2. Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 4. Running Training
To start a federated training session:
1. Start the server:
   ```bash
   python scripts/run_pipeline.py --phase server
   ```
2. Start one or more clients:
   ```bash
   python scripts/run_pipeline.py --phase client
   ```

## 📊 Current Training Status
The project is currently in the training phase. 
- **Target Rounds**: 30
- **Completed Rounds**: 20 (as of last checkpoint)
- **Primary Metric**: CIDEr / BLEU4
