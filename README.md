# Enhanced Keyword Spotter 🎙️ – MLOps Ready

**Goal:** Detect spoken keywords in real-time or from uploaded audio files using an advanced CNN-LSTM-Attention model — fully integrated with modern MLOps practices for scalability, monitoring, and CI/CD automation.  
Built upon the original [Keyword Spotting Project](https://github.com/Jithsaavvy/Deploying-an-end-to-end-keyword-spotting-model-into-cloud-server-by-integrating-CI-CD-pipeline), this enhanced version transforms it into a **production-grade AI audio system**.

---

## ⚡ Major Differences Between Original vs Enhanced Version

| Area | Original Project | Enhanced Version (by Ngoc Quang Le) |
|------|------------------|--------------------------------------|
| **Core Functionality** | Basic file upload & prediction | 🎙️ Real-time recording, file upload, and batch processing |
| **Interface** | Simple HTML form | 💻 Modern tabbed UI with live waveform visualization |
| **Backend** | Minimal Flask app | ⚙️ Production-ready Flask with API endpoints & error handling |
| **Model** | Basic CNN-LSTM | 🧠 CNN-LSTM + Attention + data augmentation |
| **Training** | Manual training | Automated pipeline with LR scheduling, early stopping |
| **Deployment** | Manual | 🚀 CI/CD with GitHub Actions + Docker containerization |
| **Monitoring** | None | 📈 Real-time metrics dashboard + health checks |
| **MLOps** | Not implemented | ✅ MLflow experiment tracking + versioning + monitoring |

---

## 🧠 Key AI/ML Enhancements

- **Advanced CNN-LSTM-Attention model** for more accurate keyword recognition  
- **Learning rate scheduling** and **early stopping** for optimized convergence  
- **Data augmentation** for noise robustness  
- **Fallback inference** for missing model scenarios  
- **MLflow integration** for experiment tracking and version control  
- **Hydra configuration management** for flexible experiments  

---

## 🧩 New Backend Architecture

### REST API Endpoints
| Endpoint | Description |
|-----------|-------------|
| `/api/predict` | Single keyword prediction |
| `/api/batch-predict` | Batch processing for multiple files |
| `/api/metrics` | Model performance and latency metrics |
| `/api/model-info` | Model metadata and version details |
| `/health` | System health and service check |

### Backend Features
- Comprehensive error handling and logging  
- Real-time inference with GPU/CPU fallback  
- Configurable model paths and thresholds  
- Integrated CI/CD pipeline with **GitHub Actions**  
- Docker containerization for reproducible deployment  

---

## 🎨 User Interface Transformation

| Original | Enhanced |
|-----------|-----------|
| Basic HTML form | 💻 Modern responsive design (React + Tailwind or Flask templates) |
| No visual feedback | 🎧 Live waveform visualization using Web Audio API |
| File upload only | 🗂️ Tabs for **Real-time**, **Upload**, **Batch**, **Dashboard** |
| No analytics | 📊 Dashboard with metrics, processing time, and history |

**New UI Features**
- 🎙️ **Real-time microphone recording**
- 📈 **Performance dashboard with accuracy & latency metrics**
- 🗃️ **Batch processing with progress tracking**
- ⚡ **Live visual feedback and system health status**

---

## 🔧 Technical Architecture

```
User → UI (Web Audio / Upload)
     → Flask API (/api/predict or /api/batch-predict)
     → Model Inference (CNN-LSTM-Attention)
     → MLflow Tracking + Logging
     → Dashboard (Metrics + Predictions)
```

| Layer | Technology |
|--------|-------------|
| **Frontend** | HTML5, JS, Web Audio API, Bootstrap / Tailwind |
| **Backend** | Flask (REST API) |
| **Model** | CNN-LSTM + Attention |
| **Training** | TensorFlow / Keras |
| **Tracking** | MLflow |
| **Containerization** | Docker |
| **Automation** | GitHub Actions (CI/CD) |
| **Monitoring** | Health checks + performance metrics |

---

## 🧱 File Structure

```
enhanced-keyword-spotter/
├── app/
│   ├── api.py                 # Flask REST API endpoints
│   ├── utils/                 # Audio processing helpers
│   ├── inference/             # Model loading & prediction logic
│   └── templates/             # Modern web UI (Dashboard, Upload, Realtime)
├── model/
│   ├── train.py               # Advanced model training pipeline
│   ├── model_architecture.py  # CNN-LSTM-Attention network
│   ├── data_preprocessing.py  # Audio data cleaning & augmentation
│   └── config.yaml            # Hydra configuration
├── mlflow/                    # MLflow tracking setup
├── docker/                    # Dockerfile and environment config
├── tests/                     # Unit tests & benchmarks
├── .github/workflows/         # CI/CD pipeline
└── README.md
```

---

## 🚀 MLOps Pipeline

**Enhanced Workflow**
1. 🧑‍💻 Developer pushes code → triggers **GitHub Actions**  
2. 🧱 CI/CD builds & tests automatically  
3. 🐳 Docker container built → deployed to server  
4. 🧠 Model tracked with **MLflow** (metrics, loss, accuracy)  
5. 📊 Dashboard monitors performance and usage in real time  

**Core Tools**
- **GitHub Actions** → CI/CD automation  
- **Docker** → Containerized runtime  
- **MLflow** → Experiment tracking & model registry  
- **Hydra** → Centralized config management  
- **Unit Testing Suite** → Ensures reliability  

---

## 📈 Performance Dashboard

- Real-time accuracy & latency tracking  
- Processing time per file  
- Recent predictions history  
- GPU/CPU utilization monitoring  
- Health check status  
- API usage overview  

---

## 🧰 Future Enhancements

- 🔊 Voice activity detection (VAD)  
- 🌍 Multi-language keyword support  
- 🧠 On-device model inference (TensorFlow Lite)  
- 🧾 User management + API tokens  
- ☁️ Cloud deployment on AWS/GCP/Azure  
- 📦 Model registry integration with MLflow  

---

## 🧩 Educational Value

This enhanced project is ideal for learning:
- End-to-end **AI system design** (model → API → UI → MLOps)
- Real-world **CI/CD + MLflow + Docker** integration
- Scalable audio inference systems
- Robust production-level Flask API design

---

## 🙏 Credits

Original project by [Jithsaavvy](https://github.com/Jithsaavvy)  
Enhanced, refactored, and modernized by [Ngoc Quang Le](https://github.com/QuxngLe).  
Licensed under the MIT License.

---
