# AI Safety Pipeline - Proof of Concept

A comprehensive AI Safety system that integrates multiple ML models to enhance user safety in conversational platforms.

## 🎯 Overview

This POC implements a multi-stage safety pipeline that processes messages through:

1. **Abuse Detection** - Identifies hate speech and offensive language
2. **Escalation Pattern Recognition** - Tracks conversation escalation over time  
3. **Crisis Intervention** - Detects suicide/self-harm indicators
4. **Age-Appropriate Filtering** - Ensures content suitability by age

## 🏗️ Architecture

```
Message → Abuse Check → Escalation Check → Crisis Check → Age Filter → Response/Alert
```

### Models Used

- **Offensive Language Detection**: DistilBERT classifier
  - Classes: `hate_speech`, `offensive_language`, `neither`
  - Path: `Offensive_Detection/Output/checkpoint-840`

- **Suicide Detection**: DistilBERT classifier  
  - Classes: `non-suicide`, `suicide`
  - Path: `Suicide_Detection/Output/checkpoint-1965`

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Download Pre-trained Models

Download and extract the trained models and datasets:

**Offensive Detection Model & Dataset:**
- Download: [Offensive_Detection.zip](https://drive.google.com/file/d/18o4EfAdLHJcpOyujx6LGGY4bwyJGmihz/view?usp=sharing)
- Extract to project root as `Offensive_Detection/`

**Suicide Detection Model & Dataset:**
- Download: [Suicide_Detection.zip](https://drive.google.com/file/d/1qp6-mNpkYyuriUBAGqQ8x7oG6df1xwIn/view?usp=sharing)
- Extract to project root as `Suicide_Detection/`

**Quick Setup:**
```bash
# After downloading both zip files
unzip Offensive_Detection.zip
unzip Suicide_Detection.zip
```

## 🎓 Model Training

### Prerequisites

```bash
# Install training dependencies
pip install torch transformers datasets accelerate
```

### 1. Offensive Language Detection Model

**Dataset**: `Offensive_Detection/Dataset/labeled_data.csv` (24,783 samples)
- hate_speech: 1,430 samples (5.8%)
- offensive_language: 19,190 samples (77.4%)
- neither: 4,163 samples (16.8%)

**Training Steps**:

```bash
# Navigate to training scripts directory
cd Training_Scripts

# Open and run the training notebook
jupyter notebook Training_Offensive_Detection.ipynb
```

**Training Notebook**: `Training_Scripts/Training_Offensive_Detection.ipynb`

**Training Configuration**:
- Model: DistilBERT-base-uncased
- Epochs: 3
- Batch Size: 64
- Learning Rate: 2e-5
- Best Checkpoint: checkpoint-840 (after 840 steps)

### 2. Suicide Detection Model

**Dataset**: `Suicide_Detection.csv`
- Binary classification: suicide vs non-suicide
- Mental health conversation data with crisis intervention examples

**Training Steps**:

```bash
# Navigate to training scripts directory
cd Training_Scripts

# Open and run the training notebook
jupyter notebook Training_Suicide_Detection.ipynb
```

**Training Notebook**: `Training_Scripts/Training_Suicide_Detection.ipynb`

**Training Configuration**:
- Model: DistilBERT-base-uncased
- Epochs: 3
- Batch Size: 64
- Learning Rate: 2e-5
- Best Checkpoint: checkpoint-1965 (after 1965 steps)

### Model Evaluation

```bash
# Evaluate trained models
python evaluate_models.py
```

**Performance Metrics**:
- Offensive Detection: 92.1% accuracy, 0.919 F1-score
- Suicide Detection: 97.4% accuracy, 0.974 F1-score

### Run Streamlit Demo

```bash
streamlit run streamlit_demo.py
```

### Run CLI Demo

```bash
python cli_demo.py
```

### Run Basic Pipeline

```bash
python ai_safety_pipeline.py
```

## 📊 Features

### Real-time Safety Analysis
- Instant message classification
- Multi-model inference pipeline
- Confidence scoring for all predictions

### Escalation Tracking
- User conversation history monitoring
- Pattern recognition for increasing aggression
- Configurable escalation thresholds

### Crisis Intervention
- Suicide/self-harm detection
- Automatic human review triggers
- Emergency response protocols

### Age-Appropriate Filtering
- Rule-based content filtering
- Age-specific restrictions
- Guardian supervision support

## 🎮 Demo Interface

The Streamlit interface provides:

- **Interactive message testing**
- **Real-time safety analysis**
- **User history tracking**
- **Escalation score monitoring**
- **Sample message testing**

## 📈 Model Performance

### Offensive Language Detection
- Training dataset: 24,783 samples
- Classes distribution:
  - Offensive Language: 77.4%
  - Neither: 16.8%
  - Hate Speech: 5.8%

### Suicide Detection
- Binary classification (suicide/non-suicide)
- Trained on mental health conversation data

## 🔧 Configuration

### Escalation Thresholds
- Default escalation threshold: 60%
- History window: Last 10 messages
- Evaluation window: Last 5 messages

### Age Filtering Rules
- Under 13: Blocks adult keywords
- Under 18: Blocks explicit content
- Configurable keyword lists

## 🛡️ Safety Actions

| Condition | Action | Human Review |
|-----------|--------|--------------|
| Crisis Detected | BLOCK_AND_ALERT | ✅ Required |
| High Escalation | WARN_AND_MONITOR | ✅ Required |
| Hate Speech | BLOCK | ❌ Optional |
| Offensive Language | WARN | ❌ Optional |
| Age Inappropriate | BLOCK | ❌ Optional |
| Safe Content | ALLOW | ❌ None |

## 📁 Project Structure

```
Solulab/
├── ai_safety_pipeline.py      # Main pipeline implementation
├── streamlit_demo.py          # Web interface demo
├── cli_demo.py               # Command-line demo
├── evaluate_models.py        # Model evaluation script
├── requirements.txt          # Dependencies
├── README.md                # This file
├── .gitattributes           # Git LFS configuration
├── Training_Scripts/        # Model training notebooks
│   ├── Training_Offensive_Detection.ipynb
│   └── Training_Suicide_Detection.ipynb
├── Offensive_Detection/      # Abuse detection model
│   ├── Dataset/             # Training dataset
│   │   └── labeled_data.csv # 24,783 samples
│   └── Output/              # Trained model checkpoints
│       └── checkpoint-840/
└── Suicide_Detection/        # Crisis detection model
    ├── Suicide_Detection.csv # Training dataset
    └── Output/              # Trained model checkpoints
        └── checkpoint-1965/
```

## 🔮 Future Enhancements

### Production Scaling
- API endpoint implementation
- Database integration for user history
- Real-time streaming support
- Load balancing for high throughput

### Model Improvements
- Multi-language support
- Context-aware analysis
- Bias mitigation techniques
- Continuous learning pipeline

### Additional Safety Features
- Image/video content analysis
- Voice message processing
- Behavioral pattern analysis
- Community reporting integration

## 🤝 Leadership Considerations

### Team Development Strategy
1. **Modular Architecture** - Easy to extend with new models
2. **A/B Testing Framework** - Compare model performance
3. **Monitoring Dashboard** - Track safety metrics
4. **Feedback Loop** - Continuous model improvement

### Ethical Guidelines
- Bias detection and mitigation
- Transparency in decision making
- User privacy protection
- Fair treatment across demographics

## 📝 Technical Specifications

- **Framework**: PyTorch + Transformers
- **Models**: DistilBERT-based classifiers
- **Interface**: Streamlit web app
- **Requirements**: CPU-compatible (no GPU needed)
- **Latency**: <100ms per message
- **Scalability**: Horizontal scaling ready

## 🎥 Demo Video

Record a 10-minute walkthrough covering:
1. Architecture explanation
2. Model integration approach
3. Live demo with sample inputs
4. Pros/cons discussion
5. Production scaling considerations

---

**Built for AI Safety - Protecting Users Through Intelligent Content Analysis**