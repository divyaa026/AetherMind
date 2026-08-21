# AetherMInd: Mental Health Crisis Detection System

A production-ready, privacy-preserving mental health crisis detection system with multi-modal ML capabilities, federated learning, and real-time intervention capabilities.

## 🚨 Critical Safety Notice

**This system is designed for clinical use and requires proper medical oversight. Always ensure:**
- Clinical validation before deployment
- Emergency protocols are in place
- HIPAA/GDPR compliance
- Fallback mechanisms for system failures

## 🏗️ Architecture Overview

### Core Components
1. **Multi-modal ML Pipeline**
   - Text Analysis: Fine-tuned BERT + BiLSTM
   - Behavioral Analysis: Isolation Forest + One-Class SVM
   - Temporal Modeling: LSTM + Transformer Encoder
   - Ensemble: Stacking classifier with meta-learner

2. **Privacy-Preserving Infrastructure**
   - Federated Learning: FedAvg with differential privacy
   - Secure data handling: Homomorphic encryption
   - On-device processing: TensorFlow Lite/Core ML

3. **Full Stack Architecture**
   - Backend: FastAPI + PostgreSQL + Redis
   - Frontend: React + TypeScript + Tailwind CSS
   - Mobile: React Native (iOS/Android)
   - Real-time: WebSocket/Socket.io for crisis interventions

## 📊 Datasets

### Available Datasets
- `depression_dataset_reddit_cleaned.csv`: Reddit depression posts (2.7MB)
- `Suicide_Detection.csv`: Suicide detection dataset (159MB)

### Required External Datasets
1. **Clinical Text Data:**
   - Crisis Text Line (CTL) dataset
   - Reddit r/SuicideWatch dataset (Kaggle)
   - DAIC-WOZ depression corpus (AVEC 2016)

2. **Behavioral & Temporal Data:**
   - StudentLife dataset (Dartmouth College)
   - Tesserae mobile sensor dataset (Harvard)
   - Mindful Mood Tracker dataset (Kaggle)

3. **Clinical Validation Data:**
   - PHQ-9/GAD-7 labeled datasets (UK Biobank)
   - Columbia Suicide Severity Rating Scale samples (NIMH)
   - DASS-21 normative populations dataset

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Kubernetes cluster (for production)
- PostgreSQL 13+
- Redis 6+

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd MindGuard

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Start development environment
docker-compose up -d

# Run data ingestion pipeline
python scripts/data_ingestion.py

# Train initial models
python scripts/train_models.py

# Start the application
python main.py
```

## 📁 Project Structure

```
MindGuard/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core functionality
│   │   ├── models/         # ML models
│   │   ├── services/       # Business logic
│   │   └── utils/          # Utilities
│   ├── tests/              # Backend tests
│   └── requirements.txt
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API services
│   │   └── utils/          # Utilities
│   └── package.json
├── mobile/                 # React Native app
├── ml/                     # ML pipeline
│   ├── models/             # Model implementations
│   ├── training/           # Training scripts
│   ├── evaluation/         # Evaluation scripts
│   └── federated/          # Federated learning
├── infrastructure/         # Infrastructure as code
│   ├── kubernetes/         # K8s manifests
│   ├── terraform/          # Terraform scripts
│   └── docker/             # Docker configurations
├── scripts/                # Utility scripts
├── tests/                  # Integration tests
└── docs/                   # Documentation
```

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/mindguard
REDIS_URL=redis://localhost:6379

# ML Models
MODEL_PATH=/models
FEDERATED_LEARNING_ENABLED=true
DIFFERENTIAL_PRIVACY_EPSILON=1.0

# Security
JWT_SECRET=your-secret-key
ENCRYPTION_KEY=your-encryption-key

# External APIs
CRISIS_LINE_API_KEY=your-api-key
OPENAI_API_KEY=your-api-key

# Monitoring
EVIDENTLY_AI_KEY=your-key
SENTRY_DSN=your-sentry-dsn
```

## 🧪 Testing

### Run Tests
```bash
# Backend tests
cd backend && pytest

# Frontend tests
cd frontend && npm test

# Integration tests
pytest tests/integration/

# Load testing
locust -f tests/load/locustfile.py
```

## 📈 Performance Metrics

### Target Metrics
- **Sensitivity**: >95% (crisis detection)
- **Specificity**: >85% (confirmed by clinicians)
- **Response Time**: <30 seconds end-to-end
- **Federated Learning**: <5% variance in convergence
- **Inference Latency**: <30ms for text analysis

## 🔒 Privacy & Security

### Privacy Features
- Federated learning with >100 simulated nodes
- k-anonymity (k=50) for all PII
- Homomorphic encryption for sensitive data
- On-device model updates via TF Lite

### Security Measures
- HIPAA-compliant audit logging
- End-to-end encryption
- Secure model serving
- Access control and authentication

## 🚨 Crisis Intervention Protocol

### Escalation Levels
1. **Level 1**: Automated risk assessment
2. **Level 2**: Human-in-the-loop review
3. **Level 3**: Emergency services notification
4. **Level 4**: Direct crisis intervention

### Safety Mechanisms
- Fallback to human operators
- Emergency contact protocols
- Geographic-based intervention routing
- Clinical oversight dashboard

## 📊 Monitoring & Observability

### Model Monitoring
- Evidently AI for drift detection
- Performance metrics tracking
- Bias detection and mitigation
- Clinical validation reports

### System Monitoring
- Prometheus + Grafana
- Distributed tracing (Jaeger)
- Log aggregation (ELK stack)
- Health checks and alerts

## 🚀 Deployment

### Development
```bash
docker-compose up -d
```

### Production
```bash
# Deploy to Kubernetes
kubectl apply -f infrastructure/kubernetes/

# Deploy infrastructure
terraform apply -f infrastructure/terraform/
```

## 📚 Documentation

- [API Documentation](docs/api.md)
- [ML Pipeline Guide](docs/ml-pipeline.md)
- [Privacy Implementation](docs/privacy.md)
- [Clinical Integration](docs/clinical.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This system is for research and clinical use only. Always ensure proper medical oversight and emergency protocols are in place before deployment.

## 🆘 Emergency Contacts

- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741
- Emergency Services: 911

---

**Built with ❤️ for mental health awareness and crisis prevention**
