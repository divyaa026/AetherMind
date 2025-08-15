# MindGuard Deployment Guide

This guide provides step-by-step instructions for deploying the MindGuard mental health crisis detection system in various environments.

## 🚀 Quick Start Deployment

### Prerequisites

- Docker & Docker Compose
- Python 3.9+
- 8GB+ RAM
- 50GB+ disk space
- NVIDIA GPU (optional, for faster training)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd MindGuard

# Create environment file
cp .env.example .env

# Edit environment variables
nano .env
```

### 2. Environment Configuration

```bash
# Required environment variables
POSTGRES_PASSWORD=your_secure_password
JWT_SECRET=your_jwt_secret_key
ENCRYPTION_KEY=your_encryption_key
EVIDENTLY_AI_KEY=your_evidently_key
GRAFANA_PASSWORD=admin_password

# Optional variables
FEDERATED_LEARNING_ENABLED=true
DIFFERENTIAL_PRIVACY_EPSILON=1.0
```

### 3. Data Processing

```bash
# Process available datasets
python scripts/data_ingestion.py

# Verify processed data
ls -la processed_data/
```

### 4. Model Training

```bash
# Train initial models
python scripts/train_models.py --epochs 5 --batch-size 8

# Verify trained models
ls -la trained_models/
```

### 5. Start Services

```bash
# Start core services
docker-compose up -d postgres redis backend frontend

# Start with monitoring
docker-compose --profile monitoring up -d

# Start with federated learning
docker-compose --profile federated up -d
```

## 🏗️ Production Deployment

### Kubernetes Deployment

1. **Create Kubernetes Cluster**

```bash
# Using minikube for local testing
minikube start --cpus 4 --memory 8192

# Or use cloud provider (AWS EKS, GKE, Azure AKS)
```

2. **Deploy Infrastructure**

```bash
# Apply infrastructure manifests
kubectl apply -f infrastructure/kubernetes/

# Verify deployment
kubectl get pods -n mindguard
```

3. **Configure Ingress**

```bash
# Apply ingress configuration
kubectl apply -f infrastructure/kubernetes/ingress.yaml

# Get external IP
kubectl get ingress -n mindguard
```

### AWS Deployment

1. **Setup AWS Infrastructure**

```bash
# Initialize Terraform
cd infrastructure/terraform/aws
terraform init

# Configure variables
cp terraform.tfvars.example terraform.tfvars
nano terraform.tfvars

# Deploy infrastructure
terraform plan
terraform apply
```

2. **Deploy Application**

```bash
# Build and push Docker images
docker build -t mindguard-backend ./backend
docker build -t mindguard-frontend ./frontend
docker push your-registry/mindguard-backend:latest
docker push your-registry/mindguard-frontend:latest

# Deploy to EKS
kubectl apply -f infrastructure/kubernetes/aws/
```

### GCP Deployment

1. **Setup GCP Infrastructure**

```bash
# Initialize Terraform
cd infrastructure/terraform/gcp
terraform init

# Configure variables
cp terraform.tfvars.example terraform.tfvars
nano terraform.tfvars

# Deploy infrastructure
terraform plan
terraform apply
```

2. **Deploy to GKE**

```bash
# Get cluster credentials
gcloud container clusters get-credentials mindguard-cluster --zone=us-central1-a

# Deploy application
kubectl apply -f infrastructure/kubernetes/gcp/
```

## 🔒 Security Configuration

### SSL/TLS Setup

```bash
# Generate SSL certificates
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/nginx.key \
  -out nginx/ssl/nginx.crt

# Configure nginx
cp nginx/nginx.conf.example nginx/nginx.conf
```

### Database Security

```bash
# Enable SSL for PostgreSQL
echo "ssl = on" >> postgresql.conf
echo "ssl_cert_file = '/etc/ssl/certs/server.crt'" >> postgresql.conf
echo "ssl_key_file = '/etc/ssl/private/server.key'" >> postgresql.conf
```

### API Security

```bash
# Configure rate limiting
# Add to nginx.conf
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req zone=api burst=20 nodelay;
```

## 📊 Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mindguard-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboards

```bash
# Import dashboards
cp monitoring/grafana/dashboards/* /etc/grafana/provisioning/dashboards/
cp monitoring/grafana/datasources/* /etc/grafana/provisioning/datasources/
```

### Alerting Rules

```yaml
# monitoring/alerts.yml
groups:
  - name: mindguard-alerts
    rules:
      - alert: HighCrisisDetectionRate
        expr: rate(crisis_detections_total[5m]) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High crisis detection rate detected"
```

## 🔄 CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy MindGuard

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # Deploy to production environment
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

test:
  stage: test
  script:
    - pip install -r requirements.txt
    - pytest tests/

build:
  stage: build
  script:
    - docker build -t mindguard:$CI_COMMIT_SHA .
    - docker push registry/mindguard:$CI_COMMIT_SHA

deploy:
  stage: deploy
  script:
    - kubectl set image deployment/mindguard mindguard=registry/mindguard:$CI_COMMIT_SHA
```

## 🧪 Testing

### Load Testing

```bash
# Run load tests
docker-compose --profile testing up -d locust

# Access Locust UI at http://localhost:8089
# Configure test parameters and start testing
```

### Performance Testing

```bash
# Run performance tests
python tests/performance/test_api_performance.py

# Generate performance report
python tests/performance/generate_report.py
```

### Security Testing

```bash
# Run security scans
docker run --rm -v $(pwd):/app owasp/zap2docker-stable zap-baseline.py -t http://localhost:8000

# Run dependency vulnerability scan
safety check -r requirements.txt
```

## 🔧 Maintenance

### Database Backups

```bash
# Create backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker exec mindguard-postgres pg_dump -U mindguard_user mindguard > backup_$DATE.sql

# Restore backup
docker exec -i mindguard-postgres psql -U mindguard_user mindguard < backup_20231201_120000.sql
```

### Log Management

```bash
# Configure log rotation
cat > /etc/logrotate.d/mindguard << EOF
/var/log/mindguard/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 root root
}
EOF
```

### Model Updates

```bash
# Retrain models periodically
python scripts/train_models.py --data-path processed_data --output-dir models/latest

# Deploy updated models
kubectl rollout restart deployment/mindguard-backend
```

## 🚨 Emergency Procedures

### Crisis Escalation

1. **Immediate Response**
   ```bash
   # Check system status
   kubectl get pods -n mindguard
   
   # Check logs for errors
   kubectl logs -f deployment/mindguard-backend
   ```

2. **Fallback Procedures**
   ```bash
   # Switch to backup systems
   kubectl patch deployment/mindguard-backend -p '{"spec":{"replicas":0}}'
   kubectl patch deployment/mindguard-backup -p '{"spec":{"replicas":3}}'
   ```

3. **Emergency Contacts**
   - System Administrator: [admin@mindguard.com]
   - Clinical Lead: [clinical@mindguard.com]
   - Emergency Services: 911

### Data Recovery

```bash
# Restore from backup
docker exec -i mindguard-postgres psql -U mindguard_user mindguard < latest_backup.sql

# Verify data integrity
python scripts/verify_data_integrity.py
```

## 📈 Scaling

### Horizontal Scaling

```bash
# Scale backend services
kubectl scale deployment mindguard-backend --replicas=5

# Scale database
kubectl scale statefulset postgres --replicas=3
```

### Vertical Scaling

```bash
# Update resource limits
kubectl patch deployment mindguard-backend -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "backend",
          "resources": {
            "requests": {"memory": "2Gi", "cpu": "1000m"},
            "limits": {"memory": "4Gi", "cpu": "2000m"}
          }
        }]
      }
    }
  }
}'
```

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Issues**
   ```bash
   # Check database status
   docker exec mindguard-postgres pg_isready -U mindguard_user
   
   # Check connection logs
   docker logs mindguard-postgres
   ```

2. **Model Loading Issues**
   ```bash
   # Check model files
   ls -la trained_models/
   
   # Verify model integrity
   python scripts/verify_models.py
   ```

3. **Memory Issues**
   ```bash
   # Check memory usage
   docker stats
   
   # Increase memory limits
   docker-compose down
   docker-compose up -d --scale backend=2
   ```

### Performance Optimization

```bash
# Enable caching
docker-compose up -d redis

# Optimize database queries
docker exec mindguard-postgres psql -U mindguard_user -d mindguard -c "ANALYZE;"

# Enable compression
echo "gzip on;" >> nginx/nginx.conf
```

## 📚 Additional Resources

- [API Documentation](docs/api.md)
- [ML Pipeline Guide](docs/ml-pipeline.md)
- [Privacy Implementation](docs/privacy.md)
- [Clinical Integration](docs/clinical.md)

## 🆘 Support

For deployment support:
- Email: [support@mindguard.com]
- Documentation: [docs.mindguard.com]
- Emergency: [emergency@mindguard.com]

---

**Remember: This system handles sensitive mental health data. Always ensure proper security measures and clinical oversight are in place.**
