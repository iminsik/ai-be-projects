# Deployment Guide

This guide covers deploying the AI Job Queue System in various environments without Docker.

## 🏠 Local Development (No Docker)

### Prerequisites

1. **Python 3.9+** (for backend) and **Python 3.10+** (for AI worker)
2. **uv** package manager
3. **Redis** server

### Quick Start

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install Redis
# macOS:
brew install redis && brew services start redis

# Ubuntu/Debian:
sudo apt update && sudo apt install redis-server
sudo systemctl start redis-server

# 3. Run the system
./scripts/run_local.sh
```

### Manual Setup

```bash
# 1. Setup backend
cd backend
uv sync
uv run python -m uvicorn src.main:app --reload

# 2. Setup AI worker (in another terminal)
cd ai-worker
uv sync
uv run python src/worker.py
```

## 🖥️ Traditional Server Deployment

### Single Server Setup

```bash
# 1. Install system dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip redis-server nginx

# 2. Start Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server

# 3. Setup backend
cd backend
uv sync
sudo cp systemd/ai-backend.service /etc/systemd/system/
sudo systemctl enable ai-backend
sudo systemctl start ai-backend

# 4. Setup AI worker
cd ai-worker
uv sync
sudo cp systemd/ai-worker.service /etc/systemd/system/
sudo systemctl enable ai-worker
sudo systemctl start ai-worker

# 5. Configure Nginx
sudo cp nginx/ai-backend.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/ai-backend.conf /etc/nginx/sites-enabled/
sudo systemctl reload nginx
```

### Multi-Server Setup

```bash
# Server 1: Backend + Redis
# Install and configure backend and Redis

# Server 2: AI Workers
# Install and configure multiple AI workers
# Set REDIS_HOST to point to Server 1
```

## ☁️ Cloud Deployment

### AWS EC2

```bash
# 1. Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx

# 2. Install dependencies
sudo yum update -y
sudo yum install python3.11 redis nginx -y

# 3. Setup services (same as traditional server)
```

### Google Cloud Platform

```bash
# 1. Create VM instance
gcloud compute instances create ai-job-queue \
  --zone=us-central1-a \
  --machine-type=e2-medium \
  --image-family=debian-11 \
  --image-project=debian-cloud

# 2. Install dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv redis-server nginx -y

# 3. Setup services
```

### Azure VM

```bash
# 1. Create VM
az vm create \
  --resource-group myResourceGroup \
  --name ai-job-queue \
  --image UbuntuLTS \
  --size Standard_B2s

# 2. Install dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv redis-server nginx -y

# 3. Setup services
```

## 🐳 Kubernetes Deployment (Alternative to Docker Compose)

### Prerequisites

- Kubernetes cluster
- kubectl configured
- Helm (optional)

### Deploy with kubectl

```bash
# 1. Create namespace
kubectl create namespace ai-job-queue

# 2. Deploy Redis
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/redis-service.yaml

# 3. Deploy backend
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/backend-service.yaml

# 4. Deploy AI workers
kubectl apply -f k8s/ai-worker-deployment.yaml
```

### Deploy with Helm

```bash
# 1. Add Helm repository
helm repo add ai-job-queue https://your-repo.com/charts

# 2. Install the chart
helm install ai-job-queue ai-job-queue/ai-job-queue \
  --namespace ai-job-queue \
  --set backend.replicas=3 \
  --set aiWorker.replicas=5
```

## 🔧 Systemd Service Files

### Backend Service (`systemd/ai-backend.service`)

```ini
[Unit]
Description=AI Job Queue Backend
After=network.target redis.service
Requires=redis.service

[Service]
Type=simple
User=ai-backend
WorkingDirectory=/opt/ai-job-queue/backend
Environment=REDIS_HOST=localhost
Environment=REDIS_PORT=6379
Environment=ENVIRONMENT=production
ExecStart=/opt/ai-job-queue/backend/.venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### AI Worker Service (`systemd/ai-worker.service`)

```ini
[Unit]
Description=AI Job Queue Worker
After=network.target redis.service
Requires=redis.service

[Service]
Type=simple
User=ai-worker
WorkingDirectory=/opt/ai-job-queue/ai-worker
Environment=REDIS_HOST=localhost
Environment=REDIS_PORT=6379
Environment=MODEL_STORAGE_PATH=/opt/ai-job-queue/models
Environment=ENVIRONMENT=production
ExecStart=/opt/ai-job-queue/ai-worker/.venv/bin/python src/worker.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## 🌐 Nginx Configuration

### Backend Proxy (`nginx/ai-backend.conf`)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /docs {
        proxy_pass http://localhost:8000/docs;
        proxy_set_header Host $host;
    }
}
```

## 🔒 Security Considerations

### 1. Network Security

```bash
# Configure firewall
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw deny 6379/tcp   # Redis (internal only)
sudo ufw enable
```

### 2. Redis Security

```bash
# Edit /etc/redis/redis.conf
bind 127.0.0.1
requirepass your-strong-password
```

### 3. Service User

```bash
# Create dedicated users
sudo useradd -r -s /bin/false ai-backend
sudo useradd -r -s /bin/false ai-worker

# Set proper permissions
sudo chown -R ai-backend:ai-backend /opt/ai-job-queue/backend
sudo chown -R ai-worker:ai-worker /opt/ai-job-queue/ai-worker
sudo chown -R ai-worker:ai-worker /opt/ai-job-queue/models
```

## 📊 Monitoring and Logging

### 1. Log Management

```bash
# Configure log rotation
sudo tee /etc/logrotate.d/ai-job-queue << EOF
/var/log/ai-backend.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 ai-backend ai-backend
}

/var/log/ai-worker.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 ai-worker ai-worker
}
EOF
```

### 2. Health Checks

```bash
# Create health check script
cat > /opt/ai-job-queue/health-check.sh << 'EOF'
#!/bin/bash

# Check backend
if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "Backend health check failed"
    exit 1
fi

# Check Redis
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Redis health check failed"
    exit 1
fi

echo "All services healthy"
EOF

chmod +x /opt/ai-job-queue/health-check.sh
```

## 🚀 Production Checklist

- [ ] Redis configured with authentication
- [ ] Services running as dedicated users
- [ ] Firewall configured
- [ ] SSL/TLS certificates installed
- [ ] Log rotation configured
- [ ] Monitoring and alerting setup
- [ ] Backup strategy implemented
- [ ] Load balancer configured (if needed)
- [ ] Auto-scaling policies (if using cloud)
- [ ] Disaster recovery plan

## 🔄 CI/CD Pipeline

### GitHub Actions Example

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to server
        uses: appleboy/ssh-action@v0.1.5
        with:
          host: ${{ secrets.HOST }}
          username: ${{ secrets.USERNAME }}
          key: ${{ secrets.KEY }}
          script: |
            cd /opt/ai-job-queue
            git pull origin main
            cd backend && uv sync
            cd ../ai-worker && uv sync
            sudo systemctl restart ai-backend
            sudo systemctl restart ai-worker
```
