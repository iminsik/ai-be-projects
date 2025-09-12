# API Documentation

This document provides comprehensive API reference for the AI Job Queue System.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

## Authentication

Currently, the API does not require authentication. In production, consider implementing:
- API keys
- JWT tokens
- OAuth 2.0

## Endpoints

### Health Check

#### `GET /health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

### Training Jobs

#### `POST /jobs/training`

Submit a new training job to the appropriate framework-specific queue.

**Request Body:**
```json
{
  "model_type": "string",
  "data_path": "string",
  "hyperparameters": {
    "additionalProp": {}
  },
  "description": "string",
  "requires_gpu": false,
  "framework_override": "string"
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_type` | string | Yes | Type of model to train (bert, resnet, inception, random_forest, etc.) |
| `data_path` | string | Yes | Path to training data (file or directory) |
| `hyperparameters` | object | No | Model-specific training parameters |
| `description` | string | No | Human-readable description of the job |
| `requires_gpu` | boolean | No | Whether GPU acceleration is needed (default: false) |
| `framework_override` | string | No | Force specific framework/version |

**Response:**
```json
{
  "job_id": "job_12345678-1234-1234-1234-123456789012",
  "status": "queued",
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_duration": "00:15:00"
}
```

**Training Job Examples:**

##### PyTorch BERT Model (CPU)
```bash
curl -X POST "http://localhost:8000/jobs/training" \
     -H "Content-Type: application/json" \
     -d '{
       "model_type": "bert",
       "data_path": "/data/sentiment_analysis.csv",
       "hyperparameters": {
         "epochs": 5,
         "learning_rate": 0.0001,
         "batch_size": 16,
         "max_length": 512
       },
       "description": "Fine-tune BERT for sentiment analysis",
       "requires_gpu": false,
       "framework_override": "pytorch-2.1"
     }'
```

##### PyTorch ResNet Model (GPU)
```bash
curl -X POST "http://localhost:8000/jobs/training" \
     -H "Content-Type: application/json" \
     -d '{
       "model_type": "resnet",
       "data_path": "/data/image_classification",
       "hyperparameters": {
         "epochs": 10,
         "learning_rate": 0.001,
         "batch_size": 32,
         "num_classes": 1000
       },
       "description": "Train ResNet-50 for image classification",
       "requires_gpu": true,
       "framework_override": "pytorch-2.0-gpu"
     }'
```

##### TensorFlow Inception Model
```bash
curl -X POST "http://localhost:8000/jobs/training" \
     -H "Content-Type: application/json" \
     -d '{
       "model_type": "inception",
       "data_path": "/data/cifar10",
       "hyperparameters": {
         "epochs": 20,
         "learning_rate": 0.01,
         "batch_size": 64,
         "dropout_rate": 0.2
       },
       "description": "Train Inception v3 on CIFAR-10",
       "requires_gpu": false,
       "framework_override": "tensorflow"
     }'
```

##### Scikit-learn Random Forest
```bash
curl -X POST "http://localhost:8000/jobs/training" \
     -H "Content-Type: application/json" \
     -d '{
       "model_type": "random_forest",
       "data_path": "/data/tabular_data.csv",
       "hyperparameters": {
         "n_estimators": 100,
         "max_depth": 10,
         "min_samples_split": 5,
         "random_state": 42
       },
       "description": "Train Random Forest for classification",
       "requires_gpu": false,
       "framework_override": "sklearn"
     }'
```

##### PyTorch Transformer (GPU)
```bash
curl -X POST "http://localhost:8000/jobs/training" \
     -H "Content-Type: application/json" \
     -d '{
       "model_type": "transformer",
       "data_path": "/data/text_generation",
       "hyperparameters": {
         "epochs": 3,
         "learning_rate": 0.0003,
         "batch_size": 8,
         "max_length": 1024,
         "num_layers": 12,
         "hidden_size": 768
       },
       "description": "Train GPT-style transformer for text generation",
       "requires_gpu": true,
       "framework_override": "pytorch-2.1-gpu"
     }'
```

### Inference Jobs

#### `POST /jobs/inference`

Submit an inference job using a trained model.

**Request Body:**
```json
{
  "model_id": "string",
  "input_data": "string",
  "parameters": {
    "additionalProp": {}
  }
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | Yes | ID of the trained model to use |
| `input_data` | string | Yes | Input data for inference |
| `parameters` | object | No | Inference parameters (temperature, max_length, etc.) |

**Response:**
```json
{
  "job_id": "job_87654321-4321-4321-4321-210987654321",
  "status": "queued",
  "created_at": "2024-01-15T10:35:00Z",
  "estimated_duration": "00:01:00"
}
```

**Inference Job Examples:**

##### Text Classification
```bash
curl -X POST "http://localhost:8000/jobs/inference" \
     -H "Content-Type: application/json" \
     -d '{
       "model_id": "bert_model_123",
       "input_data": "This is a great product!",
       "parameters": {
         "temperature": 0.7,
         "max_length": 100
       }
     }'
```

##### Image Classification
```bash
curl -X POST "http://localhost:8000/jobs/inference" \
     -H "Content-Type: application/json" \
     -d '{
       "model_id": "resnet_model_456",
       "input_data": "/path/to/image.jpg",
       "parameters": {
         "top_k": 5,
         "confidence_threshold": 0.8
       }
     }'
```

##### Tabular Data Prediction
```bash
curl -X POST "http://localhost:8000/jobs/inference" \
     -H "Content-Type: application/json" \
     -d '{
       "model_id": "random_forest_model_789",
       "input_data": "feature1,feature2,feature3,feature4",
       "parameters": {
         "probability": true
       }
     }'
```

### Job Status

#### `GET /jobs/{job_id}/status`

Get the current status of a job.

**Path Parameters:**
- `job_id` (string): The unique identifier of the job

**Response:**
```json
{
  "job_id": "job_12345678-1234-1234-1234-123456789012",
  "status": "running",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:32:00Z",
  "progress": 0.6,
  "result": {
    "model_id": "pytorch_model_123",
    "model_path": "/models/pytorch_model_123",
    "framework": "pytorch",
    "metrics": {
      "accuracy": 0.95,
      "training_time": 900
    }
  }
}
```

**Status Values:**
- `pending`: Job is waiting to be processed
- `running`: Job is currently being processed
- `completed`: Job completed successfully
- `failed`: Job failed with an error
- `cancelled`: Job was cancelled by user

**Example:**
```bash
curl "http://localhost:8000/jobs/job_12345678-1234-1234-1234-123456789012/status"
```

### Cancel Job

#### `DELETE /jobs/{job_id}`

Cancel a running or pending job.

**Path Parameters:**
- `job_id` (string): The unique identifier of the job to cancel

**Response:**
```json
{
  "job_id": "job_12345678-1234-1234-1234-123456789012",
  "status": "cancelled",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:35:00Z",
  "error": "Job cancelled by user"
}
```

**Error Responses:**
- `404`: Job not found
- `400`: Job cannot be cancelled (already completed, failed, or cancelled)

**Example:**
```bash
curl -X DELETE "http://localhost:8000/jobs/job_12345678-1234-1234-1234-123456789012"
```

### List Jobs

#### `GET /jobs`

List all jobs with optional filtering.

**Query Parameters:**
- `status` (string, optional): Filter by job status
- `limit` (integer, optional): Maximum number of jobs to return (default: 100)
- `offset` (integer, optional): Number of jobs to skip (default: 0)

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "job_12345678-1234-1234-1234-123456789012",
      "job_type": "training",
      "status": "completed",
      "created_at": "2024-01-15T10:30:00Z",
      "model_type": "bert"
    },
    {
      "job_id": "job_87654321-4321-4321-4321-210987654321",
      "job_type": "inference",
      "status": "running",
      "created_at": "2024-01-15T10:35:00Z",
      "model_id": "bert_model_123"
    }
  ],
  "total": 2,
  "limit": 100,
  "offset": 0
}
```

**Examples:**
```bash
# List all jobs
curl "http://localhost:8000/jobs"

# List only completed jobs
curl "http://localhost:8000/jobs?status=completed"

# List with pagination
curl "http://localhost:8000/jobs?limit=10&offset=20"
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "status_code": 400
}
```

**Common HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (job not found)
- `422`: Unprocessable Entity (validation error)
- `500`: Internal Server Error

**Example Error Response:**
```json
{
  "error": "Validation error",
  "detail": "model_type is required",
  "status_code": 422
}
```

## Framework Routing

The backend automatically routes jobs to the appropriate worker based on the following logic:

1. **Framework Override**: If `framework_override` is specified, use that worker type
2. **Model Type + GPU**: Route based on model type and GPU requirements
3. **Default**: Fall back to PyTorch 2.1 CPU worker

**Routing Table:**

| Model Type | GPU Required | Target Worker |
|------------|--------------|---------------|
| `bert`, `transformer`, `gpt` | `false` | PyTorch 2.1 CPU |
| `bert`, `transformer`, `gpt` | `true` | PyTorch 2.1 GPU |
| `resnet`, `vgg`, `alexnet` | `false` | PyTorch 2.0 CPU |
| `resnet`, `vgg`, `alexnet` | `true` | PyTorch 2.0 GPU |
| `inception`, `mobilenet`, `efficientnet` | `false` | TensorFlow |
| `inception`, `mobilenet`, `efficientnet` | `true` | TensorFlow |
| `random_forest`, `svm`, `logistic_regression` | `false` | Scikit-learn |
| `random_forest`, `svm`, `logistic_regression` | `true` | Scikit-learn |

## Rate Limiting

Currently, there are no rate limits implemented. In production, consider implementing:
- Per-IP rate limiting
- Per-user rate limiting
- Queue size limits

## Monitoring

### Health Checks

The system provides health check endpoints for monitoring:

```bash
# API health
curl http://localhost:8000/health

# Redis health (internal)
curl http://localhost:8000/health/redis
```

### Metrics

Consider implementing metrics collection for:
- Job completion rates
- Average processing times
- Queue lengths
- Worker utilization
- Error rates

## SDK Examples

### Python SDK Example

```python
import requests
import time

class AIJobClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def submit_training_job(self, model_type, data_path, **kwargs):
        payload = {
            "model_type": model_type,
            "data_path": data_path,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/jobs/training", json=payload)
        return response.json()
    
    def submit_inference_job(self, model_id, input_data, **kwargs):
        payload = {
            "model_id": model_id,
            "input_data": input_data,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/jobs/inference", json=payload)
        return response.json()
    
    def get_job_status(self, job_id):
        response = requests.get(f"{self.base_url}/jobs/{job_id}/status")
        return response.json()
    
    def wait_for_completion(self, job_id, timeout=3600):
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)
            if status["status"] in ["completed", "failed"]:
                return status
            time.sleep(5)
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

# Usage
client = AIJobClient()

# Submit training job
job = client.submit_training_job(
    model_type="bert",
    data_path="/data/training.csv",
    hyperparameters={"epochs": 5, "batch_size": 16},
    description="BERT training"
)

# Wait for completion
result = client.wait_for_completion(job["job_id"])
print(f"Training completed: {result}")
```

### JavaScript SDK Example

```javascript
class AIJobClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async submitTrainingJob(modelType, dataPath, options = {}) {
        const payload = {
            model_type: modelType,
            data_path: dataPath,
            ...options
        };
        
        const response = await fetch(`${this.baseUrl}/jobs/training`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });
        
        return await response.json();
    }
    
    async submitInferenceJob(modelId, inputData, options = {}) {
        const payload = {
            model_id: modelId,
            input_data: inputData,
            ...options
        };
        
        const response = await fetch(`${this.baseUrl}/jobs/inference`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });
        
        return await response.json();
    }
    
    async getJobStatus(jobId) {
        const response = await fetch(`${this.baseUrl}/jobs/${jobId}/status`);
        return await response.json();
    }
    
    async waitForCompletion(jobId, timeout = 3600000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < timeout) {
            const status = await this.getJobStatus(jobId);
            if (status.status === 'completed' || status.status === 'failed') {
                return status;
            }
            await new Promise(resolve => setTimeout(resolve, 5000));
        }
        
        throw new Error(`Job ${jobId} did not complete within ${timeout}ms`);
    }
}

// Usage
const client = new AIJobClient();

// Submit training job
const job = await client.submitTrainingJob(
    'bert',
    '/data/training.csv',
    {
        hyperparameters: { epochs: 5, batch_size: 16 },
        description: 'BERT training'
    }
);

// Wait for completion
const result = await client.waitForCompletion(job.job_id);
console.log('Training completed:', result);
```

## Testing

Use the provided test script to verify the API:

```bash
python scripts/test_api.py
```

This script will:
1. Test the health endpoint
2. Submit a training job
3. Monitor job progress
4. Submit an inference job
5. Test multiple frameworks
6. List all jobs
