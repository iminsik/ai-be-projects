export interface TrainingJobRequest {
  model_type: string;
  data_path: string;
  hyperparameters?: Record<string, any>;
  description?: string;
  requires_gpu?: boolean;
  framework_override?: string;
}

export interface InferenceJobRequest {
  model_id: string;
  input_data: string;
  parameters?: Record<string, any>;
}

export interface JobStatus {
  job_id: string;
  job_type: 'training' | 'inference';
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  created_at: string;
  updated_at: string;
  result?: Record<string, any>;
  error?: string;
  metadata?: Record<string, any>;
  framework?: string;
  worker_type?: string;
}

export interface FrameworkInfo {
  frameworks: string[];
  model_registry: Record<string, {
    framework: string;
    version: string;
    queue: string;
  }>;
  queues: Record<string, string>;
}

export interface WorkerStatus {
  workers_by_type: Record<string, number>;
  total_workers: number;
  available_workers: number;
}

export interface HealthStatus {
  status: string;
  redis?: string;
  timestamp?: string;
  version?: string;
}
