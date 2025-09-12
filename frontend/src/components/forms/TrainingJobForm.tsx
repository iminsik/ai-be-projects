import React, { useState, useEffect } from 'react';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { Select } from '../ui/Select';
import { Textarea } from '../ui/Textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/Card';
import { api } from '../../lib/api';
import type { TrainingJobRequest, FrameworkInfo } from '../../types';

interface TrainingJobFormProps {
  onJobSubmitted: (jobId: string) => void;
}

const modelTypes = [
  { value: 'bert', label: 'BERT (Transformer)' },
  { value: 'gpt', label: 'GPT (Transformer)' },
  { value: 'resnet', label: 'ResNet (CNN)' },
  { value: 'vgg', label: 'VGG (CNN)' },
  { value: 'inception', label: 'Inception (CNN)' },
  { value: 'mobilenet', label: 'MobileNet (CNN)' },
  { value: 'random_forest', label: 'Random Forest' },
  { value: 'svm', label: 'SVM' },
  { value: 'logistic_regression', label: 'Logistic Regression' },
];

const defaultHyperparameters = {
  bert: { epochs: 5, learning_rate: 0.0001, batch_size: 16, max_length: 512 },
  gpt: { epochs: 3, learning_rate: 0.0003, batch_size: 8, max_length: 1024 },
  resnet: { epochs: 10, learning_rate: 0.001, batch_size: 32, num_classes: 1000 },
  vgg: { epochs: 10, learning_rate: 0.001, batch_size: 32, num_classes: 1000 },
  inception: { epochs: 20, learning_rate: 0.01, batch_size: 64, dropout_rate: 0.2 },
  mobilenet: { epochs: 20, learning_rate: 0.01, batch_size: 64, dropout_rate: 0.2 },
  random_forest: { n_estimators: 100, max_depth: 10, min_samples_split: 5, random_state: 42 },
  svm: { C: 1.0, kernel: 'rbf', gamma: 'scale' },
  logistic_regression: { C: 1.0, max_iter: 1000, random_state: 42 },
};

export function TrainingJobForm({ onJobSubmitted }: TrainingJobFormProps) {
  const [formData, setFormData] = useState<TrainingJobRequest>({
    model_type: 'bert',
    data_path: '',
    hyperparameters: defaultHyperparameters.bert,
    description: '',
    requires_gpu: false,
    framework_override: '',
  });
  const [frameworks, setFrameworks] = useState<FrameworkInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadFrameworks();
  }, []);

  useEffect(() => {
    // Update hyperparameters when model type changes
    const newHyperparams = defaultHyperparameters[formData.model_type as keyof typeof defaultHyperparameters];
    if (newHyperparams) {
      setFormData(prev => ({
        ...prev,
        hyperparameters: newHyperparams
      }));
    }
  }, [formData.model_type]);

  const loadFrameworks = async () => {
    try {
      const frameworkInfo = await api.getFrameworks();
      setFrameworks(frameworkInfo);
    } catch (err) {
      console.error('Failed to load frameworks:', err);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const job = await api.submitTrainingJob(formData);
      onJobSubmitted(job.job_id);
      
      // Reset form
      setFormData({
        model_type: 'bert',
        data_path: '',
        hyperparameters: defaultHyperparameters.bert,
        description: '',
        requires_gpu: false,
        framework_override: '',
      });
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to submit training job');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field: keyof TrainingJobRequest, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleHyperparameterChange = (key: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      hyperparameters: {
        ...prev.hyperparameters,
        [key]: value
      }
    }));
  };

  const frameworkOptions = frameworks ? 
    Object.entries(frameworks.queues).map(([key, value]) => ({
      value: key,
      label: key.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    })) : [];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Submit Training Job</CardTitle>
        <CardDescription>
          Train a machine learning model using various frameworks
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          {error && (
            <div className="rounded-md bg-error-50 p-4">
              <div className="text-sm text-error-700">{error}</div>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Select
              label="Model Type"
              value={formData.model_type}
              onChange={(e) => handleInputChange('model_type', e.target.value)}
              options={modelTypes}
              required
            />

            <Input
              label="Data Path"
              placeholder="/data/training.csv"
              value={formData.data_path}
              onChange={(e) => handleInputChange('data_path', e.target.value)}
              required
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium text-gray-700">GPU Required</label>
              <div className="mt-1">
                <label className="inline-flex items-center">
                  <input
                    type="checkbox"
                    checked={formData.requires_gpu}
                    onChange={(e) => handleInputChange('requires_gpu', e.target.checked)}
                    className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="ml-2 text-sm text-gray-700">Use GPU acceleration</span>
                </label>
              </div>
            </div>

            <Select
              label="Framework Override (Optional)"
              value={formData.framework_override || ''}
              onChange={(e) => handleInputChange('framework_override', e.target.value || undefined)}
              options={[{ value: '', label: 'Auto-select' }, ...frameworkOptions]}
            />
          </div>

          <Textarea
            label="Description"
            placeholder="Describe your training job..."
            value={formData.description || ''}
            onChange={(e) => handleInputChange('description', e.target.value)}
            rows={3}
          />

          <div>
            <label className="text-sm font-medium text-gray-700 mb-2 block">
              Hyperparameters
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4 bg-gray-50 rounded-md">
              {Object.entries(formData.hyperparameters || {}).map(([key, value]) => (
                <Input
                  key={key}
                  label={key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  value={value}
                  onChange={(e) => {
                    const newValue = isNaN(Number(e.target.value)) 
                      ? e.target.value 
                      : Number(e.target.value);
                    handleHyperparameterChange(key, newValue);
                  }}
                  type={typeof value === 'number' ? 'number' : 'text'}
                />
              ))}
            </div>
          </div>

          <div className="flex justify-end">
            <Button type="submit" loading={loading} disabled={loading}>
              Submit Training Job
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
