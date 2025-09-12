import React, { useState } from 'react';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { Textarea } from '../ui/Textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/Card';
import { apiClient } from '../../lib/api';
import type { InferenceJobRequest } from '../../types';

interface InferenceJobFormProps {
  onJobSubmitted: (jobId: string) => void;
  availableModels: string[];
}

export function InferenceJobForm({ onJobSubmitted, availableModels }: InferenceJobFormProps) {
  const [formData, setFormData] = useState<InferenceJobRequest>({
    model_id: '',
    input_data: '',
    parameters: {},
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const job = await apiClient.submitInferenceJob(formData);
      onJobSubmitted(job.job_id);
      
      // Reset form
      setFormData({
        model_id: '',
        input_data: '',
        parameters: {},
      });
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to submit inference job');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field: keyof InferenceJobRequest, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleParameterChange = (key: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        [key]: value
      }
    }));
  };

  const modelOptions = availableModels.length > 0 
    ? availableModels.map(model => ({ value: model, label: model }))
    : [{ value: '', label: 'No trained models available' }];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Submit Inference Job</CardTitle>
        <CardDescription>
          Run inference using a trained model
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
            <div>
              <label className="text-sm font-medium text-gray-700">Model ID</label>
              <select
                value={formData.model_id}
                onChange={(e) => handleInputChange('model_id', e.target.value)}
                className="mt-1 flex h-10 w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                required
              >
                <option value="">Select a model...</option>
                {modelOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            <Input
              label="Input Data"
              placeholder="Enter input data for inference..."
              value={formData.input_data}
              onChange={(e) => handleInputChange('input_data', e.target.value)}
              required
            />
          </div>

          <div>
            <label className="text-sm font-medium text-gray-700 mb-2 block">
              Inference Parameters (Optional)
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 bg-gray-50 rounded-md">
              <Input
                label="Temperature"
                type="number"
                step="0.1"
                min="0"
                max="2"
                value={formData.parameters?.temperature || ''}
                onChange={(e) => handleParameterChange('temperature', e.target.value ? Number(e.target.value) : undefined)}
                placeholder="0.7"
              />
              <Input
                label="Max Length"
                type="number"
                min="1"
                value={formData.parameters?.max_length || ''}
                onChange={(e) => handleParameterChange('max_length', e.target.value ? Number(e.target.value) : undefined)}
                placeholder="100"
              />
              <Input
                label="Top K"
                type="number"
                min="1"
                value={formData.parameters?.top_k || ''}
                onChange={(e) => handleParameterChange('top_k', e.target.value ? Number(e.target.value) : undefined)}
                placeholder="5"
              />
              <Input
                label="Confidence Threshold"
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={formData.parameters?.confidence_threshold || ''}
                onChange={(e) => handleParameterChange('confidence_threshold', e.target.value ? Number(e.target.value) : undefined)}
                placeholder="0.8"
              />
            </div>
          </div>

          <div className="flex justify-end">
            <Button 
              type="submit" 
              loading={loading} 
              disabled={loading || availableModels.length === 0}
            >
              Submit Inference Job
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
