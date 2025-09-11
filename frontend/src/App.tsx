import React, { useState, useEffect } from 'react';
import { TrainingJobForm } from './components/forms/TrainingJobForm';
import { InferenceJobForm } from './components/forms/InferenceJobForm';
import { JobList } from './components/jobs/JobList';
import { SystemStatus } from './components/dashboard/SystemStatus';
import { Button } from './components/ui/Button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/Card';
import { apiClient } from './lib/api';
import type { JobStatus } from './types';

function App() {
  const [activeTab, setActiveTab] = useState<'training' | 'inference' | 'jobs' | 'status'>('training');
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [availableModels, setAvailableModels] = useState<string[]>([]);

  // Load available models from completed training jobs
  useEffect(() => {
    const loadAvailableModels = async () => {
      try {
        const jobs = await apiClient.listJobs(100);
        const completedTrainingJobs = jobs.filter(
          job => job.job_type === 'training' && 
                 job.status === 'completed' && 
                 job.result?.model_id
        );
        const modelIds = completedTrainingJobs.map(job => job.result!.model_id);
        setAvailableModels([...new Set(modelIds)]);
      } catch (err) {
        console.error('Failed to load available models:', err);
      }
    };

    loadAvailableModels();
  }, [refreshTrigger]);

  const handleJobSubmitted = (jobId: string) => {
    console.log('Job submitted:', jobId);
    setRefreshTrigger(prev => prev + 1);
    // Switch to jobs tab to show the new job
    setActiveTab('jobs');
  };

  const tabs = [
    { id: 'training', label: 'Training', icon: '🧠' },
    { id: 'inference', label: 'Inference', icon: '🔮' },
    { id: 'jobs', label: 'Jobs', icon: '📋' },
    { id: 'status', label: 'Status', icon: '📊' },
  ] as const;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">
                AI Job Queue System
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-500">
                Backend: localhost:8000
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        {activeTab === 'training' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-3xl font-bold text-gray-900">Training Jobs</h2>
              <p className="mt-2 text-gray-600">
                Submit machine learning training jobs using various frameworks
              </p>
            </div>
            <TrainingJobForm onJobSubmitted={handleJobSubmitted} />
          </div>
        )}

        {activeTab === 'inference' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-3xl font-bold text-gray-900">Inference Jobs</h2>
              <p className="mt-2 text-gray-600">
                Run inference using trained models
              </p>
            </div>
            <InferenceJobForm 
              onJobSubmitted={handleJobSubmitted} 
              availableModels={availableModels}
            />
          </div>
        )}

        {activeTab === 'jobs' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-3xl font-bold text-gray-900">Job History</h2>
                <p className="mt-2 text-gray-600">
                  Monitor and manage your training and inference jobs
                </p>
              </div>
              <Button 
                onClick={() => setRefreshTrigger(prev => prev + 1)}
                variant="outline"
              >
                Refresh
              </Button>
            </div>
            <JobList refreshTrigger={refreshTrigger} />
          </div>
        )}

        {activeTab === 'status' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-3xl font-bold text-gray-900">System Status</h2>
              <p className="mt-2 text-gray-600">
                Monitor system health, workers, and available frameworks
              </p>
            </div>
            <SystemStatus />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="text-center text-sm text-gray-500">
            <p>AI Job Queue System - Microservices Architecture</p>
            <p className="mt-1">
              FastAPI Backend + Redis + AI Workers (PyTorch, TensorFlow, Scikit-learn)
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
