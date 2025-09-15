import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation, useNavigate } from 'react-router-dom';
import { TrainingJobForm } from './components/forms/TrainingJobForm';
import { InferenceJobForm } from './components/forms/InferenceJobForm';
import { JobList } from './components/jobs/JobList';
import { SystemStatus } from './components/dashboard/SystemStatus';
import { Button } from './components/ui/Button';
import { api } from './lib/api';
import type { JobStatus, ModelInfo } from './types';

// Context for sharing state between layout and pages
const AppContext = React.createContext<{
  refreshTrigger: number;
  setRefreshTrigger: (fn: (prev: number) => number) => void;
  availableModels: string[];
  onJobSubmitted: (jobId: string) => void;
  onJobCancelled: (jobId: string) => void;
}>({
  refreshTrigger: 0,
  setRefreshTrigger: () => {},
  availableModels: [],
  onJobSubmitted: () => {},
  onJobCancelled: () => {},
});

// Layout component with navigation
function Layout({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const navigate = useNavigate();
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);

  // Load available models from the new models endpoint
  useEffect(() => {
    const loadAvailableModels = async () => {
      try {
        const modelsData = await api.getModels();
        setAvailableModels(modelsData.models);
      } catch (err) {
        console.error('Failed to load available models:', err);
      }
    };

    loadAvailableModels();
  }, [refreshTrigger]);

  const handleJobSubmitted = (jobId: string) => {
    console.log('Job submitted:', jobId);
    setRefreshTrigger(prev => prev + 1);
    // Navigate to jobs page to show the new job
    navigate('/jobs');
  };

  const handleJobCancelled = (jobId: string) => {
    console.log('Job cancelled:', jobId);
    setRefreshTrigger(prev => prev + 1);
  };

  const tabs = [
    { id: 'training', label: 'Training', icon: '🧠', path: '/training' },
    { id: 'inference', label: 'Inference', icon: '🔮', path: '/inference' },
    { id: 'jobs', label: 'Jobs', icon: '📋', path: '/jobs' },
    { id: 'status', label: 'Status', icon: '📊', path: '/status' },
  ] as const;

  const contextValue = {
    refreshTrigger,
    setRefreshTrigger,
    availableModels,
    onJobSubmitted: handleJobSubmitted,
    onJobCancelled: handleJobCancelled,
  };

  return (
    <AppContext.Provider value={contextValue}>
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center">
                <Link to="/" className="text-2xl font-bold text-gray-900">
                  AI Job Queue System
                </Link>
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
                <Link
                  key={tab.id}
                  to={tab.path}
                  className={`py-4 px-1 border-b-2 font-medium text-sm ${
                    location.pathname === tab.path
                      ? 'border-primary-500 text-primary-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <span className="mr-2">{tab.icon}</span>
                  {tab.label}
                </Link>
              ))}
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          {children}
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
    </AppContext.Provider>
  );
}

// Page components
function TrainingPage() {
  const { onJobSubmitted } = React.useContext(AppContext);
  
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-gray-900">Training Jobs</h2>
        <p className="mt-2 text-gray-600">
          Submit machine learning training jobs using various frameworks
        </p>
      </div>
      <TrainingJobForm onJobSubmitted={onJobSubmitted} />
    </div>
  );
}

function InferencePage() {
  const { onJobSubmitted, availableModels } = React.useContext(AppContext);
  
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-gray-900">Inference Jobs</h2>
        <p className="mt-2 text-gray-600">
          Run inference using trained models
        </p>
      </div>
      <InferenceJobForm 
        onJobSubmitted={onJobSubmitted} 
        availableModels={availableModels}
      />
    </div>
  );
}

function JobsPage() {
  const { refreshTrigger, setRefreshTrigger, onJobCancelled } = React.useContext(AppContext);
  
  return (
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
      <JobList refreshTrigger={refreshTrigger} onJobCancelled={onJobCancelled} />
    </div>
  );
}

function StatusPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-gray-900">System Status</h2>
        <p className="mt-2 text-gray-600">
          Monitor system health, workers, and available frameworks
        </p>
      </div>
      <SystemStatus />
    </div>
  );
}

function HomePage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-gray-900">Welcome to AI Job Queue System</h2>
        <p className="mt-2 text-gray-600">
          Submit training and inference jobs using various machine learning frameworks
        </p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">🧠 Training Jobs</h3>
          <p className="text-gray-600 mb-4">Train machine learning models using PyTorch, TensorFlow, or Scikit-learn</p>
          <Link to="/training" className="text-primary-600 hover:text-primary-700 font-medium">
            Start Training →
          </Link>
        </div>
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">🔮 Inference Jobs</h3>
          <p className="text-gray-600 mb-4">Run inference using your trained models</p>
          <Link to="/inference" className="text-primary-600 hover:text-primary-700 font-medium">
            Run Inference →
          </Link>
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/inference" element={<InferencePage />} />
          <Route path="/jobs" element={<JobsPage />} />
          <Route path="/status" element={<StatusPage />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
