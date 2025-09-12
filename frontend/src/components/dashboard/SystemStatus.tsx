import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';
import { api } from '../../lib/api';
import type { HealthStatus, WorkerStatus, FrameworkInfo } from '../../types';

export function SystemStatus() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [workers, setWorkers] = useState<WorkerStatus | null>(null);
  const [frameworks, setFrameworks] = useState<FrameworkInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadStatus = async () => {
    try {
      setError(null);
      const [healthData, workersData, frameworksData] = await Promise.all([
        api.getHealth(),
        api.getWorkerStatus(),
        api.getFrameworks(),
      ]);
      
      setHealth(healthData);
      setWorkers(workersData);
      setFrameworks(frameworksData);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to load system status');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStatus();
    // Refresh every 30 seconds
    const interval = setInterval(loadStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
            <span className="ml-2 text-gray-600">Loading system status...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="text-center">
            <div className="text-error-600 mb-4">{error}</div>
            <Button onClick={loadStatus} variant="outline">
              Try Again
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {/* Health Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${health?.status === 'healthy' ? 'bg-success-500' : 'bg-error-500'}`}></div>
            System Health
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">API Status:</span>
              <Badge variant={health?.status === 'healthy' ? 'success' : 'error'}>
                {health?.status || 'Unknown'}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Redis:</span>
              <Badge variant={health?.redis === 'connected' ? 'success' : 'error'}>
                {health?.redis || 'Unknown'}
              </Badge>
            </div>
            {health?.version && (
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Version:</span>
                <span className="text-sm text-gray-900">{health.version}</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Worker Status */}
      <Card>
        <CardHeader>
          <CardTitle>Worker Status</CardTitle>
          <CardDescription>
            {workers?.total_workers || 0} total workers
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Available:</span>
              <span className="text-sm font-medium text-gray-900">
                {workers?.available_workers || 0}
              </span>
            </div>
            {workers?.workers_by_type && Object.keys(workers.workers_by_type).length > 0 && (
              <div className="space-y-2">
                <span className="text-sm font-medium text-gray-700">By Type:</span>
                <div className="space-y-1">
                  {Object.entries(workers.workers_by_type).map(([type, count]) => (
                    <div key={type} className="flex justify-between text-sm">
                      <span className="text-gray-600 capitalize">
                        {type.replace(/-/g, ' ')}
                      </span>
                      <span className="text-gray-900">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Frameworks */}
      <Card>
        <CardHeader>
          <CardTitle>Available Frameworks</CardTitle>
          <CardDescription>
            {frameworks?.frameworks?.length || 0} frameworks
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {frameworks?.frameworks?.map((framework) => (
              <div key={framework} className="flex items-center justify-between">
                <span className="text-sm text-gray-600 capitalize">
                  {framework.replace(/-/g, ' ')}
                </span>
                <Badge variant="secondary" className="text-xs">
                  Available
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
