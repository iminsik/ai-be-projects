import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';
import { formatDate, getStatusColor, getStatusIcon, truncateText } from '../../lib/utils';
import { apiClient } from '../../lib/api';
import type { JobStatus } from '../../types';

interface JobStatusCardProps {
  job: JobStatus;
  onRefresh?: () => void;
  onJobCancelled?: (jobId: string) => void;
}

export function JobStatusCard({ job, onRefresh, onJobCancelled }: JobStatusCardProps) {
  const [cancelling, setCancelling] = useState(false);
  const statusColor = getStatusColor(job.status);
  const statusIcon = getStatusIcon(job.status);

  const canCancel = job.status === 'pending' || job.status === 'running';

  const handleCancel = async () => {
    if (!canCancel) return;
    
    setCancelling(true);
    try {
      await apiClient.cancelJob(job.job_id);
      onJobCancelled?.(job.job_id);
      onRefresh?.();
    } catch (error) {
      console.error('Failed to cancel job:', error);
    } finally {
      setCancelling(false);
    }
  };

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg">
              {job.job_type === 'training' ? 'Training Job' : 'Inference Job'}
            </CardTitle>
            <CardDescription className="text-sm">
              {job.job_id}
            </CardDescription>
          </div>
          <Badge 
            variant={job.status === 'completed' ? 'success' : 
                   job.status === 'failed' ? 'error' : 
                   job.status === 'running' ? 'warning' : 'default'}
            className="flex items-center gap-1"
          >
            <span>{statusIcon}</span>
            <span className="capitalize">{job.status}</span>
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="font-medium text-gray-600">Created:</span>
            <p className="text-gray-900">{formatDate(job.created_at)}</p>
          </div>
          <div>
            <span className="font-medium text-gray-600">Updated:</span>
            <p className="text-gray-900">{formatDate(job.updated_at)}</p>
          </div>
        </div>

        {job.framework && (
          <div>
            <span className="font-medium text-gray-600">Framework:</span>
            <p className="text-gray-900 capitalize">{job.framework}</p>
          </div>
        )}

        {job.worker_type && (
          <div>
            <span className="font-medium text-gray-600">Worker Type:</span>
            <p className="text-gray-900">{job.worker_type}</p>
          </div>
        )}

        {job.metadata && (
          <div>
            <span className="font-medium text-gray-600">Details:</span>
            <div className="mt-1 space-y-1">
              {job.metadata.model_id && (
                <p className="text-sm text-gray-700">
                  <span className="font-medium">Model ID:</span> {job.metadata.model_id}
                </p>
              )}
              {job.metadata.model_type && (
                <p className="text-sm text-gray-700">
                  <span className="font-medium">Model Type:</span> {job.metadata.model_type}
                </p>
              )}
              {job.metadata.input_data && (
                <p className="text-sm text-gray-700">
                  <span className="font-medium">Input Data:</span> {job.metadata.input_data}
                </p>
              )}
              {job.metadata.data_path && (
                <p className="text-sm text-gray-700">
                  <span className="font-medium">Data Path:</span> {truncateText(job.metadata.data_path, 50)}
                </p>
              )}
              {job.metadata.model_framework && (
                <p className="text-sm text-gray-700">
                  <span className="font-medium">Model Framework:</span> {job.metadata.model_framework}
                </p>
              )}
              {job.metadata.description && (
                <p className="text-sm text-gray-700">
                  <span className="font-medium">Description:</span> {truncateText(job.metadata.description, 100)}
                </p>
              )}
              {job.metadata.parameters && Object.keys(job.metadata.parameters).length > 0 && (
                <div className="mt-2">
                  <p className="text-sm font-medium text-gray-600">Parameters:</p>
                  <div className="grid grid-cols-2 gap-2 mt-1">
                    {Object.entries(job.metadata.parameters).map(([key, value]) => (
                      <p key={key} className="text-sm text-gray-700">
                        <span className="font-medium">{key}:</span> {typeof value === 'number' ? value.toFixed(4) : String(value)}
                      </p>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {job.error && (
          <div className="rounded-md bg-error-50 p-3">
            <p className="text-sm font-medium text-error-800">Error:</p>
            <p className="text-sm text-error-700 mt-1">{job.error}</p>
          </div>
        )}

        {job.result && job.status === 'completed' && (
          <div className="rounded-md bg-success-50 p-3">
            <p className="text-sm font-medium text-success-800">Results:</p>
            <div className="mt-2 space-y-1">
              {job.result.model_id && (
                <p className="text-sm text-success-700">
                  <span className="font-medium">Model ID:</span> {job.result.model_id}
                </p>
              )}
              {job.result.model_path && (
                <p className="text-sm text-success-700">
                  <span className="font-medium">Model Path:</span> {truncateText(job.result.model_path, 50)}
                </p>
              )}
              {job.result.metrics && (
                <div className="mt-2">
                  <p className="text-sm font-medium text-success-800">Metrics:</p>
                  <div className="grid grid-cols-2 gap-2 mt-1">
                    {Object.entries(job.result.metrics).map(([key, value]) => (
                      <p key={key} className="text-sm text-success-700">
                        <span className="font-medium">{key}:</span> {typeof value === 'number' ? value.toFixed(4) : value}
                      </p>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <div className="pt-2 border-t flex justify-between items-center">
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="text-sm text-primary-600 hover:text-primary-700 font-medium"
            >
              Refresh Status
            </button>
          )}
          
          {canCancel && (
            <Button
              onClick={handleCancel}
              loading={cancelling}
              variant="destructive"
              size="sm"
            >
              Cancel Job
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
