import React, { useState, useEffect } from 'react';
import { JobStatusCard } from './JobStatusCard';
import { Button } from '../ui/Button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/Card';
import { api } from '../../lib/api';
import type { JobStatus } from '../../types';

interface JobListProps {
  refreshTrigger: number;
  onJobCancelled?: (jobId: string) => void;
}

export function JobList({ refreshTrigger, onJobCancelled }: JobListProps) {
  const [jobs, setJobs] = useState<JobStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadJobs = async () => {
    try {
      setError(null);
      const jobList = await api.getJobs();
      setJobs(jobList);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to load jobs');
    } finally {
      setLoading(false);
    }
  };

  const refreshJob = async (jobId: string) => {
    try {
      const updatedJob = await api.getJobStatus(jobId);
      setJobs(prev => prev.map(job => 
        job.job_id === jobId ? updatedJob : job
      ));
    } catch (err) {
      console.error('Failed to refresh job:', err);
    }
  };

  const refreshAll = async () => {
    setRefreshing(true);
    await loadJobs();
    setRefreshing(false);
  };

  useEffect(() => {
    loadJobs();
  }, [refreshTrigger]);

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
            <span className="ml-2 text-gray-600">Loading jobs...</span>
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
            <Button onClick={loadJobs} variant="outline">
              Try Again
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (jobs.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Job History</CardTitle>
          <CardDescription>No jobs found</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <p className="text-gray-500">Submit a training or inference job to get started</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Job History</CardTitle>
            <CardDescription>
              {jobs.length} job{jobs.length !== 1 ? 's' : ''} found
            </CardDescription>
          </div>
          <Button 
            onClick={refreshAll} 
            loading={refreshing}
            variant="outline"
            size="sm"
          >
            Refresh All
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {jobs.map((job) => (
            <JobStatusCard
              key={job.job_id}
              job={job}
              onRefresh={() => refreshJob(job.job_id)}
              onJobCancelled={onJobCancelled}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
