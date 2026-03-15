import { useState, useEffect, useCallback } from 'react';
import { getDocuments, uploadDocument } from '../api/client';

export function useDocuments() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState(null); // 'uploading' | 'chunking' | 'done' | null

  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getDocuments();
      setDocuments(res.data?.documents ?? []);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to load documents');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const upload = useCallback(async (file) => {
    setUploadProgress(0);
    setUploadStatus('uploading');
    if (typeof window !== 'undefined') {
      window.__onUploadProgress = (pct) => setUploadProgress(pct);
    }
    try {
      setUploadStatus('chunking');
      const res = await uploadDocument(file);
      setUploadProgress(100);
      setUploadStatus('done');
      await fetchDocuments();
      return res.data;
    } catch (err) {
      setUploadStatus(null);
      setUploadProgress(0);
      throw err;
    } finally {
      if (typeof window !== 'undefined') window.__onUploadProgress = undefined;
    }
  }, [fetchDocuments]);

  const clearUploadState = useCallback(() => {
    setUploadProgress(0);
    setUploadStatus(null);
  }, []);

  return {
    documents,
    loading,
    error,
    fetchDocuments,
    upload,
    uploadProgress,
    uploadStatus,
    clearUploadState,
  };
}
