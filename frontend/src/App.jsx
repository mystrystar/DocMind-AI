import { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { SearchBar } from './components/SearchBar';
import { ChatWindow } from './components/ChatWindow';
import { UploadModal } from './components/UploadModal';
import { ChunksModal } from './components/ChunksModal';
import { useDocuments } from './hooks/useDocuments';
import { useChat } from './hooks/useChat';
import { useDocumentChunks } from './hooks/useDocumentChunks';

function App() {
  const [activeDocId, setActiveDocId] = useState(null);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [chunksModalDocId, setChunksModalDocId] = useState(null);
  const [chunksModalFilename, setChunksModalFilename] = useState('');
  const { chunks, loading: chunksLoading, error: chunksError } = useDocumentChunks(chunksModalDocId);

  const {
    documents,
    loading: docsLoading,
    error: docsError,
    upload,
    uploadProgress,
    uploadStatus,
    clearUploadState,
    deleteDoc,
  } = useDocuments();

  const { messages, loading: chatLoading, error: chatError, sendMessage, clearMessages } = useChat(activeDocId);

  const activeDoc = documents.find((d) => d.doc_id === activeDocId);

  const handleUploadDone = () => {
    clearUploadState();
    setUploadModalOpen(false);
  };

  return (
    <div className="flex h-screen bg-dark-bg text-gray-200">
      <Sidebar
        documents={documents}
        loading={docsLoading}
        error={docsError}
        activeDocId={activeDocId}
        onSelectDoc={setActiveDocId}
        onViewChunks={(id, filename) => {
          setChunksModalDocId(id);
          setChunksModalFilename(filename || '');
        }}
        onDeleteDoc={(id, filename) => {
          const name = filename || 'this document';
          if (window.confirm(`Delete ${name} and its vectors?`)) {
            deleteDoc(id);
            if (activeDocId === id) {
              setActiveDocId(null);
            }
          }
        }}
      />

      <main className="flex flex-1 flex-col min-w-0">
        {/* Top bar: semantic search */}
        <header className="flex flex-shrink-0 items-center justify-between border-b border-gray-800 bg-dark-card px-4 py-3">
          <h1 className="text-xl font-semibold text-white">DocMind AI</h1>
          <SearchBar
            documentId={activeDocId}
            placeholder={activeDocId ? 'Search in document...' : 'Select a document to search'}
          />
        </header>

        {/* Main: chat */}
        <div className="flex-1 min-h-0 overflow-hidden">
          <ChatWindow
            documentId={activeDocId}
            messages={messages}
            loading={chatLoading}
            error={chatError}
            sendMessage={sendMessage}
            activeDocName={activeDoc?.filename}
          />
        </div>
      </main>

      {/* Floating upload button */}
      <button
        type="button"
        onClick={() => setUploadModalOpen(true)}
        className="fixed bottom-6 right-6 flex h-14 w-14 items-center justify-center rounded-full bg-accent text-white shadow-lg transition hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-dark-bg"
        aria-label="Upload document"
      >
        <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
        </svg>
      </button>

      {uploadModalOpen && (
        <UploadModal
          onClose={() => {
            if (uploadStatus !== 'uploading' && uploadStatus !== 'chunking') {
              clearUploadState();
              setUploadModalOpen(false);
            }
          }}
          onUpload={upload}
          uploadProgress={uploadProgress}
          uploadStatus={uploadStatus}
        />
      )}

      {chunksModalDocId && (
        <ChunksModal
          docId={chunksModalDocId}
          filename={chunksModalFilename}
          onClose={() => {
            setChunksModalDocId(null);
            setChunksModalFilename('');
          }}
          chunks={chunks}
          loading={chunksLoading}
          error={chunksError}
        />
      )}
    </div>
  );
}

export default App;
