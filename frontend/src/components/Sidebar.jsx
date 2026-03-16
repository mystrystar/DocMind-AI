/**
 * Left sidebar: list of uploaded documents; clicking one sets active document.
 * "View chunks" opens a modal to see the vector chunks for the selected document.
 * "Delete" removes the document and its vectors.
 */

export function Sidebar({
  documents,
  loading,
  error,
  activeDocId,
  onSelectDoc,
  onViewChunks,
  onDeleteDoc,
}) {
  return (
    <aside className="flex w-64 flex-shrink-0 flex-col border-r border-gray-800 bg-dark-card">
      <div className="border-b border-gray-800 px-4 py-3">
        <h2 className="font-semibold text-gray-200">Documents</h2>
      </div>
      <div className="flex-1 overflow-y-auto p-2">
        {loading && (
          <div className="space-y-2 p-2">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-12 animate-pulse rounded bg-gray-700" />
            ))}
          </div>
        )}
        {error && (
          <p className="p-2 text-sm text-red-400">{error}</p>
        )}
        {!loading && !error && documents.length === 0 && (
          <p className="p-2 text-sm text-gray-500">No documents yet. Upload a PDF to get started.</p>
        )}
        {!loading &&
          documents.map((doc) => (
            <div key={doc.doc_id} className="mb-1">
              <button
                type="button"
                onClick={() => onSelectDoc(doc.doc_id)}
                className={`w-full rounded-lg px-3 py-2.5 text-left text-sm transition ${
                  activeDocId === doc.doc_id
                    ? 'bg-accent/20 text-accent'
                    : 'text-gray-300 hover:bg-gray-800'
                }`}
              >
                <span className="block truncate font-medium">{doc.filename}</span>
                <span className="text-xs text-gray-500">{doc.chunk_count} chunks</span>
              </button>
              {activeDocId === doc.doc_id && (
                <>
                  {onViewChunks && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onViewChunks(doc.doc_id, doc.filename);
                      }}
                      className="mt-0.5 w-full rounded px-3 py-1 text-left text-xs text-accent hover:bg-gray-800"
                    >
                      View vector chunks →
                    </button>
                  )}
                  {onDeleteDoc && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteDoc(doc.doc_id, doc.filename);
                      }}
                      className="mt-0.5 w-full rounded px-3 py-1 text-left text-xs text-red-400 hover:bg-gray-800"
                    >
                      Delete document
                    </button>
                  )}
                </>
              )}
            </div>
          ))}
      </div>
    </aside>
  );
}
