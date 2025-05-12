// Upload form handler
document.getElementById('uploadForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  const fileInput = this.querySelector('input[name="file"]');
  if (!fileInput.files.length) return;
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  const res = await fetch('/api/documents/', {
    method: 'POST',
    body: formData
  });
  if (res.ok) {
    window.location.reload();
  } else {
    const err = await res.json();
    alert(err.detail || 'Upload failed');
  }
});

// Delete document by ID
async function deleteDocument(id) {
  if (!confirm('Delete this document?')) return;
  const res = await fetch(`/api/documents/${id}`, { method: 'DELETE' });
  if (res.ok) {
    window.location.reload();
  } else {
    const err = await res.json();
    alert(err.detail || 'Delete failed');
  }
}

// Send a chat question
async function sendQuestion() {
  const input = document.getElementById('questionInput');
  const question = input.value.trim();
  if (!question) return;
  appendMessage('user', question);
  input.value = '';

  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: 3 })
  });
  const data = await res.json();
  if (res.ok) {
    appendMessage('bot', data.answer);
  } else {
    appendMessage('bot', data.detail || 'Error');
  }
}

// Append a chat message to history
function appendMessage(role, text) {
  const history = document.getElementById('chatHistory');
  const msg = document.createElement('div');
  msg.className = `chat-message ${role}`;
  msg.textContent = text;
  history.appendChild(msg);
  history.scrollTop = history.scrollHeight;
}
