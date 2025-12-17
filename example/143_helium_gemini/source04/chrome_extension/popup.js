// javascript
// Single source of truth for available models and their display labels
const MODELS = [
  {
    value: "gemini-3-flash-preview| input-price: 0.5 output-price: 3 max-context-length: 128_000",
    label: "gemini-3-flash-preview"
  },
  {
    value: "gemini-2.5-flash-preview-09-2025| input-price: 0.3 output-price: 2.5 max-context-length: 128_000",
    label: "gemini-2.5-flash-preview-09-2025"
  },
  {
    value: "gemini-2.5-flash-lite-preview-09-2025| input-price: 0.1 output-price: 0.4 max-context-length: 128_000",
    label: "gemini-2.5-flash-lite-preview-09-2025"
  }
];

document.addEventListener('DOMContentLoaded', () => {
  const modelSelect = document.getElementById('modelSelect');
  if (modelSelect) {
    // build options from MODELS
    MODELS.forEach((m) => {
      const opt = document.createElement('option');
      opt.value = m.value;
      opt.textContent = m.label;
      modelSelect.appendChild(opt);
    });

    // restore saved model or select the first model by default
    const saved = localStorage.getItem('selectedModel');
    const hasSaved = saved && Array.from(modelSelect.options).some(o => o.value === saved);
    modelSelect.value = hasSaved ? saved : MODELS[0].value;

    // persist changes
    modelSelect.addEventListener('change', () => {
      localStorage.setItem('selectedModel', modelSelect.value);
    });
  }

  const summarizeButtonEl = document.getElementById('summarizeButton');
  if (!summarizeButtonEl) return;

  summarizeButtonEl.addEventListener('click', () => {
    const summarizeButton = document.getElementById('summarizeButton');
    const status = document.getElementById('status');

    summarizeButton.disabled = true;
    status.textContent = 'Working...';

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const currentTab = tabs && tabs[0];
      if (!currentTab) {
        status.textContent = 'Could not get current tab.';
        summarizeButton.disabled = false;
        return;
      }

      // Inject a small script to read selection and page text
      chrome.scripting.executeScript({
        target: { tabId: currentTab.id, allFrames: false },
        func: () => {
          const sel = typeof window.getSelection === 'function' ? window.getSelection().toString() : '';
          const txt = document && document.body ? document.body.innerText : '';
          return {
            selection: sel.trim(),
            text: txt.trim()
          };
        }
      }).then((injectionResults) => {
        const pageResult = injectionResults && injectionResults[0] && injectionResults[0].result ? injectionResults[0].result : {};
        const selectionText = pageResult.selection || '';
        const pageText = pageResult.text || '';
        const pageUrl = currentTab.url || '';

        let transcriptToSend = '';
        if (selectionText) {
          transcriptToSend = selectionText;
        } else if (pageUrl.includes('youtube.com/watch')) {
          transcriptToSend = '';
        } else {
          transcriptToSend = pageText;
        }

        const formData = new FormData();
        formData.append('original_source_link', pageUrl);
        formData.append('transcript', transcriptToSend);

        const modelSelectEl = document.getElementById('modelSelect');
        const selectedModel = (modelSelectEl && modelSelectEl.value) ? modelSelectEl.value : MODELS[0].value;
        formData.append('model', selectedModel);

        formData.append('output_language', 'en');
        formData.append('include_timestamps', 'on');

        fetch('https://rocketrecap.com/process_transcript', {
          method: 'POST',
          body: formData,
        })
        .then(response => {
          status.textContent = response.ok ? 'Summary request sent!' : 'Error sending request.';
          summarizeButton.disabled = false;
        })
        .catch(error => {
          console.error('Error:', error);
          status.textContent = 'An error occurred.';
          summarizeButton.disabled = false;
        });
      }).catch((err) => {
        console.error('Injection error:', err);
        status.textContent = 'Could not read page content.';
        summarizeButton.disabled = false;
      });
    });
  });
});