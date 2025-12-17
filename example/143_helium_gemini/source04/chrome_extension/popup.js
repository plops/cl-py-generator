document.getElementById('summarizeButton').addEventListener('click', () => {
  const summarizeButton = document.getElementById('summarizeButton');
  const status = document.getElementById('status');

  summarizeButton.disabled = true;
  status.textContent = 'Working...';

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const currentTab = tabs[0];
    if (!currentTab) {
      status.textContent = 'Could not get current tab.';
      summarizeButton.disabled = false;
      return;
    }

    // Inject a small script to read selection and page text
    chrome.scripting.executeScript({
      target: { tabId: currentTab.id, allFrames: false },
      func: () => {
        return {
          selection: (window.getSelection ? window.getSelection().toString() : '').trim(),
          text: (document.body ? document.body.innerText : '').trim()
        };
      }
    }).then((injectionResults) => {
      const pageResult = (injectionResults && injectionResults[0] && injectionResults[0].result) || {};
      const selectionText = pageResult.selection || '';
      const pageText = pageResult.text || '';
      const pageUrl = currentTab.url || '';

      // Decide what to send:
      // - If user selected text, send that only (but still include original link).
      // - Else if it's a YouTube watch page, send the link and empty transcript.
      // - Else (non-YouTube, no selection) send page text and the page link.
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

      // use selected model value from the dropdown (populated from MODELS)
      const modelSelect = document.getElementById('modelSelect');
      const selectedModel = (modelSelect && modelSelect.value) ? modelSelect.value : MODELS[0].value;
      formData.append('model', selectedModel);

      formData.append('output_language', 'en');
      formData.append('include_timestamps', 'on');

      fetch('https://rocketrecap.com/process_transcript', {
        method: 'POST',
        body: formData,
      })
      .then(response => {
        if (response.ok) {
          status.textContent = 'Summary request sent!';
        } else {
          status.textContent = 'Error sending request.';
        }
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

// Single source of truth for available models and their display labels
const MODELS = [
  {
    value: "gemini-3-flash-preview| input-price: 0.5 output-price: 3 max-context-length: 128_000",
    label: "gemini-3-flash-preview"
  },{
    value: "gemini-2.5-flash-preview-09-2025| input-price: 0.3 output-price: 2.5 max-context-length: 128_000",
    label: "gemini-2.5-flash-preview-09-2025"
  },
  {
    value: "gemini-2.5-flash-lite-preview-09-2025| input-price: 0.1 output-price: 0.4 max-context-length: 128_000",
    label: "gemini-2.5-flash-lite-preview-09-2025"
  },
  {
    value: "gemini-2.5-pro| input-price: 1.25 output-price: 10 max-context-length: 200_000",
    label: "gemini-2.5-pro"
  }
];

// Populate the model select and restore saved selection
document.addEventListener('DOMContentLoaded', () => {
  const modelSelect = document.getElementById('modelSelect');
  if (!modelSelect) return;

  // build options from MODELS
  MODELS.forEach((m) => {
    const opt = document.createElement('option');
    opt.value = m.value;
    opt.textContent = m.label;
    modelSelect.appendChild(opt);
  });

  // restore saved model or select the first model by default
  const saved = localStorage.getItem('selectedModel');
  if (saved && [...modelSelect.options].some(o => o.value === saved)) {
    modelSelect.value = saved;
  } else {
    modelSelect.value = MODELS[0].value;
  }

  // persist changes
  modelSelect.addEventListener('change', () => {
    localStorage.setItem('selectedModel', modelSelect.value);
  });
});
