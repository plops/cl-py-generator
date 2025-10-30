document.getElementById('summarizeButton').addEventListener('click', async () => {
  const summarizeButton = document.getElementById('summarizeButton');
  const status = document.getElementById('status');

  summarizeButton.disabled = true;
  status.textContent = 'Working...';

  try {
    const tabs = await browser.tabs.query({ active: true, currentWindow: true });
    const currentTab = tabs[0];
    if (!currentTab) {
      status.textContent = 'Could not get current tab.';
      summarizeButton.disabled = false;
      return;
    }

    const injectionResults = await browser.scripting.executeScript({
      target: { tabId: currentTab.id, allFrames: false },
      func: () => {
        return {
          selection: (window.getSelection ? window.getSelection().toString() : '').trim(),
          text: (document.body ? document.body.innerText : '').trim()
        };
      }
    });

    const pageResult = (injectionResults && injectionResults[0] && injectionResults[0].result) || {};
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

    const modelSelect = document.getElementById('modelSelect');
    const selectedModel = (modelSelect && modelSelect.value) ? modelSelect.value : MODELS[0].value;
    formData.append('model', selectedModel);
    formData.append('output_language', 'en');
    formData.append('include_timestamps', 'on');

    const response = await fetch('https://rocketrecap.com/process_transcript', {
      method: 'POST',
      body: formData,
    });

    if (response.ok) {
      status.textContent = 'Summary request sent!';
    } else {
      status.textContent = 'Error sending request.';
    }
  } catch (error) {
    console.error('Error:', error);
    status.textContent = 'An error occurred.';
  } finally {
    summarizeButton.disabled = false;
  }
});

// Single source of truth for available models and their display labels
const MODELS = [
  {
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

  MODELS.forEach((m) => {
    const opt = document.createElement('option');
    opt.value = m.value;
    opt.textContent = m.label;
    modelSelect.appendChild(opt);
  });

  const saved = localStorage.getItem('selectedModel');
  if (saved && [...modelSelect.options].some(o => o.value === saved)) {
    modelSelect.value = saved;
  } else {
    modelSelect.value = MODELS[0].value;
  }

  modelSelect.addEventListener('change', () => {
    localStorage.setItem('selectedModel', modelSelect.value);
  });
});
