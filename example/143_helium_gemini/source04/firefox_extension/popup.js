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
    value: "gemini-3.1-flash-lite-preview| input: $0.25 | output: $1.5 | context: 1_000_000 | rpm: 15 | rpd: 500",
    label: "gemini-3.1-flash-lite-preview"
  },
  {
    value: "gemini-3-flash-preview| input: $0.5 | output: $3.0 | context: 1_000_000 | rpm: 5 | rpd: 20",
    label: "gemini-3-flash-preview"
  },
  {
    value: "gemini-2.5-flash| input: $0.3 | output: $2.5 | context: 1_000_000 | rpm: 5 | rpd: 20",
    label: "gemini-2.5-flash"
  },
  {
    value: "gemini-2.5-flash-lite| input: $0.1 | output: $0.4 | context: 1_000_000 | rpm: 10 | rpd: 20",
    label: "gemini-2.5-flash-lite"
  },
  {
    value: "gemini-robotics-er-1.5-preview| input: $0.3 | output: $2.5 | context: 1_000_000 | rpm: 10 | rpd: 20",
    label: "gemini-robotics-er-1.5-preview"
  }
];

// Populate the model select and restore saved selection
document.addEventListener('DOMContentLoaded', async () => {
  const modelSelect = document.getElementById('modelSelect');
  if (!modelSelect) return;

  MODELS.forEach((m) => {
    const opt = document.createElement('option');
    opt.value = m.value;
    opt.textContent = m.label;
    modelSelect.appendChild(opt);
  });

  const result = await browser.storage.local.get('selectedModel');
  const saved = result.selectedModel;
  if (saved && [...modelSelect.options].some(o => o.value === saved)) {
    modelSelect.value = saved;
  } else {
    modelSelect.value = MODELS[0].value;
  }

  modelSelect.addEventListener('change', () => {
    browser.storage.local.set({ selectedModel: modelSelect.value });
  });
});
