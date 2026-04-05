const ENDPOINT_URL = 'https://rocketrecap.com/process_transcript';

const MODELS = [
  {
    value: 'gemini-3.1-flash-lite-preview| input: $0.25 | output: $1.5 | context: 1_000_000 | rpm: 15 | rpd: 500',
    label: 'gemini-3.1-flash-lite-preview'
  },
  {
    value: 'gemini-3-flash-preview| input: $0.5 | output: $3.0 | context: 1_000_000 | rpm: 5 | rpd: 20',
    label: 'gemini-3-flash-preview'
  },
  {
    value: 'gemini-2.5-flash| input: $0.3 | output: $2.5 | context: 1_000_000 | rpm: 5 | rpd: 20',
    label: 'gemini-2.5-flash'
  },
  {
    value: 'gemini-2.5-flash-lite| input: $0.1 | output: $0.4 | context: 1_000_000 | rpm: 10 | rpd: 20',
    label: 'gemini-2.5-flash-lite'
  },
  {
    value: 'gemini-robotics-er-1.5-preview| input: $0.3 | output: $2.5 | context: 1_000_000 | rpm: 10 | rpd: 20',
    label: 'gemini-robotics-er-1.5-preview'
  }
];

function formatByteCount(byteCount) {
  if (byteCount < 1024) {
    return `${byteCount} B`;
  }
  if (byteCount < 1024 * 1024) {
    return `${(byteCount / 1024).toFixed(1)} KB`;
  }
  return `${(byteCount / (1024 * 1024)).toFixed(1)} MB`;
}

function isYouTubeWatchUrl(url) {
  try {
    const parsedUrl = new URL(url);
    return (
      (parsedUrl.hostname === 'www.youtube.com' || parsedUrl.hostname === 'youtube.com') &&
      parsedUrl.pathname === '/watch'
    );
  } catch (_error) {
    return url.includes('youtube.com/watch');
  }
}

async function getActiveTab() {
  const tabs = await browser.tabs.query({ active: true, currentWindow: true });
  return tabs[0] || null;
}

async function extractPageContent(tabId) {
  const injectionResults = await browser.scripting.executeScript({
    target: { tabId, allFrames: false },
    func: () => {
      const normalizeWhitespace = (text) => (text || '').replace(/\s+/g, ' ').trim();
      const getText = (selector) => {
        const element = document.querySelector(selector);
        return normalizeWhitespace(element ? element.innerText : '');
      };

      const selection = normalizeWhitespace(
        typeof window.getSelection === 'function' ? window.getSelection().toString() : ''
      );
      const bodyText = normalizeWhitespace(document.body ? document.body.innerText : '');
      const candidates = [
        getText('article'),
        getText('main'),
        getText('[role="main"]')
      ].filter(Boolean);
      const primaryText = candidates.reduce(
        (best, current) => (current.length > best.length ? current : best),
        ''
      );

      return {
        selection,
        text: primaryText.length >= 500 ? primaryText : bodyText
      };
    }
  });

  return (injectionResults && injectionResults[0] && injectionResults[0].result) || {};
}

function chooseTranscript(pageUrl, selectionText, pageText) {
  if (selectionText) {
    return selectionText;
  }
  if (isYouTubeWatchUrl(pageUrl)) {
    return '';
  }
  return pageText;
}

async function encodePayload(payload) {
  const encoder = new TextEncoder();
  const jsonPayload = JSON.stringify(payload);
  const uncompressedBytes = encoder.encode(jsonPayload);
  const headers = {
    'Content-Type': 'application/json'
  };

  if (typeof CompressionStream === 'function') {
    const compressionStream = new CompressionStream('gzip');
    const writer = compressionStream.writable.getWriter();
    await writer.write(uncompressedBytes);
    await writer.close();

    const compressedBuffer = await new Response(compressionStream.readable).arrayBuffer();
    headers['Content-Encoding'] = 'gzip';

    return {
      body: new Uint8Array(compressedBuffer),
      headers,
      originalBytes: uncompressedBytes.byteLength,
      encodedBytes: compressedBuffer.byteLength
    };
  }

  return {
    body: jsonPayload,
    headers,
    originalBytes: uncompressedBytes.byteLength,
    encodedBytes: uncompressedBytes.byteLength
  };
}

async function loadSelectedModel(modelSelect) {
  const result = await browser.storage.local.get('selectedModel');
  const savedModel = result.selectedModel;
  if (savedModel && [...modelSelect.options].some((option) => option.value === savedModel)) {
    modelSelect.value = savedModel;
    return;
  }
  modelSelect.value = MODELS[0].value;
}

async function handleSummarizeClick() {
  const summarizeButton = document.getElementById('summarizeButton');
  const status = document.getElementById('status');
  const modelSelect = document.getElementById('modelSelect');

  summarizeButton.disabled = true;
  status.textContent = 'Working...';

  try {
    const currentTab = await getActiveTab();
    if (!currentTab) {
      status.textContent = 'Could not get current tab.';
      return;
    }

    const pageResult = await extractPageContent(currentTab.id);
    const selectionText = pageResult.selection || '';
    const pageText = pageResult.text || '';
    const pageUrl = currentTab.url || '';
    const transcript = chooseTranscript(pageUrl, selectionText, pageText);
    const selectedModel = modelSelect && modelSelect.value ? modelSelect.value : MODELS[0].value;
    const payload = {
      original_source_link: pageUrl,
      transcript,
      model: selectedModel,
      output_language: 'en',
      include_timestamps: true
    };
    const requestData = await encodePayload(payload);
    const response = await fetch(ENDPOINT_URL, {
      method: 'POST',
      headers: requestData.headers,
      body: requestData.body
    });

    if (!response.ok) {
      status.textContent = 'Error sending request.';
      return;
    }

    if (requestData.encodedBytes < requestData.originalBytes) {
      status.textContent = `Sent ${formatByteCount(requestData.originalBytes)} as ${formatByteCount(requestData.encodedBytes)}.`;
      return;
    }

    status.textContent = 'Summary request sent!';
  } catch (error) {
    console.error('Error:', error);
    status.textContent = 'An error occurred.';
  } finally {
    summarizeButton.disabled = false;
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  const modelSelect = document.getElementById('modelSelect');
  const summarizeButton = document.getElementById('summarizeButton');
  if (!modelSelect || !summarizeButton) {
    return;
  }

  MODELS.forEach((model) => {
    const option = document.createElement('option');
    option.value = model.value;
    option.textContent = model.label;
    modelSelect.appendChild(option);
  });

  await loadSelectedModel(modelSelect);

  modelSelect.addEventListener('change', () => {
    browser.storage.local.set({ selectedModel: modelSelect.value });
  });

  summarizeButton.addEventListener('click', handleSummarizeClick);
});
