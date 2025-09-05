document.getElementById('summarizeButton').addEventListener('click', () => {
  const summarizeButton = document.getElementById('summarizeButton');
  const status = document.getElementById('status');

  summarizeButton.disabled = true;
  status.textContent = 'Working...';

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const currentTab = tabs[0];
    if (currentTab) {
      const youtubeUrl = currentTab.url;

      // Optional: Check if it's a YouTube URL
      if (!youtubeUrl.includes("youtube.com/watch")) {
        status.textContent = "Not a YouTube video page.";
        summarizeButton.disabled = false;
        return;
      }

      const formData = new FormData();
      formData.append('original_source_link', youtubeUrl);
      formData.append('transcript', '');
      formData.append('model', 'gemini-2.5-pro| input-price: 1.25 output-price: 10 max-context-length: 200_000');
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
    } else {
        status.textContent = 'Could not get current tab.';
        summarizeButton.disabled = false;
    }
  });
});
