
### How to Test in Mozilla Firefox

Firefox refers to extensions as "Add-ons." You'll load it as a temporary add-on.

1.  **Open Firefox** and type `about:debugging` into the address bar, then press Enter.
2.  In the left-hand menu, click on **"This Firefox"**.
3.  Click the **"Load Temporary Add-on..."** button.
4.  Navigate to your `firefox_extension` directory and select the `manifest.json` file (or any other file in that directory). Click **Open**.
5.  Your "RocketRecap Summarizer" extension will now appear in the "Temporary Extensions" list and its icon will be added to the Firefox toolbar.

The extension will remain installed until you close Firefox. If you make changes to your code, you can click the "Reload" button next to the extension's entry on the `about:debugging` page to apply them.

### How to Test in Google Chrome

1.  **Open Chrome** and type `chrome://extensions` into the address bar, then press Enter.
2.  In the top-right corner, toggle the **"Developer mode"** switch to the "on" position.
3.  Three new buttons will appear: "Load unpacked," "Pack extension," and "Update." Click the **"Load unpacked"** button.
4.  A file dialog will open. Navigate to and select your `chrome_extension` directory, then click **"Select Folder"**.
5.  The "RocketRecap Summarizer" extension will now be installed locally and its icon will appear in your Chrome toolbar.

If you make changes to your code, you can go back to the `chrome://extensions` page and click the reload icon for your extension to update it.

### Debugging Your Extension

Once loaded, you can debug your extension using the browser's developer tools.

*   **To debug the popup (`popup.html` and `popup.js`):**
    1.  Click the extension's icon in the toolbar to open the popup.
    2.  Right-click anywhere inside the popup and select **"Inspect"**.
    3.  This will open a dedicated DevTools window for the popup, where you can inspect the HTML, debug JavaScript, and view console logs and network requests.

*   **To debug content scripts (if you had any):**
    1.  Go to a webpage where the content script is injected.
    2.  Open the regular DevTools for that page (by right-clicking the page and selecting "Inspect").
    3.  In the "Sources" (Chrome) or "Debugger" (Firefox) tab, you can find your extension's content scripts and set breakpoints.

### Test Scenarios to Run

To ensure your extension works as expected, try these scenarios in both browsers:

1.  **Test on a YouTube Video:**
    *   Navigate to a YouTube video.
    *   Click the extension icon and then the "Summarize" button.
    *   Verify that the status changes to "Working..." and then "Summary request sent!".
    *   Check your server logs to ensure it received a request with the YouTube URL and an empty `transcript` field.

2.  **Test with Selected Text:**
    *   Go to any webpage with text (like a news article).
    *   Highlight a paragraph of text.
    *   Click the extension icon and then the "Summarize" button.
    *   Verify that the server receives the request with the page URL and the selected text in the `transcript` field.

3.  **Test on a Standard Webpage (No Selection):**
    *   Go to a webpage with text.
    *   Do *not* select any text.
    *   Click the extension icon and then the "Summarize" button.
    *   Confirm that the server receives the request with the page URL and the entire page's inner text.

4.  **Test Model Selection:**
    *   Open the popup and select a different model from the dropdown.
    *   Close and reopen the popup to ensure your selection was saved and is still selected.
