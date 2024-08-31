
import markdown

doc = """
# Usage
    
    * Open the terminal
    * Run the command `python host.py`
    * Open a browser and navigate to `http://localhost:8000`
    * Enter the transcript in the text box
    * Select the model
    * Click on the "Send Transcript" button

""" 

html = markdown.markdown(doc)