# python -m venv helium_env; . helium_env/activate; pip install helium


from helium import *
import time

# This code assumes you are already logged into the platform where the button exists.

try:
    # Open the website or platform where the button is located
    # Example:
    start_chrome("http://localhost:5173") 
    # Replace with the actual URL

    # Wait for the page to load
    time.sleep(5)

    # Find and click the "Create New Lobby" button
    click(Button("Create New Lobby"))

    # Optionally, you can add more actions after creating the lobby

    print("New lobby created successfully!")

except Exception as e:
    print(f"An error occurred: {e}")

#finally:
    # Close the browser window
#    kill_browser()
