
# Install Helium into a virtual environment
# python -m venv helium_env; . helium_env/bin/activate; pip install helium


from helium import *
import time

try:
    start_chrome("http://localhost:5173") 
    wait_until(Button("Create New Lobby").exists, timeout_secs=10)
    click(Button("Create New Lobby"))
    wait_until(Button("Join").exists, timeout_secs=10)
    write("test_user", into=TextField("Enter username"))
    press(ENTER)
    click(Button("Join"))
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Close the browser window
    kill_browser()