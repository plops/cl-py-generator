import openal
import time
import numpy as np  # For generating sound data

# Initialize OpenAL
openal.oalInit()

# Create a listener
listener = openal.Listener()

# Create a source
source = openal.Source()

# Generate some simple sound data (a sine wave)
frequency = 440  # Hz
duration = 2  # seconds
sample_rate = 44100
num_samples = int(duration * sample_rate)
time_array = np.linspace(0, duration, num_samples, False)
sound_data = (np.sin(2 * np.pi * frequency * time_array) * 32767).astype(np.int16)  # Scale to 16-bit

# Create a buffer and fill it with the sound data
buffer = openal.Buffer()
buffer.data(sound_data, format=openal.AL_FORMAT_MONO16, rate=sample_rate)

# Attach the buffer to the source
source.buffer = buffer

# Set the source's position (3D coordinates)
source.position = (1, 0, 0)  # Place the source slightly to the right

# Set the listener's position and orientation (optional)
listener.position = (0, 0, 0)  # Listener at the origin
listener.orientation = (0, 0, -1, 0, 1, 0)  # Looking forward

# Play the sound
source.play()

# Keep the program running while the sound plays
time.sleep(duration)

# Clean up
source.delete()
buffer.delete()
openal.oalQuit()

print("Sound played.")
