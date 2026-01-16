import grpc
# Replace with your actual generated stub names
from google.ai.generativelanguage.v1beta import generative_service_pb2_grpc as gen_service
from google.ai.generativelanguage.v1beta import generative_service_pb2 as gen_types

# 1. Establish a secure channel to the Google endpoint
channel = grpc.secure_channel(
    'generativelanguage.googleapis.com:443',
    grpc.ssl_channel_credentials()
)
stub = gen_service.GenerativeServiceStub(channel)

# 2. Build the request message
content = gen_types.Content(
    parts=[gen_types.Part(text="Explain gRPC in one sentence.")]
)
request = gen_types.GenerateContentRequest(
    model="models/gemini-1.5-flash",
    contents=[content]
)

# 3. Add API Key to metadata (Mandatory for authentication)
# The header key must be 'x-goog-api-key'
metadata = [('x-goog-api-key', 'YOUR_API_KEY')]

# 4. Execute the call
response = stub.GenerateContent(request, metadata=metadata)

# 5. Handle the response
for candidate in response.candidates:
    for part in candidate.content.parts:
        print(part.text)
