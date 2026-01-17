import grpc
from google.ai.generativelanguage.v1beta import (
    generative_service_pb2_grpc as gen_service,
    generative_service_pb2 as gen_types,
    content_pb2 as content_types
)
from pydantic import BaseModel, Field
from typing import List

class Employee(BaseModel):
    """Represents an employee in an organization."""
    name: str
    employee_id: int
    reports: List["Employee"] = Field(
        default_factory=list,
        description="A list of employees reporting to this employee."
    )

print(Employee.model_json_schema())

channel = grpc.secure_channel(
    "generativelanguage.googleapis.com:443", grpc.ssl_channel_credentials()
)
stub = gen_service.GenerativeServiceStub(channel)
with open("/home/kiel/api_key.txt") as f:
    api_key = f.read().strip()
metadata = [("x-goog-api-key", api_key)]

content = content_types.Content(
    parts=[content_types.Part(text="""Generate an organization chart for a small team.
The manager is Alice, who manages Bob and Charlie. Bob manages David.""")])

config = gen_types.GenerationConfig(
    response_mime_type="application/json",
#    response_json_schema=Employee.model_json_schema()
)

request = gen_types.GenerateContentRequest(
    model="models/gemini-2.0-flash-lite-preview",
    contents=[content],
    generation_config=config
)


# Execute the call
response = stub.GenerateContent(request, metadata=metadata)

# Handle the response
for candidate in response.candidates:
    for part in candidate.content.parts:
        print(part.text)
