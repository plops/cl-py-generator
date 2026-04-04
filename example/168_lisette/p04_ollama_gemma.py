from lisette import Chat  
  
# Define multiple tools  
def add_numbers(a: int, b: int) -> int:  
    """Add two numbers."""  
    return a + b  
  
def multiply(a: int, b: int) -> int:  
    """Multiply two numbers."""  
    return a * b  
  
# Initialize with ollama  
chat = Chat(  
    model="openai/gemma4:e2b",  
    api_base="http://localhost:11434",  
    api_key="ollama",  
    tools=[add_numbers, multiply],  
    max_steps=4  
)  
  
# Test tool usage  
response = chat(  
    "What is (15 + 25) * 3? Use the tools step by step.",  
    return_all=True  # Show intermediate steps  
)  
  
# Display all steps  
for step in response:  
    print(step.choices[0].message.content)
