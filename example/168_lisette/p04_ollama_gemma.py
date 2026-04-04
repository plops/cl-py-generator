from lisette import Chat  
  
def add_numbers(a: int, b: int) -> int:  
    """Add two numbers."""  
    return a + b  
  
def multiply(a: int, b: int) -> int:  
    """Multiply two numbers."""  
    return a * b  
  
# Initialize chat  
chat = Chat(  
    model="openai/gemma4:e2b",  
    api_base="http://localhost:11434",  
    api_key="ollama",  # ollama doesn't need real API key  
    tools=[add_numbers, multiply]  
)  
  
# Use max_steps in the call  
response = chat(  
    "Calculate (15 + 25) * 3 step by step using the tools.",  
    max_steps=4,  
    return_all=True  # Optional: see all intermediate steps  
)  
  
# Display results  
for step in response:  
    print(step.choices[0].message.content)
