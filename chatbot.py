from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import sys
import time

# System prompt definition
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a helpful AI assistant that can engage in conversation and remember context from previous messages."
)

# Initialize LLM engine with the DeepSeek R1 model
llm_engine = ChatOllama(
    model="deepseek-r1:7b",
    base_url="http://localhost:11434",
    temperature=0.3,
)

# Initialize message log to maintain conversation history
message_log = []

# Initialize prompt sequence with system prompt
prompt_sequence = [system_prompt]
prompt_chain = ChatPromptTemplate.from_messages(prompt_sequence)

def print_streaming_response(response_stream):
    """Print the streaming response to console, simulating typing effect"""
    full_response = ""
    for chunk in response_stream:
        # Extract just the content from the AIMessageChunk
        if hasattr(chunk, 'content'):
            chunk_text = chunk.content
        else:
            chunk_text = str(chunk)
        
        # Skip empty content and metadata
        if not chunk_text or chunk_text.isspace():
            continue
            
        # Skip the thinking tags used by this model
        if "<think>" in chunk_text or "</think>" in chunk_text:
            continue
            
        full_response += chunk_text
        print(chunk_text, end="", flush=True)
        # Small delay to make the streaming visible
        time.sleep(0.01)
    print()  # New line after response is complete
    return full_response

print("\n💬 AI Assistant: Hello! How can I assist you today?")

while True:
    user_input = input("\n🧑 User: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("\n💬 AI Assistant: Goodbye! Have a great day!")
        break
    
    # Add user message to conversation history
    message_log.append({"role": "user", "content": user_input})
    
    # Rebuild prompt sequence with conversation history
    prompt_sequence = [system_prompt]
    for msg in message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
            
    prompt_chain = ChatPromptTemplate.from_messages(prompt_sequence)
    
    # Generate response with streaming
    try:
        print("\n💬 AI Assistant: ", end="", flush=True)
        
        # Create the messages for the LLM
        messages = prompt_chain.invoke({})
        
        # Stream the response from the LLM
        response_stream = llm_engine.stream(messages)
        
        # Process the streaming response
        ai_response = print_streaming_response(response_stream)
        
        # Add AI response to message log
        message_log.append({"role": "ai", "content": ai_response})
    except Exception as e:
        error_message = f"An error occurred: {str(e)}. Please check if Ollama is running and the model is correctly installed."
        print(error_message)
        message_log.append({"role": "ai", "content": error_message})
