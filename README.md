# AI Chatbot with DeepSeek R1

A CLI-based chatbot using DeepSeek R1 model that maintains conversation history and includes streaming responses.

## Features

- Uses DeepSeek R1 7B distilled model through Ollama
- CLI-based interface with emoji indicators for better user experience
- Maintains conversation context and memory across messages
- Real-time streaming responses with a typing effect
- Simple setup and usage

## Prerequisites

- Python 3.7+
- [Ollama](https://ollama.ai/) installed on your system

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/AI_CHATBOT.git
   cd AI_CHATBOT
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Install the DeepSeek R1 model using Ollama:
   ```
   ollama pull deepseek-r1:7b
   ```

## Usage

1. Ensure Ollama is running:
   ```
   ollama serve
   ```

2. Run the chatbot:
   ```
   python chatbot.py
   ```

3. Start chatting with the assistant. Type `exit`, `quit`, or `bye` to end the conversation.

## How It Works

The chatbot uses the following components:

1. **LangChain**: Framework for working with language models
2. **Ollama**: Local inference engine for running the DeepSeek R1 model
3. **Message History**: Tracks conversation context to maintain coherent discussions

### Code Explanation

- `system_prompt`: Defines the bot's personality and capabilities
- `llm_engine`: Connects to the Ollama service to use the DeepSeek R1 model
- `message_log`: Maintains conversation history
- `print_streaming_response()`: Displays AI responses with a realistic typing effect
- Main loop: Continuously processes user input and generates responses

## Example Conversation

```
💬 AI Assistant: Hello! How can I assist you today?

🧑 User: how are you?

💬 AI Assistant: I'm here to help you with any questions or concerns you have. How can I assist you today?

🧑 User: my name is tariq bin bashar. what is the 2 + 2?

💬 AI Assistant: Hello, Tariq Bin Bashir! The answer is 4. 😊
```

## Troubleshooting

- **Connection Error**: Make sure Ollama is running with `ollama serve`
- **Model Not Found**: Ensure you've pulled the model with `ollama pull deepseek-r1:7b`
- **Slow Responses**: The first response might be slower as the model loads into memory

## Additional Notes

- The model runs locally and doesn't require any API keys or internet connection once installed
- Response speed depends on your hardware capabilities
- The temperature parameter (0.3) is set to provide more focused and deterministic responses
