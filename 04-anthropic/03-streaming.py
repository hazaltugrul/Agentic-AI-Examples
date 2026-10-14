from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic


def add_user_message(messages, content):
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages, content):
    messages.append({"role": "assistant", "content": content})


def chat(messages):
    """Stream the response from the LLM and return the complete text."""
    client = Anthropic()
    full_response = ""

    print("Assistant: ", end="", flush=True)
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response += text

    print("\n")
    return full_response

def display_messages (messages) :
    """Display all messages being sent to the LLM"""
    print("\n" +"=" * 60)
    print("MESSAGES BEING SENT TO LLM:")
    print("=" * 60)
    for i, msg in enumerate(messages, 1):
        role = msg["role"].upper()
        content = msg["content"]
        print(f"\n[Message {i} -{role}]")
        print(f"{content}")
    print("=" * 60 + "\n")


def main():
    messages = []

    print("Streaming Console Chatbot")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        add_user_message(messages, user_input)
        display_messages(messages)

        print("Thinking...\n")
        response = chat(messages)
        add_assistant_message(messages, response)

if __name__ == "__main__":
    main()