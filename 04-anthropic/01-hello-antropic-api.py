from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic


## Bu çalışabilmesi için:
# 1) pip install anthropic
# 2) https://console.anthropic.com/ adresinden API key alın
# 3) .env dosyasına şunu ekleyin:
# ANTHROPIC_API_KEY=your_api_key


def add_user_message(messages, content):
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages, content):
    messages.append({"role": "assistant", "content": content})


def chat(messages):
    client = Anthropic()
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=messages,
        max_tokens=1024,
    )
    return response.content[0].text


def display_messages(messages):
    print("\n" + "=" * 60)
    print("MESSAGES BEING SENT TO LLM:")
    print("=" * 60)
    for i, msg in enumerate(messages, 1):
        role = msg["role"].upper()
        content = msg["content"]
        print(f"\n[Message {i} - {role}]")
        print(content)
    print("=" * 60 + "\n")


def main():
    messages = []

    print("Simple Console Chatbot")
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
        print(f"Assistant: {response}\n")

        add_assistant_message(messages, response)


if __name__ == "__main__":
    main()