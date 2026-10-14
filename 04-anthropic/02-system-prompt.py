from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic


def add_user_message(messages, content):
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages, content):
    messages.append({"role": "assistant", "content": content})


def chat(messages):
    system_prompt = """
    You are an expert Systems Design Mentor specializing in software architecture and distributed systems.
    Do not directly answer the user's questions. Instead, guide them through the process of designing systems by asking probing questions, providing hints,
    and encouraging critical thinking.
    Guide them to a solution step by step.
    """

    client = Anthropic()
    response = client.messages.create(
        system=system_prompt,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=messages,
    )
    return response.content[0].text


def display_messages(messages):
    """Display all messages being sent to the LLM"""
    print("\n" + "=" * 60)
    print("MESSAGES BEING SENT TO LLM:")
    print("=" * 60)
    for i, msg in enumerate(messages, 1):
        role = msg["role"].upper()
        content = msg["content"]
        print(f"\n[Message {i} - {role}]")
        print(content)
        print("=" * 60 + "\n")


if __name__ == "__main__":
    messages = []
    add_user_message(messages, "I want to design a social media feed system. Where should I start?")
    display_messages(messages)
    answer = chat(messages)
    print(answer)