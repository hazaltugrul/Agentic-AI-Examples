from dotenv import load_dotenv
from anthropic import Anthropic
import json

load_dotenv()


def get_current_datetime_schema():
    return {
        "name": "get_current_datetime",
        "description": "Get the current date and time in a specified timezone with configurable output format. Returns datetime string, timezone, and unix timestamp.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone_str": {
                    "type": "string",
                    "enum": ["UTC", "EST", "PST"],
                    "description": "The timezone to return the current time in. Supported values: UTC (Coordinated Universal Time), EST (Eastern Standard Time, UTC-5), PST (Pacific Standard Time, UTC-8)",
                    "default": "UTC",
                },
                "format": {
                    "type": "string",
                    "enum": ["iso", "readable"],
                    "description": "Output format for the datetime string. 'iso' returns ISO 8601 format (e.g., 2024-01-15T14:30:00), 'readable' returns human-friendly format (e.g., January 15, 2024 at 2:30 PM)",
                    "default": "iso",
                },
            },
            "required": [],
        },
    }


def get_current_datetime(timezone_str="UTC", format="iso"):
    """
    Get the current date and time in the specified timezone and format.
    """
    from datetime import datetime, timezone

    now_utc = datetime.now(timezone.utc)

    if timezone_str == "UTC":
        current_time = now_utc
    elif timezone_str == "EST":
        current_time = now_utc.replace(hour=(now_utc.hour - 5) % 24)
    elif timezone_str == "PST":
        current_time = now_utc.replace(hour=(now_utc.hour - 8) % 24)
    else:
        current_time = now_utc

    if format == "iso":
        return {
            "datetime": current_time.isoformat(),
            "timezone": timezone_str,
            "unix_timestamp": int(current_time.timestamp()),
        }
    elif format == "readable":
        return {
            "datetime": current_time.strftime("%B %d, %Y at %I:%M %p"),
            "timezone": timezone_str,
            "unix_timestamp": int(current_time.timestamp()),
        }
    else:
        return {
            "datetime": str(current_time),
            "timezone": timezone_str,
            "unix_timestamp": int(current_time.timestamp()),
        }


def process_tool_calls(client, messages, response):
    """Process tool calls and continue the conversation until the model returns a final answer."""

    if response.stop_reason != "tool_use":
        return response

    tool_results = []

    for block in response.content:
        if block.type == "text":
            print(f"Claude says: {block.text}")
        elif block.type == "tool_use":
            print(f"Tool call detected: {block.name}")
            print(f"Parameters: {block.input}")

            if block.name == "get_current_datetime":
                tool_output = get_current_datetime(**block.input)
                print(f"Tool output: {tool_output}")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(tool_output),
                    }
                )

    if not tool_results:
        return response

    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results})

    follow_up_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=messages,
        max_tokens=1024,
        tools=[get_current_datetime_schema()],
    )

    if follow_up_response.stop_reason == "tool_use":
        return process_tool_calls(client, messages, follow_up_response)

    return follow_up_response


def main():
    client = Anthropic()
    messages = []

    print("Interactive Timezone Chatbot with Tool Use")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        print("Thinking...\n")
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=messages,
            max_tokens=1024,
            tools=[get_current_datetime_schema()],
        )

        final_response = process_tool_calls(client, messages, response)

        print("-" * 40)

        for content in final_response.content:
            if content.type == "text":
                print(f"Assistant: {content.text}")

        print("-" * 40)


if __name__ == "__main__":
    main()

