from dotenv import load_dotenv
import json
from anthropic import Anthropic

load_dotenv()


def main():
    client = Anthropic()

    print("Generating cloudformation template for EC2 instance \n")
    print("=" * 70)

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Generate a CloudFormation JSON template that provisions an EC2 instance. Return ONLY the raw JSON without markdown formatting, code blocks, or explanations."
            }
        ],
        stop_sequences=["\n\nNotes:", "\n\nThis", "\n\nHere"],
    )

    cloudformation_json = response.content[0].text

    print("cloudformation template (clean json):\n")
    print("=" * 70)
    print(cloudformation_json)
    print("=" * 70)

    try:
        parsed = json.loads(cloudformation_json)
        print("Valid JSON ready to copy and paste")
        print(f"\nResources in template: {list(parsed.get('Resources', {}).keys())}")
    except json.JSONDecodeError as e:
        print(f"\nInvalid JSON generated. Please check the output. {e}")


if __name__ == "__main__":
    main()
