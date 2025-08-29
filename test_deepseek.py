import os
from dotenv import load_dotenv
from openai import OpenAI

def test_deepseek_api():
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        print("Error: DEEPSEEK_API_KEY not found in environment variables")
        return False

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "Say this is a test!"}
            ],
            temperature=0.7,
            max_tokens=50
        )

        print("✅ DeepSeek API Response:")
        print(response.choices[0].message.content)
        return True

    except Exception as e:
        print(f"❌ Error testing DeepSeek API: {str(e)}")
        return False

if __name__ == "__main__":
    test_deepseek_api()
