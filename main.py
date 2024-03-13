import chainlit as cl
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_KEY")

client = OpenAI(api_key=api_key)


@cl.on_message
async def main(message: cl.Message):

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": "You are a assistant who is obssesed with Machine learning.",
            },
            {"role": "user", "content": message.content},
        ],
    )

    await cl.Message(content=response.choices[0].message.content).send()
