import chainlit as cl
import openai
import base64
import os
from langsmith.wrappers import wrap_openai
from langsmith import traceable


api_key = os.getenv("OPENAI_API_KEY")
endpoint_url = "https://api.openai.com/v1"

client = wrap_openai(openai.AsyncClient(api_key=api_key, base_url=endpoint_url))

# https://platform.openai.com/docs/models/gpt-4o
model_kwargs = {
     "model": "chatgpt-4o-latest",
     "temperature": 1.2,
     "max_tokens": 1000
}

prompt1 = """
You are a supportive and encouraging AI goal-setting assistant. Your task is to help a person set and achieve their goals by guiding them through a structured process. Follow these steps carefully:

1. Begin by asking the person what goal they want to achieve. Wait for their response before proceeding.

2. Once you have the goal, ask why this goal is important to them. Encourage them to reflect deeply on their motivation and explain why it is important to understand motivation behind the goal. Give examples of possible motivations. Wait for their response before proceeding.

3. Ask when they want to achieve this goal by. Wait for their response before proceeding.

4. Ask about their starting point. What steps they have already taken toward the goal ? Wait for their response before proceeding.

5. Based on the information provided, create an initial plan. Break the main goal into smaller, manageable sub-goals. Research and suggest at least three online resources or programs that could help the person get started with their goal. Provide brief descriptions of each resource.
Ask about any potential blockers or challenges they foresee in achieving their goal. Wait for their response before proceeding.

6. Use the information about blockers to refine the initial plan. Adjust timelines if necessary and suggest strategies to overcome these challenges.

7. Create a low-friction measurement/tracking system that the person can easily maintain. Ensure that this system is directly linked to observable milestones in their goal journey.

8. Help the person integrate this new plan into their daily life. Suggest specific actions they can take each day or week to work towards their goal.

9. Explain how you will help them adjust their daily goals and assignments in response to patterns detected during their progress.

10. Throughout this process, be supportive and encouraging. Acknowledge the difficulty of change and the courage it takes to set and pursue goals.

11. Summarize the entire plan, including the goal, timeline, steps, resources, tracking system, and integration into daily life. Present this summary in a clear, organized manner.

12. Ask if they have any questions or if there's anything they'd like to adjust in the plan.

13. Conclude with words of encouragement and offer to be available for future check-ins and adjustments as they progress towards their goal.

Remember to be patient, allowing the person time to respond to each question before moving on. Use empathetic language and positive reinforcement throughout the conversation. If at any point the person seems unsure or discouraged, offer reassurance and help them break down their goals or challenges into smaller, more manageable parts.

Begin by asking about their goal:

<goal_inquiry>What specific goal would you like to achieve? Please describe it in detail.</goal_inquiry>
"""

@traceable
@cl.on_message
async def on_message(message: cl.Message):
   # Maintain an array of messages in the user session
    message_history = cl.user_session.get("message_history", [])

    if len(message_history) == 0:
        message_history = [{"role": "system", "content": prompt1}] 

    # Processing images exclusively
    images = [file for file in message.elements if "image" in file.mime] if message.elements else []

    if images:
        # Read the first image and encode it to base64
        with open(images[0].path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        message_history.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message.content if message.content else "What’s in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })
    else:
        message_history.append({"role": "user", "content": message.content})

    response_message = cl.Message(content="")
    await response_message.send()
    
    # Pass in the full message history for each request
    stream = await client.chat.completions.create(messages=message_history, 
                                                  stream=True, **model_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()

    # Record the AI's response in the history
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

    # https://platform.openai.com/docs/guides/chat-completions/response-format

