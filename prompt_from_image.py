import os
from dotenv import load_dotenv
from openai import OpenAI
from urllib.parse import urlparse
from encode_img import encode_image
import json
from markdown import markdown

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
GENERATES_DIR = "generates"


def get_response_on_upload(prompt, base64_image, chat, db_file, db_session):
    new_prompt = "You should have to answer in around 300 tokens:\n" + prompt
    completion = client.chat.completions.create(
        model="gpt-4-vision-preview",
        temperature=1,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": prompt
                     },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        stream=True
    )

    reply = ''
    for event in completion:
        current_response = event.choices[0].delta.content
        if current_response:
            make = current_response
            reply += make

            yield make

    file_url = ''
    make = "\n$$$$\n" + json.dumps({
        "status": "success",
        "message": "File and text extraction successful",
        "data": {
            "file_id": db_file.id,
            "file_url": file_url,
            "extracted_text_html": markdown(reply),
            "extracted_text": reply,
            "conversation_id": db_file.chat_id
        },
        "status_code": 200
    })
    yield make

    chat.title = db_file.message
    chat.messages = json.dumps([
        {
            "role": "user",
            "content": db_file.message
        },
        {
            "role": "assistant",
            "content": reply
        }
    ])
    db_session.add(chat)
    db_session.commit()


def get_desc_from_image_again(conversation, db, UploadedFile, chat_id, UPLOADS_DIR, user_message):
    db_file = db.query(UploadedFile).filter(UploadedFile.chat_id == chat_id).first()
    filename = db_file.filename
    file_location = os.path.join(UPLOADS_DIR, filename)
    base64_image = encode_image(file_location)
    prompt = "You should have to answer in around 300 tokens:\n" + user_message
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        temperature=1,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": prompt
                     },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        stream=True
    )
    reply = ''
    for event in response:
        current_response = event.choices[0].delta.content
        if current_response:
            make = current_response
            reply += make
            yield make

            file_url = ''
    make = "\n$$$$\n" + json.dumps({
        "status": "success",
        "data": markdown(reply),
        "text": reply,
        "file_url": file_url,
        "conversation": conversation.id
    })
    yield make
    existing_messages = json.loads(conversation.messages) if conversation.messages else []
    # Append the new user message
    user_message_dict = {"role": "user", "content": user_message}
    existing_messages.append(user_message_dict)
    # Append the new assistant message
    assistant_message_dict = {"role": "assistant", "content": reply}
    existing_messages.append(assistant_message_dict)
    # Convert the list of messages back to a JSON-formatted string
    conversation.messages = json.dumps(existing_messages)
    db.add(conversation)
    db.commit()


def get_prompt_description(existing_messages, user_message):

    if existing_messages:
        # Access the last dictionary in the list
        last_message = existing_messages[-1]

        # Extract the file_url from the last dictionary
        file_url = last_message.get("file_url", "")
        if file_url:
            parsed_url = urlparse(file_url)

            # Extract the filename from the path
            filename = parsed_url.path.split("/")[-1]
            file_location = os.path.join(GENERATES_DIR, filename)
            base64_image = encode_image(file_location)
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",
                             "text": "write a prompt for this image that describes it's purpose and content with each and every detail."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ]
            )
            old_image_prompt = response.choices[0].message.content
            prompt_from_image = old_image_prompt + "\nrevision: " + user_message
            return prompt_from_image
