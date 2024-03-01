from openai import OpenAI
import os
from dotenv import load_dotenv
from models import UploadedFile, ChatData
import json
from datetime import datetime
import requests
import uuid
from prompt_from_image import get_prompt_description
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
FILE_URL = os.getenv("FILE_URL")
GENERATES_DIR = "generates"


def generate_image(db, chat_id="", user_message="", user_id=None, title=""):
    file_url = ""
    if chat_id is not None:
        conversation = db.query(ChatData).filter(ChatData.id == chat_id).first()
        file = db.query(UploadedFile).filter(UploadedFile.chat_id == chat_id).first()
        if file is not None:
            file_url = f"{FILE_URL}/uploads/{file.filename}"
        else:
            file_url = ""

        existing_messages = json.loads(conversation.messages) if conversation.messages else []
    else:
        existing_messages = []
    # print(len(existing_messages))
    if len(existing_messages) > 0:
        prompt_from_image = get_prompt_description(existing_messages, user_message)
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt_from_image,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        if image_url:
            existing_messages.append({"role": "user", "content": user_message})
        # Download the image from the URL
        image_data = requests.get(image_url).content
        filename = f"{uuid.uuid4()}.png"

        # Save the decoded image to the server
        with open(os.path.join(GENERATES_DIR, filename), "wb") as buffer:
            buffer.write(image_data)

        file_url = f"{FILE_URL}/generates/{filename}"

        reply = ""

        assistant_reply_data = {"role": "assistant", "content": "", "file_url": f"{file_url}"}
        existing_messages.append(assistant_reply_data)

    else:
        response = client.images.generate(
            model="dall-e-3",
            prompt=user_message,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        if image_url:
            existing_messages.append({"role": "user", "content": user_message})
# Download the image from the URL
        image_data = requests.get(image_url).content
        filename = f"{uuid.uuid4()}.png"

        # Save the decoded image to the server
        with open(os.path.join(GENERATES_DIR, filename), "wb") as buffer:
            buffer.write(image_data)

        file_url = f"{FILE_URL}/generates/{filename}"

        reply = ""

        assistant_reply_data = {"role": "assistant", "content": "", "file_url": f"{file_url}"}
        existing_messages.append(assistant_reply_data)
    if len(title) == 0:
        title = user_message
    if chat_id is not None:
        conversation.messages = json.dumps(existing_messages)
    else:
        conversation = ChatData(user_id=user_id, title=title, content_type="Image/Generate", messages=json.dumps(existing_messages),
                                created_at=datetime.utcnow())
        db.add(conversation)
    db.commit()
    # print(reply, conversation, file_url)
    return reply, conversation, file_url
