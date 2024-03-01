from openai import OpenAI
import os
from dotenv import load_dotenv
from models import ChatData
import json
from datetime import datetime
from line_convert import convert_newlines_to_br
from markdown import markdown

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def stream_logic(db, chat_id="", user_message="", user_id=None, title=""):
    file_url = ""
    if chat_id is not None:
        conversation = db.query(ChatData).filter(ChatData.id == chat_id).first()
        existing_messages = json.loads(conversation.messages) if conversation.messages else []
    else:
        existing_messages = []
    existing_messages.append({"role": "user", "content": user_message})
    if len(title) == 0:
        title = user_message
    if chat_id is not None:
        conversation.messages = json.dumps(existing_messages)
    else:
        conversation = ChatData(user_id=user_id, title=title, content_type="Text/Chat", messages=json.dumps(existing_messages),
                                created_at=datetime.utcnow())
        db.add(conversation)
    db.commit()
    completion = client.chat.completions.create(
        model='gpt-4-1106-preview',
        messages=existing_messages,
        temperature=1,
        stream=True
    )
    text = ''
    for event in completion:
        # print(event.choices[0])
        current_response = event.choices[0].delta.content
        if current_response:
            make = current_response

            text += make
            yield make

    make = "$$$$\n" + json.dumps({
        "content-type": conversation.content_type,
        "status": "success",
        "data": markdown(convert_newlines_to_br(text)),
        "text": text,
        "file_url": file_url,
        "conversation": conversation.id
    })
    yield make

    assistant_reply_data = {"role": "assistant", "content": text}
    existing_messages.append(assistant_reply_data)

    conversation.messages = json.dumps(existing_messages)
    db.commit()
