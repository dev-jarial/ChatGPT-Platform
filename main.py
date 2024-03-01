from fastapi import Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from sqlalchemy import desc
from jose import JWTError, jwt
from bcrypt import checkpw, hashpw, gensalt
from datetime import datetime, timedelta, timezone
from typing import Optional, Union
from fastapi.responses import JSONResponse
import os
from openai import OpenAI
from fastapi.staticfiles import StaticFiles
import base64
import uuid
from pydantic import BaseModel, EmailStr
from DB.database import engine, SessionLocal, Base
from models import User, UploadedFile, ChatData
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from markdown import markdown
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, Request
import requests
import json
from generate_img import generate_image
from encode_img import encode_image
# Stream Chat
from fastapi.responses import StreamingResponse
from stream_text import stream_logic
from prompt_from_image import get_prompt_description, get_response_on_upload
from line_convert import convert_newlines_to_br
from prompt_from_image import get_desc_from_image_again

load_dotenv()

# Token dependency
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
openai_api_key = OPENAI_API_KEY
FILE_URL = os.getenv("FILE_URL")


# Dependency to get the database session
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = "qwert@321"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# Dependency to get the current user from the token
def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    return email


# Token creation function
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Password hashing
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserCreate(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


# User creation function
def create_user(db, user, secret):
    hashed_password = hashpw(user.password.encode('utf-8'), gensalt())
    db_user = User(email=user.email, first_name=user.first_name, last_name=user.last_name,
                   hashed_password=hashed_password, totp_secret=secret, created_at=datetime.utcnow())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


from fastapi_mail import FastMail, MessageSchema, ConnectionConfig

conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("SENT_BY"),
    MAIL_PASSWORD=os.getenv("EMAIL_PASSWD"),
    MAIL_FROM=os.getenv("SENT_BY"),
    MAIL_PORT=465,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=False,
    MAIL_SSL_TLS=True,
    MAIL_DEBUG=0,
    MAIL_FROM_NAME=None,  # Optional, can be None
    TEMPLATE_FOLDER=None,  # Optional, can be None
    SUPPRESS_SEND=0,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True,
    TIMEOUT=300  # Replace with your desired timeout value
)

mail = FastMail(conf)
import pyotp


# Set custom time window (e.g., 60 seconds)
custom_time_window = 300


# Generate a new TOTP secret
def generate_totp_secret():
    totp = pyotp.TOTP(pyotp.random_base32(), interval=custom_time_window)
    return totp.secret


# Generate TOTP for a specific user with the custom time window
def generate_totp_for_user(secret):
    totp = pyotp.TOTP(secret, interval=custom_time_window)
    return totp.now()


# Verify OTP for a specific user with the custom time window
def verify_totp_for_user(secret, user_otp):
    totp = pyotp.TOTP(secret, interval=custom_time_window)
    return totp.verify(user_otp)


@app.post("/signup")
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        return JSONResponse(content={"status": "failed", "message": "Email already registered"})

    secret = generate_totp_secret()
    otp = generate_totp_for_user(secret)
    print(datetime.now())
    print("generated otp: ", otp)
    created_user = create_user(db, user, secret)
    message = MessageSchema(
        subject="Verify Your Email",
        recipients=[user.email],
        body=f'Here is the Your OTP: {otp}\nValid for 3 mins.',
        subtype="html"
    )
    # await mail.send_message(message)

    return {"status": "success", "message": "User created successfully", "data": created_user}


@app.post("/verify-otp")
async def verify_otp(UserDetails: dict = Body(...), db: Session = Depends(get_db)):
    user_email = UserDetails.get("email")
    user_otp = UserDetails.get("user_otp")
    db_user = db.query(User).filter(User.email == user_email).first()
    if not db_user:
        return JSONResponse(content={"status": "failed", "message": "User not exist"})
    # Verify the OTP using the stored secret key
    totp = pyotp.TOTP(db_user.totp_secret, interval=custom_time_window)
    print(datetime.now())
    if totp.verify(user_otp):
        return {"status": "success", "message": "OTP verification successful"}
    else:
        return JSONResponse(content={"status": "failed", "message": "Invalid OTP"})


# Token route
@app.post("/login")
async def login(user_login: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_login.email).first()
    if not user or not checkpw(user_login.password.encode("utf-8"), user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # if user.expired_on <= datetime.now():
    #     user.status = False
    #     db.commit()
    try:
        access_token = create_access_token(data={"sub": user.email})
        return {
            "status": "success",
            "message": "Login successful",
            "data": {
                "id": user.id,
                "email": user.email,
                "access_token": access_token,
                "token_type": "bearer",
                "plan_status": user.status,
                "duration": user.plan_expired_on
            }
        }
    except Exception as e:
        return JSONResponse(content={"status": "failed", "message": "Internal Server Error"})


@app.put("/update-plan/{user_id}")
async def update_plan(user_id: str, PlanUpdate: dict = Body(...), db_session: Session = Depends(get_db)):
    db = db_session.query(User).filter(User.id == user_id).first()
    request_current_time = PlanUpdate.get("currentDate")
    request_plan_expiry = PlanUpdate.get("expiryDate")
    request_pId = PlanUpdate.get("productId")
    request_price = PlanUpdate.get("price")
    request_period = PlanUpdate.get("period")
    request_cd = PlanUpdate.get("currencyCode")
    request_plan = PlanUpdate.get("productId").lower()

    try:
        if request_plan == "free" and not db.trial:
            db.plan_created_on = datetime.fromisoformat(request_current_time.rstrip('Z')).replace(tzinfo=timezone.utc)
            db.plan_expired_on = datetime.fromisoformat(request_plan_expiry.rstrip('Z')).replace(tzinfo=timezone.utc)
            db.product_id = request_pId
            db.period = request_period
            db.price = request_price
            db.currency_code = request_cd
            db.plan = request_plan
            db.status = True
            db.trial = True
            db_session.add(db)

        elif request_plan != "free":
            db.plan_created_on = datetime.fromisoformat(request_current_time.rstrip('Z')).replace(tzinfo=timezone.utc)
            db.plan_expired_on = datetime.fromisoformat(request_plan_expiry.rstrip('Z')).replace(tzinfo=timezone.utc)
            db.product_id = request_pId
            db.period = request_period
            db.price = request_price
            db.currency_code = request_cd
            db.plan = request_plan
            db.status = True
            db.trial = True
            db_session.add(db)

        else:
            return {"status": "failed", "message": "Please Subscribe to the Plan with month or year!"}

        db_session.commit()
        db_session.refresh(db)
        return {"status": "success","message": "Plan Updated successfully"}
    except Exception as e:
        return {"status": "failed", "message": f"error while updating the plan: {e}"}


@app.get("/user-access/{user_id}")
async def user_access(user_id: int, db_session: Session = Depends(get_db)):
    db = db_session.query(User).filter(User.id == user_id).first()
    # print(db.expired_on, datetime.now())
    if db:
        if db.plan_expired_on is not None and db.plan_expired_on >= datetime.now():
            return {
                "status": "success",
                "plan_status": db.status,
                "duration": db.plan,
                "expired_on": db.plan_expired_on
            }
        elif db.plan_expired_on is not None and db.plan_expired_on <= datetime.now():
            db.status = False
            db_session.commit()
            return JSONResponse(content={"status": "failed", "message": "Please Subscribe to the Plan!"})
        else:
            return JSONResponse(content={"status": "failed", "message": "You have not Subscribe Yet!"})
    else:
        return JSONResponse(content={"status": "failed", "message": "Purchase the Plan!"})


UPLOADS_DIR = "uploads"
GENERATES_DIR = "generates"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(GENERATES_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="static")
app.mount("/generates", StaticFiles(directory="generates"), name="static")


@app.post("/vision")
async def create_upload_file(data: dict, db_session: SessionLocal = Depends(get_db)):
    try:
        chat_id = data.get("chat_id")
        if chat_id is None:
            chat = ChatData(user_id=data.get("user_id"))
            chat.created_at = datetime.utcnow()
            db_session.add(chat)
            db_session.commit()
            chat_id = chat.id
            images = data.get("image")
            user_message = data.get("user_message")

        uploaded_files = []
        # Decode Base64 string to bytes
        image_data = base64.b64decode(images)

        filename = f"{uuid.uuid4()}.png"

        # Save the decoded image to the server
        with open(os.path.join(UPLOADS_DIR, filename), "wb") as buffer:
            buffer.write(image_data)

        # Save file information to the database
        db_file = UploadedFile(chat_id=chat_id, filename=filename, message=user_message, content_type="Image/Reader",
                               created_at=datetime.utcnow())
        chat = db_session.query(ChatData).filter(ChatData.id == db_file.chat_id).first()
        chat.title = user_message
        chat.content_type = "Image/Reader"
        chat.messages = json.dumps([
            {
                "role": "user",
                "content": user_message
            }
        ])
        db_session.add(db_file, chat)
        db_session.commit()

        file_url = f"{FILE_URL}/uploads/{filename}"  # Replace with your actual server domain

        uploaded_files.append({
            "id": db_file.id,
            "filename": filename,
            "user_message": user_message,
            "content_type": "Image/Reader",
            "file_url": file_url,
            "convertation_id": chat_id
        })
        
        return StreamingResponse(get_response_on_upload(user_message, images, chat, db_file, db_session),
                                 media_type='application/json')

    except KeyError:
        return JSONResponse(content={"status": "failed", "message": "Invalid request. 'images' key not found."}, status_code=400)

    except Exception as e:
        print(e)
        return JSONResponse(content={"status": "failed", "message": "Internal Server Error"}, status_code=500)


# New Chat
@app.post("/stream/chat")
async def openai_new_chat_completion(request: Request, db: SessionLocal = Depends(get_db)):
    try:
        json_data = await request.json()
        user_message = json_data.get("user_message", "")
        context_type = json_data.get("type", "")
        user_id = json_data.get("user_id", "")
        title = json_data.get("title", "")

        if context_type == "Image/Generate":
            reply, conversation, file_url = generate_image(db, None, user_message, user_id, title)
            return JSONResponse(
                content={
                    "status": "success",
                    "data": markdown(convert_newlines_to_br(reply)),
                    "text": reply,
                    "conversation": conversation.id,
                    "file_url": file_url
                },
                status_code=200,
            )
        else:
            return StreamingResponse(stream_logic(db, None, user_message, user_id, title),
                                     media_type='application/json')

    except Exception as e:
        return JSONResponse(
            content={
                "status": "failed",
                "message": str(e)
            },
            status_code=500,
        )


# Existing Chat
@app.post("/stream/chat/{chat_id}")
async def openai_chat_completion(request: Request, db: SessionLocal = Depends(get_db),
                                 chat_id: Optional[Union[int, None]] = None):
    try:
        json_data = await request.json()
        user_message = json_data.get("user_message", "")
        images = json_data.get("images")
        context_type = json_data.get("type", "")

        if images is not None and len(images) > 0:
            # Decode Base64 string to bytes
            image_data = base64.b64decode(images)
            filename = f"{uuid.uuid4()}.png"  # You can use a different extension if needed
            # Save the decoded image to the server
            with open(os.path.join(UPLOADS_DIR, filename), "wb") as buffer:
                buffer.write(image_data)
            # Save file information to the database
            db_file = UploadedFile(chat_id=chat_id, filename=filename, content_type="Image/Reader")
            db.add(db_file)
            db.commit()

            file_url = f"{FILE_URL}/uploads/{filename}"
            data = await get_vision(db_file.id, db_session=db)
            return data

        if context_type == "Image/Generate":
            reply, conversation, file_url = generate_image(db, chat_id, user_message)
            return JSONResponse(
                content={
                    "status": "success",
                    "data": markdown(convert_newlines_to_br(reply)),
                    "text": reply,
                    "conversation": conversation.id,
                    "file_url": file_url
                },
                status_code=200,
            )
        else:
            conversation = db.query(ChatData).filter(ChatData.id == chat_id).first()
            if conversation.content_type == "Image/Reader":
                return StreamingResponse(
                    get_desc_from_image_again(conversation, db, UploadedFile, chat_id, UPLOADS_DIR, user_message))

            else:
                return StreamingResponse(stream_logic(db, chat_id, user_message), media_type='application/json')

    except Exception as e:
        return JSONResponse(
            content={
                "status": "failed",
                "message": str(e)
            },
            status_code=500,
        )



@app.get("/chat-data/{chat_id}")
async def read_chat_data_by_id(chat_id: int, db_session: Session = Depends(get_db)):
    chat_data = db_session.query(ChatData).filter(ChatData.id == chat_id).order_by(desc(ChatData.created_at)).first()
    if not chat_data:
        raise HTTPException(status_code=404, detail=f"Chat data not found for id {chat_id}")
    serialized_chat_data = jsonable_encoder(chat_data)

    return JSONResponse(
        content={
            "status": "success",
            "message": "Chat data by id successful",
            "content-type": chat_data.content_type,
            "data": serialized_chat_data
        },
        status_code=200,
    )


@app.get("/agents")
async def agents():
    data = [
        {"agentName": "DALL-E", "agentTitle": "Whatever you imagine, DALL·E creates",
         "initalQuestion": ""},
        {"agentName": "Rewriting", "agentTitle": "Create unique rewrites with high text authenticity",
         "initalQuestion": "I want to rewrite my context."},
        {"agentName": "Writing", "agentTitle": "Fight writer's block and finish your text",
         "initalQuestion": "Help me writing something for my context"},
        {"agentName": "Essay", "agentTitle": "Have a well-structured essay written especially for you",
         "initalQuestion": "Write an essay for on my context"},
        {"agentName": "Math", "agentTitle": "Have any question solved in a flash",
         "initalQuestion": "Solve this maths equation"},
        {"agentName": "History", "agentTitle": "Know about any historical event/date.",
         "initalQuestion": "I want to know about this historical event."},
        {"agentName": "Coffee Date", "agentTitle": "Get some exciting idea for your next coffee date.",
         "initalQuestion": "How can I make my coffee date more interesting"},
        {"agentName": "Facebook Posts", "agentTitle": "Generate great posts for your social account.",
         "initalQuestion": "I want you to generate some posts on  the context I'll provide"},
        {"agentName": "Text Check", "agentTitle": "Check any text for spelling and grammatical mistakes",
         "initalQuestion": "Check this text for me"},
        {"agentName": "Text Style", "agentTitle": "Have any question solved in a flash",
         "initalQuestion": "Check the text style for me"},
    ]
    serialized_chat_data = jsonable_encoder(data)

    return JSONResponse(
        content={
            "status": "success",
            "message": "Agents data extraction successful",
            "data": serialized_chat_data
        },
        status_code=200,
    )


@app.get("/chat-data-user/{user_id}")
async def read_chat_data_by_user_id(user_id: int, db_session: Session = Depends(get_db)):
    chat_data = db_session.query(ChatData).filter(ChatData.user_id == user_id).order_by(desc(ChatData.created_at)).all()
    if not chat_data:
        raise HTTPException(status_code=404, detail=f"Chat data not found for user id {user_id}")
    serialized_chat_data = jsonable_encoder(chat_data)

    return JSONResponse(
        content={
            "status": "success",
            "message": "Chat data by user id successful",
            "data": serialized_chat_data
        },
        status_code=200,
    )


@app.get("/chat/{chat_id}")
async def read_chat(chat_id: int, db_session: Session = Depends(get_db)):
    chat = db_session.query(ChatData).filter(ChatData.id == chat_id).order_by(desc(ChatData.created_at)).first()
    if not chat:
        raise HTTPException(status_code=404, detail=f"Chat not found")

    chat = jsonable_encoder(chat)
    file = db_session.query(UploadedFile).filter(UploadedFile.chat_id == chat_id).first()

    messages = []
    for message in json.loads(chat.get("messages")):
        messages.append({
            "content": markdown(convert_newlines_to_br(message.get("content"))),
            "text": (message.get("content")),
            "role": message.get("role"),
            "file_url": message.get("file_url")
        })

    file_url = ""
    if file is not None:
        file_url = f"{FILE_URL}/uploads/{file.filename}"
    return JSONResponse(
        content={
            "status": "success",
            "message": "Chat data successfully received",
            "data": {
                "id": chat.get("id"),
                "content-type": chat.get("content_type"),
                "created_at": chat.get("created_at"),
                "user_id": chat.get("user_id"),
                "messages": messages,
                "file_url": file_url
            }
        },
        status_code=200,
    )


# Delete chat/id
@app.delete("/chat/{chat_id}")
async def delete_chat(request: Request, chat_id: int, db: Session = Depends(get_db)):
    try:
        json_data = await request.json()
        user_id = json_data.get("user_id")

        chat = db.query(ChatData).filter(
            ChatData.user_id == user_id,
            ChatData.id == chat_id
        ).first()

        if chat:
            # Delete messages associated with the chat
            db.query(ChatData).filter(ChatData.id == chat_id).delete(synchronize_session=False)
            # Delete the chat
            db.delete(chat)
            db.commit()

            return {"status": "success", "message": "Chat and conversation history deleted."}
        else:
            return  JSONResponse(content={"status": "failed", "message": "Chat not found for the specified user"})
    except Exception as e:
        return JSONResponse(
            content={
                "status": "failed",
                "message": str(e)
            },
            status_code=500,
        )


data = [
    {"question": "Can you provide tips for taking professional-quality photos with my phone? Any settings I should change?", "category": "General"},
    {"question": "What’s the most efficient way to learn a new language?", "category": "General"},
    {"question": "Can you help me edit this email to make it sound more professional", "category": "General"},
    {"question": "I’m feeling anxious about an upcoming job interview, any advice or tips?", "category": "General"},
    {"question": "What are some strategies to improve memory and concentration?", "category": "General"},
    {"question": "Let’s role-play a scenario where I’m a detective and you’re my AI assistant. Ready?", "category": "General"},
    {"question": "How can I effectively negotiate a higher salary during a job interview?", "category": "General"},
    {"question": "What are the health benefits of a Mediterranean diet?", "category": "Health"},
    {"question": "What are some creative dinner ideas using just the ingredients in my pantry? I can provide a picture of my current ingredients", "category": "Health"},
    {"question": "How can I make a healthy meal plan on a budget?", "category": "Health"},
    {"question": "What are the most effective exercises for overall cardiovascular health?", "category": "Health"},
    {"question": "Can you design a 7-day vegetarian meal plan that's high in protein?", "category": "Health"},
    {"question": "What are the health benefits and potential risks of intermittent fasting?", "category": "Health"},
    {"question": "How much sleep do I really need and how can I improve sleep quality", "category": "Health"},
    {"question": "What’s the latest in technology right now?", "category": "Technology"},
    {"question": "What are the emerging trends in artificial intelligence for the next decade?", "category": "Technology"},
    {"question": "How can blockchain technology transform the banking industry?", "category": "Technology"},
    {"question": "What are the key differences between 5G and 4G mobile technology, and how will 5G impact us?", "category": "Technology"},
    {"question": "How are advancements in AI expected to transform industries like healthcare or finance in the next decade?", "category": "Technology"},
    {"question": "Can you help me create a business plan for a startup?", "category": "Business"},
    {"question": "What is the best business to start in 2024?", "category": "Business"},
    {"question": "How do I start investing in stocks as a beginner?", "category": "Business"},
    {"question": "What are the key elements of a successful marketing strategy for a small business?", "category": "Business"},
    {"question": "Can you suggest some innovative business models for e-commerce?", "category": "Business"},
    {"question": "I have $100 dollars, help me turn this into thousands.", "category": "Business"},
    {"question": "What are the most important financial metrics for startups to monitor?", "category": "Business"},
    {"question": "What’s the philosophical significance of free will versus determinism?", "category": "Education"},
    {"question": "Explain the theory of relativity as if you’re talking to a high school student", "category": "Education"},
    {"question": "Can you summarize the plot of “1984” by George Orwell", "category": "Education"},
    {"question": "Can you provide a brief history of the Roman Empire?", "category": "Education"},
    {"question": "What are the key differences between solar and wind energy", "category": "Education"},
    {"question": "Explain the significance of the Renaissance in shaping modern thought", "category": "Education"},
    {"question": "What are the best ways to set and achieve career goals?", "category": "Personal Development"},
    {"question": "How can I improve my public speaking and presentation skills?", "category": "Personal Development"},
    {"question": "What are the steps to developing a strong personal brand?", "category": "Personal Development"},
    {"question": "Can you guide me on how to expand my professional network effectively?", "category": "Personal Development"},
    {"question": "What are some techniques for improving self-discipline and focus?", "category": "Personal Development"}
]


@app.get("/get-questions-data")
async def get_questions_data():
    return {"data": data}


@app.get("/stream")
async def stream_logic_call(request: Request):
    return StreamingResponse(get_openai_generator(), media_type='application/json')


def get_openai_generator():
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user', 'content': "how to take the good picture?"}
        ],
        temperature=1,
        stream=True
    )

    for event in completion:
        # print(event.choices[0])
        current_response = event.choices[0].delta.content
        # if current_response is not None:
        if current_response:
            text = current_response
            yield convert_newlines_to_br(text)


@app.delete("/delete-user/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        chat_data_list = db.query(ChatData).filter(ChatData.user_id == user_id).all()

        for chat_data in chat_data_list:

            uploaded_files = db.query(UploadedFile).filter(UploadedFile.chat_id == chat_data.id).all()

            for uploaded_file in uploaded_files:
                db.delete(uploaded_file)

            db.delete(chat_data)

        db.delete(user)
        db.commit()

        return JSONResponse(content={"message": "User Deleted Successfully", "status": "success"})
    except Exception as e:
        return JSONResponse(content={"message": "User Not Deleted", "status": "failed"})


@app.put("/change-password/{user_id}")
async def change_password(user_id: int, Password: dict = Body(...), db: Session = Depends(get_db)):
    try:
        previous_passwd = Password.get("previous-passwd")
        new_passwd = Password.get("new-passwd")
        confirm_passwd = Password.get("confirm-passwd")
        if new_passwd == confirm_passwd:
            user = db.query(User).filter(User.id == user_id).first()
            if checkpw(previous_passwd.encode("utf-8"), user.hashed_password):
                try:
                    user.hashed_password = hashpw(confirm_passwd.encode('utf-8'), gensalt())
                    db.add(user)
                    db.commit()
                    return JSONResponse(content={"status": "success", "message": "Password Updated!"})
                except Exception as e:
                    print(e)
                    return JSONResponse(content={"status": "failed", "message": "There is issue in updating the Password!"})
            else:
                return JSONResponse(content={"status": "failed", "message": "Your Password not matched to the Existing Password!"})
        else:
            return JSONResponse(content={"status": "failed","message": "New and Confirm Password Not Matched!"})
    except Exception as e:
        return JSONResponse(content={"status": "failed", "message": "Credentials are not correct!"})


@app.post("/forgot-passwd")
async def forgot_passwd(Email: dict = Body(...), db: Session = Depends(get_db)):
    email = Email.get("email")
    print(email)
    user = db.query(User).filter(User.email == email).first()
    print(user)
    if user:
        secret = generate_totp_secret()
        otp = generate_totp_for_user(secret)
        print(datetime.now())
        print("generated otp: ", otp)
        print(user.totp_secret)
        print(secret)
        user.totp_secret = secret
        message = MessageSchema(
            subject="Verify Your Email",
            recipients=[user.email],
            body=f'Here is the Your OTP: {otp}<br>Valid for 3 mins.',
            subtype="html"
        )
        # await mail.send_message(message)
        db.add(user)
        db.commit()
        return {"status": "success", "message": "OTP sent Successfully!"}
    else:
        return JSONResponse(content={"status": "failed", "message": "User not existed"})


@app.post("/update-passwd")
async def update_passwd(Details: dict = Body(...), db: Session = Depends(get_db)):
    email = Details.get("email")
    otp = Details.get("otp")
    new_passwd = Details.get("new_passwd")
    confirm_passwd = Details.get("confirm_passwd")
    user = db.query(User).filter(User.email == email).first()
    if user:
        totp = pyotp.TOTP(user.totp_secret, interval=custom_time_window)
        print(datetime.now())
        if totp.verify(otp):
            if new_passwd == confirm_passwd:
                user.hashed_password = hashpw(confirm_passwd.encode('utf-8'), gensalt())
                db.add(user)
                db.commit()
                return JSONResponse(content={"status": "success", "message": "Password Updated!"})
            else:
                return JSONResponse(content={"status": "failed", "message": "New Password and Confirm Password not matched"})
        else:
            return JSONResponse(content={"status": "failed", "message": "Invalid OTP"})
