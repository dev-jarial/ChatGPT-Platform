# models/user.py
from DB.database import Base
from sqlalchemy import Column, Integer, String, Sequence, Text, ForeignKey, DateTime, Boolean
from sqlalchemy.sql import func


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, Sequence("user_id_seq"), primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    hashed_password = Column(String)
    totp_secret = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)
    plan_created_on = Column(DateTime(timezone=True))
    plan_expired_on = Column(DateTime(timezone=True))
    #, default=func.datetime(func.julianday('now') + 30), nullable=True
    product_id = Column(String, nullable=True)
    period = Column(String, nullable=True)
    price = Column(String, nullable=True)
    currency_code = Column(String, nullable=True)
    plan = Column(String, nullable=True)
    status = Column(Boolean, default=False)
    trial = Column(Boolean, default=False)


class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, Sequence("uploaded_files_id_seq"), primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chat_data.id"))
    filename = Column(String, unique=True, index=True)
    message = Column(String)
    content_type = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)


class ChatData(Base):
    __tablename__ = "chat_data"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String,nullable=True)
    content_type = Column(String)
    messages = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)
