import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, field_validator, model_validator


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str | None
    display_name: str | None
    is_synthetic: bool
    created_at: datetime


class RegisterRequest(BaseModel):
    email: EmailStr
    display_name: str
    password: str
    confirm_password: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @model_validator(mode="after")
    def passwords_match(self) -> "RegisterRequest":
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class RegisterResponse(BaseModel):
    user_id: uuid.UUID
    email: str
    display_name: str | None
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class LoginResponse(BaseModel):
    user_id: uuid.UUID
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MessageResponse(BaseModel):
    message: str
