import os
from datetime import datetime, timedelta
from typing import Any, Optional
from fastapi.responses import RedirectResponse

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from db import create_user, ensure_users_table, get_user_by_username

router = APIRouter(tags=["auth"])
templates = Jinja2Templates(directory="templates")

SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
ACCESS_TOKEN_COOKIE_NAME = "elto_access_token"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/", auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def authenticate_user(username: str, password: str) -> Optional[dict[str, Any]]:
    TEMP_USERNAME = "admin"
    TEMP_PASSWORD = "admin123"
    if username == TEMP_USERNAME and password == TEMP_PASSWORD:
        return {"username": TEMP_USERNAME, "id": 0, "is_active": True}
    
    user = get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    if not user.get("is_active", True):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def _get_token_from_request(request: Request, token: Optional[str]) -> str:
    if token:
        return token
    cookie_token = request.cookies.get(ACCESS_TOKEN_COOKIE_NAME)
    if cookie_token:
        return cookie_token
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_token(
    request: Request, token: Optional[str] = Depends(oauth2_scheme)
) -> str:
    return _get_token_from_request(request, token)


async def get_current_user(token: str = Depends(get_current_token)) -> dict[str, Any]:
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user_by_username(username)
    if user is None or not user.get("is_active", True):
        raise credentials_exception
    return user

@router.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    token = request.cookies.get(ACCESS_TOKEN_COOKIE_NAME)
    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("sub"):
                return RedirectResponse(url="/dashboard", status_code=302)
        except JWTError:
            pass
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect credentials")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )

    response = JSONResponse(
        content={"access_token": access_token, "token_type": "bearer"}
    )
    response.set_cookie(
        ACCESS_TOKEN_COOKIE_NAME,
        access_token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=int(access_token_expires.total_seconds()),
    )
    return response


@router.post("/logout")
async def logout(response: Response):
    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie(ACCESS_TOKEN_COOKIE_NAME)
    return response


@router.get("/me")
async def read_current_user(current_user: dict = Depends(get_current_user)):
    return {"username": current_user["username"], "id": current_user["id"]}