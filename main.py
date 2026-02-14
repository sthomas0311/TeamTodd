
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import databases
import sqlalchemy
import boto3
from botocore.exceptions import NoCredentialsError
from openai import OpenAI

# Load environment variables
load_dotenv()

# --- Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL")

# Validate essential environment variables
if not all([DATABASE_URL, OPENROUTER_API_KEY, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, CLOUDFLARE_ACCOUNT_ID, R2_BUCKET_NAME, R2_PUBLIC_URL]):
    raise ValueError("One or more essential environment variables are not set. Please check your .env file.")

# --- Database setup ---
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# Define database tables
users = sqlalchemy.Table(
    "users",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column("username", sqlalchemy.String, unique=True, index=True),
    sqlalchemy.Column("email", sqlalchemy.String, unique=True, index=True),
    sqlalchemy.Column("hashed_password", sqlalchemy.String),
)

posts = sqlalchemy.Table(
    "posts",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, index=True),
    sqlalchemy.Column("author_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id")),
    sqlalchemy.Column("content", sqlalchemy.String),
    sqlalchemy.Column("image_url", sqlalchemy.String, nullable=True),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=sqlalchemy.func.now()),
)

engine = sqlalchemy.create_engine(DATABASE_URL)
metadata.create_all(engine)

# --- FastAPI app setup ---
app = FastAPI(title="Social Media Backend", description="FastAPI backend for a social media application.")

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

# --- Pydantic Models ---
class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int

    class Config:
        from_attributes = True # updated from orm_mode = True

class PostBase(BaseModel):
    content: str
    image_url: Optional[str] = None

class PostCreate(PostBase):
    author_id: int

class Post(PostBase):
    id: int
    author_id: int
    created_at: str # will be datetime in db, but string for Pydantic output

    class Config:
        from_attributes = True # updated from orm_mode = True

class AIDraftRequest(BaseModel):
    prompt: str
    max_length: int = 200

# --- AWS S3 (R2) Client Setup ---
r2_client = boto3.client(
    service_name='s3',
    endpoint_url=f"https://{CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name='auto'  # R2 does not use AWS regions in the same way, 'auto' is common practice
)

# --- OpenAI (OpenRouter) Client Setup ---
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- Helper function for authentication (placeholder for now) ---
async def get_current_user():
    # In a real app, this would involve token verification, fetching user from DB, etc.
    # For now, we'll return a dummy user.
    return {"id": 1, "username": "testuser"}

# --- Routes ---

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Social Media Backend!"}

# --- User Endpoints ---

@app.post("/users/", response_model=User)
async def create_user(user: UserCreate):
    # In a real app, password should be hashed
    query = users.insert().values(username=user.username, email=user.email, hashed_password=user.password)
    last_record_id = await database.execute(query)
    return {**user.dict(), "id": last_record_id}

@app.get("/users/", response_model=List[User])
async def read_users():
    query = users.select()
    return await database.fetch_all(query)

@app.get("/users/{user_id}", response_model=User)
async def read_user(user_id: int):
    query = users.select().where(users.c.id == user_id)
    user = await database.fetch_one(query)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.put("/users/{user_id}", response_model=User)
async def update_user(user_id: int, user: UserCreate): # Re-using UserCreate for update, could be a separate model
    # In a real app, password would be hashed and carefully handled
    query = users.update().where(users.c.id == user_id).values(username=user.username, email=user.email, hashed_password=user.password)
    await database.execute(query)
    return {**user.dict(), "id": user_id}

@app.delete("/users/{user_id}", response_model=dict)
async def delete_user(user_id: int):
    query = users.delete().where(users.c.id == user_id)
    deleted_rows = await database.execute(query)
    if not deleted_rows:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted successfully"}

# --- Post Endpoints ---

@app.post("/posts/", response_model=Post)
async def create_post(post: PostCreate):
    query = posts.insert().values(author_id=post.author_id, content=post.content, image_url=post.image_url)
    last_record_id = await database.execute(query)
    created_post = await database.fetch_one(posts.select().where(posts.c.id == last_record_id))
    return created_post

@app.get("/posts/", response_model=List[Post])
async def read_posts():
    query = posts.select()
    return await database.fetch_all(query)

@app.get("/posts/{post_id}", response_model=Post)
async def read_post(post_id: int):
    query = posts.select().where(posts.c.id == post_id)
    post = await database.fetch_one(query)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return post

@app.put("/posts/{post_id}", response_model=Post)
async def update_post(post_id: int, post: PostCreate): # Re-using PostCreate for update
    query = posts.update().where(posts.c.id == post_id).values(author_id=post.author_id, content=post.content, image_url=post.image_url)
    await database.execute(query)
    updated_post = await database.fetch_one(posts.select().where(posts.c.id == post_id))
    return updated_post

@app.delete("/posts/{post_id}", response_model=dict)
async def delete_post(post_id: int):
    query = posts.delete().where(posts.c.id == post_id)
    deleted_rows = await database.execute(query)
    if not deleted_rows:
        raise HTTPException(status_code=404, detail="Post not found")
    return {"message": "Post deleted successfully"}

# --- AI Service (Post Drafting) Endpoint ---

@app.post("/ai/draft_post/")
async def draft_post_with_ai(request: AIDraftRequest):
    try:
        completion = openai_client.chat.completions.create(
            model="google/gemini-pro",  # Specify the Gemini Pro model
            messages=[
                {"role": "system", "content": "You are a helpful assistant for drafting social media posts. Be concise and engaging."},
                {"role": "user", "content": f"Draft a social media post based on this idea: {request.prompt}. Keep it under {request.max_length} characters. Start directly with the post content, no conversational filler."},
            ]
        )
        ai_draft = completion.choices[0].message.content
        return {"draft": ai_draft}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI drafting failed: {str(e)}")

# --- File Upload (R2) Endpoint ---

@app.post("/upload-image/")
async def upload_image_to_r2(file: UploadFile = File(...)):
    try:
        file_name = file.filename
        file_content = await file.read()

        r2_client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=file_name,
            Body=file_content,
            ContentType=file.content_type
        )
        image_url = f"{R2_PUBLIC_URL}/{file_name}"
        return {"filename": file_name, "url": image_url}
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="R2 credentials not available.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
