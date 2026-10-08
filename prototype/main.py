from fastapi import FastAPI
from routes import score, winner, debate

app = FastAPI(title="Team Logic API")

# Register all routes
app.include_router(score.router)
app.include_router(winner.router)
app.include_router(debate.router)

@app.get("/")
def home():
    return {"message": "Team Logic API is running!"}