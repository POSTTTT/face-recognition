"""FastAPI surface for the team backend, plus a test page at /.  Run: uvicorn api:app --port 8000"""
import threading
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from face import FaceModule

app = FastAPI(title="Face recognition module")
face = FaceModule()
camera = threading.Lock()  # one camera, one scan at a time
latest = {"jpg": None}  # last annotated frame of the running scan, for the test page


def fresh_show():
    latest["jpg"] = None  # don't let the page show the previous scan's last frame
    return show


def show(frame):
    latest["jpg"] = cv2.imencode(".jpg", frame)[1].tobytes()


class Enroll(BaseModel):
    name: str
    role: str = "user"


@app.post("/enroll")
def enroll(body: Enroll):
    with camera:
        try:
            return {"user_id": face.enroll(body.name, body.role, show=fresh_show())}
        except RuntimeError as e:
            raise HTTPException(422, str(e))


@app.post("/recognize")
def recognize():
    """result: granted | unknown | spoof_suspected | no_face. The backend decides whether to open the door."""
    with camera:
        try:
            return face.recognize(show=fresh_show())
        except RuntimeError as e:
            raise HTTPException(503, str(e))


@app.get("/logs")
def logs():
    rows = face.conn.execute(
        "SELECT l.timestamp, u.name, l.result, l.score FROM access_logs l "
        "LEFT JOIN users u ON u.id = l.user_id ORDER BY l.id DESC LIMIT 20"
    ).fetchall()
    return [dict(zip(("timestamp", "name", "result", "score"), r)) for r in rows]


@app.get("/preview.jpg")
def preview():
    if latest["jpg"] is None:
        raise HTTPException(404)
    return Response(latest["jpg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/", response_class=HTMLResponse)
def page():
    return (Path(__file__).parent / "index.html").read_text()
