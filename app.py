from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List
from database import TicketDatabase
import sqlite3
import uvicorn

app = FastAPI(title="Ticket Dashboard")

templates = Jinja2Templates(directory="templates")

class StatusUpdate(BaseModel):
    ticket_id: int
    status: str

class Ticket(BaseModel):
    id: int
    user_name: str
    email: str
    query: str
    status: str
    created_at: str
    resolved_at: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("tickets.html", {"request": request})

@app.get("/api/tickets")
async def get_tickets(status: Optional[str] = None):
    try:
        db = TicketDatabase()
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if status and status != 'all':
            cursor.execute('SELECT * FROM tickets WHERE status = ? ORDER BY created_at DESC', (status,))
        else:
            cursor.execute('SELECT * FROM tickets ORDER BY created_at DESC')
            
        tickets = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return tickets
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update_status")
async def update_status(data: StatusUpdate):
    try:
        db = TicketDatabase()
        db.update_ticket_status(data.ticket_id, data.status)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)