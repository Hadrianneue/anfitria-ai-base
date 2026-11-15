import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Query

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

from app.resilience import classify_with_resilience
from app.schemas import IncomingMessage
from app.tickets import create_ticket, init_db, list_tickets

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Starts database"""
    init_db()
    yield

app = FastAPI(
    title="Anfitria AI",
    description="Automated WhatsApp triage and ticket creation for vacation rentals.",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/simulate")
def simulate_message(
    message: IncomingMessage,
    provider: str | None = Query(
        default=None,
        description="Optional override: gemini | deepseek | llamacpp | pytorch",
    ),
):
    """simulates incoming whatsapp message"""
    decision, used_provider = classify_with_resilience(
        message,
        primary_name=provider,
    )

    ticket_id = None
    if decision.needs_ticket:
        ticket_id = create_ticket(message, decision, used_provider)

    return {
        "provider_used": used_provider,
        "ticket_created": decision.needs_ticket,
        "ticket_id": ticket_id,
        "decision": decision.model_dump(),
    }

@app.get("/tickets")
def get_tickets(
    status: str | None = Query(
        default=None,
        description="Filter by status: 'open' or 'resolved'"
    )
):
    """lists stored tickets from db"""
    ticket = list_tickets(status=status)
    return {"count": len(ticket), "tickets": ticket}
