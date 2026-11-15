from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field

class MessageCategories(str, Enum):
    """Possible message Categories"""
    GENERAL_QUESTION = "general_question"
    RESERVATION_PRICING = "reservation_pricing"
    URGENT_MAINTENANCE = "urgent_maintenance"
    OTHERS = "others"

class ReservationDetails(BaseModel):
    """Extract possible provided information"""
    check_in: Optional[str] = Field (
        default=None, description="Desired date/stay/check-in (e.g:2026-10-02) or weekend"
    )
    check_out: Optional[str] = Field (
        default=None,
        description="Desired check-out date if mentioned."
    )
    guests: Optional[int] = Field (
        default=None,
        description="number o total guests/people mentioned"
    )

class AiIntent(BaseModel):
    """Required information llms must provide back"""
    category: MessageCategories = Field(
        description="Primary message intent classification"
    )
    needs_ticket: bool = Field(
        description="Should be True if guest requests action (reservation, negotiation, or maintenance) otherwise False if it was a simple already answered question."
    )
    priority: Literal["low", "medium", "sarah"] = Field(
        description="'sarah' for emergencies or problems in the accomodation, 'medium' for reservations, 'low' for general questions"
    )
    internal_summary: str = Field(
        description="short phrase summary for guest to quickly identify on the ticket panel"
    )
    guest_response: str = Field(
        description="Natural, cordial and objective message to be sent to guest on whatsapp."
    )
    reservation_data: Optional[ReservationDetails] = Field(
        default=None,
        description="Only required if guest talks about dates or guest number"
    )

class IncomingMessage(BaseModel):
    """Represents a message received on whatsapp"""
    phone: str
    guest_name: str = "Guest"
    text: str
