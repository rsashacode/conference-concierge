from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional

from src.prompts import GUARDRAIL_CLASSIFIER_PROMPT


GUARDRAIL_MODEL = "gpt-4o-mini"
INPUT_REJECT_MESSAGE = "Please keep your message on the topic of conference schedule planning."
OUTPUT_REJECT_MESSAGE = "I can't provide that. How can I help with your conference schedule?"


class GuardrailResponse(BaseModel):
    allowed: bool = Field(description="Whether the message is allowed.")
    message: str = Field(description="The message to show the user if the message is not allowed.")


def check_input(user_message: str) -> tuple[bool, str]:
    """
    Check if user input is appropriate for the conference concierge.
    Returns (allowed, rejection_message). If allowed, rejection_message is empty.
    """
    if not user_message or not user_message.strip():
        return False, INPUT_REJECT_MESSAGE
    try:
        client = OpenAI()
        r = client.chat.completions.parse(
            model=GUARDRAIL_MODEL,
            messages=[
                {"role": "system", "content": GUARDRAIL_CLASSIFIER_PROMPT},
                {"role": "user", "content": f"Message: {user_message[:2000]}"},
            ],
            response_format=GuardrailResponse,
        )
        result = r.choices[0].message
        parsed: GuardrailResponse = result.parsed if result.parsed else GuardrailResponse(allowed=False, message=INPUT_REJECT_MESSAGE)
        return parsed.allowed, parsed.message
    except Exception:
        return True, INPUT_REJECT_MESSAGE


def check_output(assistant_reply: str) -> tuple[bool, str]:
    """
    Check if assistant output is safe to show. Returns (safe, display_text).
    If not safe, display_text is a generic fallback.
    """
    if not assistant_reply or not assistant_reply.strip():
        return True, OUTPUT_REJECT_MESSAGE
    try:
        client = OpenAI()
        r = client.chat.completions.parse(
            model=GUARDRAIL_MODEL,
            messages=[
                {"role": "system", "content": GUARDRAIL_CLASSIFIER_PROMPT},
                {"role": "user", "content": f"Reply: {assistant_reply[:2000]}"},
            ],
            response_format=GuardrailResponse,
        )
        result = r.choices[0].message
        parsed: GuardrailResponse = result.parsed if result.parsed else GuardrailResponse(allowed=False, message=OUTPUT_REJECT_MESSAGE)
        return parsed.allowed, parsed.message
    except Exception:
        return True, OUTPUT_REJECT_MESSAGE
