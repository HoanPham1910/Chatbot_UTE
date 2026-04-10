"""
Google Calendar Service
"""

import datetime
import pytz

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from config.config import (
    GOOGLE_CREDENTIALS_FILE, GOOGLE_TOKEN_FILE,
    GOOGLE_CALENDAR_ID, REMINDER_MINUTES_BEFORE,
)

SCOPES   = ["https://www.googleapis.com/auth/calendar.readonly"]
TIMEZONE = pytz.timezone("Asia/Ho_Chi_Minh")

WEEKDAY_VI = ["thứ 2", "thứ 3", "thứ 4", "thứ 5", "thứ 6", "thứ 7", "chủ nhật"]


def _get_service():
    creds = None
    try:
        creds = Credentials.from_authorized_user_file(GOOGLE_TOKEN_FILE, SCOPES)
    except Exception:
        pass

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow  = InstalledAppFlow.from_client_secrets_file(GOOGLE_CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(GOOGLE_TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def _now() -> datetime.datetime:
    return datetime.datetime.now(tz=TIMEZONE)


def get_today_info() -> str:
    now = _now()
    weekday = WEEKDAY_VI[now.weekday()]
    return f"Hôm nay là {weekday}, ngày {now.strftime('%d/%m/%Y')}."


def get_events(days_offset: int = 0, days_ahead: int = 0, max_results: int = 10) -> list[dict]:
    """
    days_offset: 0=hôm nay, 1=ngày mai, -1=hôm qua
    days_ahead:  kéo dài thêm bao nhiêu ngày (0=chỉ 1 ngày, 6=cả tuần)
    """
    service = _get_service()
    now     = _now()
    base    = now + datetime.timedelta(days=days_offset)
    start   = base.replace(hour=0, minute=0, second=0, microsecond=0)
    end     = (start + datetime.timedelta(days=days_ahead + 1)).replace(
                  hour=23, minute=59, second=59)

    result = service.events().list(
        calendarId   = GOOGLE_CALENDAR_ID,
        timeMin      = start.isoformat(),
        timeMax      = end.isoformat(),
        maxResults   = max_results,
        singleEvents = True,
        orderBy      = "startTime",
    ).execute()

    return result.get("items", [])


def format_events(events: list[dict]) -> str:
    if not events:
        return "Không có sự kiện nào."

    lines = []
    for e in events:
        title = e.get("summary", "(Không có tiêu đề)")
        start = e["start"].get("dateTime") or e["start"].get("date")
        try:
            dt       = datetime.datetime.fromisoformat(start).astimezone(TIMEZONE)
            weekday  = WEEKDAY_VI[dt.weekday()]
            time_str = dt.strftime(f"{weekday} %d/%m %H:%M")
        except Exception:
            time_str = start
        lines.append(f"• {time_str} — {title}")

    return "\n".join(lines)


def get_upcoming_reminders() -> list[dict]:
    service = _get_service()
    now     = _now()
    window  = now + datetime.timedelta(minutes=REMINDER_MINUTES_BEFORE)

    result = service.events().list(
        calendarId   = GOOGLE_CALENDAR_ID,
        timeMin      = now.isoformat(),
        timeMax      = window.isoformat(),
        singleEvents = True,
        orderBy      = "startTime",
    ).execute()

    return result.get("items", [])