"""SQLAlchemy persistent conversation memory and preference extraction."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from agent.prompts import PREFERENCE_EXTRACTION_PROMPT
from agent.schemas import PreferenceUpdate

logger = logging.getLogger(__name__)

DEFAULT_DATABASE_URL = "sqlite:///./agent.db"
PREFERENCE_KEYS: tuple[str, ...] = (
    "liked_genres",
    "disliked_genres",
    "liked_movies",
    "constraints",
)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def normalize_database_url(raw_url: str | None = None) -> str:
    """Resolve and normalize DATABASE_URL for SQLAlchemy 2.x."""
    url = (raw_url if raw_url is not None else os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)).strip()
    if not url:
        url = DEFAULT_DATABASE_URL
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    return url


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class ChatSession(Base):
    """Conversation session row."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class ChatMessage(Base):
    """Persisted turn message belonging to a session."""

    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_session_created", "session_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class UserPreference(Base):
    """Key-value preference list stored per session."""

    __tablename__ = "user_preferences"
    __table_args__ = (
        UniqueConstraint("session_id", "key", name="uq_user_preferences_session_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


def _merge_unique(existing: list[str], incoming: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*existing, *incoming]:
        clean = str(item).strip()
        if not clean:
            continue
        norm = clean.lower()
        if norm not in seen:
            seen.add(norm)
            merged.append(clean)
    return merged


class ConversationMemoryStore:
    """Manages persistent chat sessions, history, and extracted user preferences."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = normalize_database_url(database_url)
        connect_args = {"check_same_thread": False} if self.database_url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(self.database_url, connect_args=connect_args, future=True)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, class_=Session)
        Base.metadata.create_all(self.engine)

    def session_exists(self, session_id: str) -> bool:
        """Return True if the session ID exists in the database."""
        with self._session_factory() as db:
            row = db.get(ChatSession, session_id)
            return row is not None

    def ensure_session(self, session_id: str | None = None) -> str:
        """Return an existing session ID or create a new session row."""
        candidate = (session_id or "").strip() or uuid.uuid4().hex
        with self._session_factory() as db:
            existing = db.get(ChatSession, candidate)
            if existing is None:
                db.add(ChatSession(id=candidate))
                db.commit()
        return candidate

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """Persist a chat message for a session."""
        with self._session_factory() as db:
            db.add(ChatMessage(session_id=session_id, role=role, content=content))
            db.commit()

    def get_recent_messages(self, session_id: str, limit: int = 10) -> list[dict[str, str]]:
        """Fetch the most recent `limit` messages for a session in chronological order."""
        with self._session_factory() as db:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
                .limit(limit)
            )
            rows = list(db.execute(stmt).scalars())
        rows.reverse()
        return [{"role": row.role, "content": row.content} for row in rows]

    def get_preferences(self, session_id: str) -> dict[str, list[str]]:
        """Return all preference lists stored for a session."""
        prefs: dict[str, list[str]] = {k: [] for k in PREFERENCE_KEYS}
        with self._session_factory() as db:
            stmt = select(UserPreference).where(UserPreference.session_id == session_id)
            for row in db.execute(stmt).scalars():
                try:
                    parsed = json.loads(row.value)
                    if isinstance(parsed, list):
                        prefs[row.key] = [str(v) for v in parsed]
                except ValueError:
                    continue
        return prefs

    def upsert_preferences(self, session_id: str, update: PreferenceUpdate) -> dict[str, list[str]]:
        """Merge and upsert preference updates for a session."""
        incoming_map = {
            "liked_genres": update.liked_genres,
            "disliked_genres": update.disliked_genres,
            "liked_movies": update.liked_movies,
            "constraints": update.constraints,
        }
        now = utc_now()
        with self._session_factory() as db:
            for key, new_values in incoming_map.items():
                if not new_values:
                    continue
                stmt = select(UserPreference).where(
                    UserPreference.session_id == session_id,
                    UserPreference.key == key,
                )
                row = db.execute(stmt).scalar_one_or_none()
                existing_values: list[str] = []
                if row is not None:
                    try:
                        loaded = json.loads(row.value)
                        if isinstance(loaded, list):
                            existing_values = [str(v) for v in loaded]
                    except ValueError:
                        existing_values = []

                merged = _merge_unique(existing_values, new_values)
                serialized = json.dumps(merged, ensure_ascii=False)
                if row is None:
                    db.add(
                        UserPreference(
                            session_id=session_id,
                            key=key,
                            value=serialized,
                            updated_at=now,
                        )
                    )
                else:
                    row.value = serialized
                    row.updated_at = now
            db.commit()
        return self.get_preferences(session_id)

    def extract_and_update_preferences(
        self,
        session_id: str,
        user_message: str,
        llm_client: Any,
        model: str | None = None,
    ) -> dict[str, list[str]]:
        """Run a lightweight LLM extraction call and upsert validated preferences."""
        try:
            resp = llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": PREFERENCE_EXTRACTION_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                tools=None,
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=15.0,
                model=model,
            )
            if not resp.content:
                return self.get_preferences(session_id)
            update = PreferenceUpdate.model_validate(json.loads(resp.content))
            return self.upsert_preferences(session_id, update)
        except (ValidationError, ValueError, Exception) as exc:
            logger.warning("Preference extraction skipped: %s", exc.__class__.__name__)
            return self.get_preferences(session_id)
