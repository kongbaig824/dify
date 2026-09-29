"""Redis-backed long-term Agent memory.

Long-term memories are explicit, user-scoped facts rather than a copy of the
conversation transcript. Callers decide what is worth remembering and provide
the normalized text. The store keeps a bounded, expiring collection per
tenant/user so memory cannot grow without limit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from redis.asyncio import Redis


@dataclass(frozen=True, slots=True)
class AgentMemory:
    """One normalized long-term memory item."""

    text: str
    created_at: datetime


class RedisLongTermMemory:
    """Bounded Redis list for explicit user-scoped Agent memories."""

    def __init__(
        self,
        redis: Redis,
        *,
        prefix: str = "dify-agent",
        max_memories: int = 100,
        retention_seconds: int = 60 * 60 * 24 * 180,
    ) -> None:
        if max_memories <= 0:
            raise ValueError("max_memories must be positive")
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")
        self.redis = redis
        self.prefix = prefix
        self.max_memories = max_memories
        self.retention_seconds = retention_seconds

    def _key(self, tenant_id: str, user_id: str) -> str:
        if not tenant_id or not user_id:
            raise ValueError("tenant_id and user_id are required for long-term memory")
        return f"{self.prefix}:memory:{tenant_id}:{user_id}"

    async def add(self, tenant_id: str, user_id: str, text: str) -> AgentMemory:
        """Append one explicit memory and trim the oldest entries."""
        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("memory text must not be blank")
        memory = AgentMemory(text=normalized, created_at=datetime.now(timezone.utc))
        payload = json.dumps(
            {"text": memory.text, "created_at": memory.created_at.isoformat()},
            ensure_ascii=False,
        )
        key = self._key(tenant_id, user_id)
        async with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.lpush(key, payload)
            pipeline.ltrim(key, 0, self.max_memories - 1)
            pipeline.expire(key, self.retention_seconds)
            await pipeline.execute()
        return memory

    async def list(self, tenant_id: str, user_id: str, *, limit: int = 20) -> list[AgentMemory]:
        """Return newest memories first."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        raw = await self.redis.lrange(self._key(tenant_id, user_id), 0, limit - 1)
        result: list[AgentMemory] = []
        for value in raw:
            if isinstance(value, bytes):
                value = value.decode()
            data = json.loads(value)
            result.append(AgentMemory(text=data["text"], created_at=datetime.fromisoformat(data["created_at"])))
        return result

    async def clear(self, tenant_id: str, user_id: str) -> None:
        """Delete all long-term memories for one tenant/user scope."""
        await self.redis.delete(self._key(tenant_id, user_id))


__all__ = ["AgentMemory", "RedisLongTermMemory"]
