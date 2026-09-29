from datetime import timezone
from unittest.mock import AsyncMock

import pytest

from dify_agent.storage.long_term_memory import RedisLongTermMemory


@pytest.mark.asyncio
async def test_add_normalizes_and_limits_memory_entries() -> None:
    redis = AsyncMock()
    pipeline = AsyncMock()
    redis.pipeline.return_value.__aenter__.return_value = pipeline
    store = RedisLongTermMemory(redis, max_memories=2)

    memory = await store.add("tenant", "user", "  prefers   Chinese  answers ")

    assert memory.text == "prefers Chinese answers"
    pipeline.lpush.assert_awaited_once()
    pipeline.ltrim.assert_awaited_once_with("dify-agent:memory:tenant:user", 0, 1)
    pipeline.expire.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_decodes_newest_memories() -> None:
    redis = AsyncMock()
    redis.lrange.return_value = [
        b'{"text":"use Chinese","created_at":"2026-09-29T00:00:00+00:00"}'
    ]
    store = RedisLongTermMemory(redis)

    memories = await store.list("tenant", "user")

    assert [item.text for item in memories] == ["use Chinese"]
    assert memories[0].created_at.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_blank_memory_is_rejected() -> None:
    store = RedisLongTermMemory(AsyncMock())

    with pytest.raises(ValueError, match="memory text"):
        await store.add("tenant", "user", "   ")
