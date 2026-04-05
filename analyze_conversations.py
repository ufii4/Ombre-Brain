#!/usr/bin/env python3
# ============================================================
# Script: Process Exported Conversations into Memory Buckets
#
# Server-side processor. Reads conversations from the SQLite
# database created by export_conversations.py, chunks the
# transcripts, runs them through the Ombre Brain pipeline
# (dehydrator.digest → merge-or-create → decay sweep).
#
# Usage:
#   python analyze_conversations.py --db conversations.db --dry-run
#   python analyze_conversations.py --db conversations.db --provider claude-cli
#   python analyze_conversations.py --db conversations.db --batch 20
# ============================================================

import argparse
import asyncio
import json
import os
import subprocess
import sys
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

from utils import load_config
from dehydrator import Dehydrator
from bucket_manager import BucketManager
from decay_engine import DecayEngine

logger = logging.getLogger("ombre_brain.analyze")

CHUNK_WINDOW = 15
CHUNK_OVERLAP = 3

# Conversation-specific digest prompt.
# Same output schema as the project's DIGEST_PROMPT (name, content, domain,
# valence, arousal, tags, importance) but written for [USER]/[ASSISTANT]
# dialogue. Valence and arousal are from the assistant's perspective —
# matching the project's philosophy that memory belongs to the assistant.
CONVERSATION_DIGEST_PROMPT = """你是一个对话记忆提取器。你会收到一段 [USER] 和 [ASSISTANT] 之间的对话片段。

你的任务：从助手的视角提取值得长期记住的记忆条目。

记忆属于助手——valence 和 arousal 反映的是助手在回应时的内心体验：
  valence（效价）：0=被纠正/感到困惑/做错了, 0.5=例行公事, 1.0=做对了/用户满意/想法被采纳
  arousal（唤醒度）：0=简单任务/自动化操作, 0.5=一般参与, 1.0=高难度挑战/激烈讨论/重大突破

注意对话角色的不同含义：
  [USER] 的消息揭示：用户的偏好、专业水平、需求、情绪状态、纠正
  [ASSISTANT] 的消息揭示：什么方法有效/无效、被纠正了什么、做出了什么承诺

提取规则：
1. 用户的纠正和推回 → 高 importance（助手下次该怎么做）
2. 用户表达的强烈偏好和习惯 → 高 importance（"不要这样做"、"我更喜欢..."）
3. 尚未解决的问题和延后的决定 → 中高 importance
4. 关于用户身份、项目、目标的背景信息 → 中 importance
5. 例行的代码操作、工具输出、文件列表 → 跳过
6. 可以从代码或文档重新获取的信息 → 跳过
7. 每个条目应独立完整，从助手第一人称视角描述

输出格式（纯 JSON 数组，无其他内容）：
[
  {
    "name": "条目标题（10字以内）",
    "content": "从助手视角：发生了什么，学到了什么，下次应该怎么做或继续做",
    "domain": ["主题域1"],
    "valence": 0.3,
    "arousal": 0.8,
    "tags": ["标签1", "标签2"],
    "importance": 7
  }
]

主题域可选（选最精确的 1~2 个）：
  日常: ["饮食", "穿搭", "出行", "居家", "购物"]
  人际: ["家庭", "恋爱", "友谊", "社交"]
  成长: ["工作", "学习", "考试", "求职"]
  身心: ["健康", "心理", "睡眠", "运动"]
  兴趣: ["游戏", "影视", "音乐", "阅读", "创作", "手工"]
  数字: ["编程", "AI", "硬件", "网络"]
  事务: ["财务", "计划", "待办"]
  内心: ["情绪", "回忆", "梦境", "自省"]
importance: 1-10，根据这条记忆对未来对话行为的影响程度判断
valence: 0~1（助手视角：0=消极体验, 0.5=中性, 1=积极体验）
arousal: 0~1（助手视角：0=平静, 0.5=普通, 1=激动）

如果对话片段没有任何值得记住的内容，返回空数组：[]"""


# ── Chunking ─────────────────────────────────────────────────

def chunk_transcript(transcript: str, window: int = CHUNK_WINDOW, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split a transcript into overlapping message-window chunks."""
    blocks = transcript.split("\n\n")
    if len(blocks) <= window:
        return [transcript]

    chunks = []
    start = 0
    while start < len(blocks):
        chunk_blocks = blocks[start:start + window]
        chunk_text = "\n\n".join(chunk_blocks)
        if len(chunk_text) >= 100:
            chunks.append(chunk_text)
        start += window - overlap
    return chunks


# ── Provider: openai-compatible API ──────────────────────────

async def digest_with_api(transcript: str, dehydrator: Dehydrator) -> list[dict]:
    """Call the configured OpenAI-compatible API with CONVERSATION_DIGEST_PROMPT."""
    if not dehydrator.api_available:
        return []

    response = await dehydrator.client.chat.completions.create(
        model=dehydrator.model,
        messages=[
            {"role": "system", "content": CONVERSATION_DIGEST_PROMPT},
            {"role": "user", "content": transcript[:3000]},
        ],
        max_tokens=dehydrator.max_tokens,
        temperature=dehydrator.temperature,
    )
    if not response.choices:
        return []
    return _parse_json_array(response.choices[0].message.content or "")


# ── Provider: claude-cli ─────────────────────────────────────

def digest_with_claude_cli(transcript: str, model: str = None) -> list[dict]:
    cmd = ["claude", "-p"]
    if model:
        cmd.extend(["--model", model])

    prompt = f"{CONVERSATION_DIGEST_PROMPT}\n\n--- 以下是对话片段 ---\n{transcript}"
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {result.stderr[:500]}")

    return _parse_json_array(result.stdout)


def _parse_json_array(raw: str) -> list[dict]:
    raw = raw.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        entries = json.loads(raw[start:end + 1])
        return entries if isinstance(entries, list) else []
    except json.JSONDecodeError:
        return []


# ── Process conversations ──────────────────────���─────────────

async def save_items(
    items: list[dict],
    conv_ts: str,
    dehydrator: Dehydrator,
    bucket_mgr: BucketManager,
    config: dict,
    stats: dict,
):
    """Save digest results to buckets (merge-or-create). Must run sequentially."""
    for item in items:
        try:
            existing = await bucket_mgr.search(item.get("content", ""), limit=1)
            if existing and existing[0].get("score", 0) > config.get("merge_threshold", 75):
                bucket = existing[0]
                merged_content = await dehydrator.merge(bucket["content"], item["content"])
                await bucket_mgr.update(
                    bucket["id"],
                    content=merged_content,
                    tags=list(set(bucket["metadata"].get("tags", []) + item.get("tags", []))),
                    importance=max(bucket["metadata"].get("importance", 5), item.get("importance", 5)),
                    domain=list(set(bucket["metadata"].get("domain", []) + item.get("domain", []))),
                    valence=item.get("valence", 0.5),
                    arousal=item.get("arousal", 0.3),
                )
                stats["merged"] += 1
                logger.info(f"    merged → {bucket['metadata'].get('name', bucket['id'])}")
            else:
                await bucket_mgr.create(
                    content=item.get("content", ""),
                    tags=item.get("tags", []),
                    importance=item.get("importance", 5),
                    domain=item.get("domain", []),
                    valence=item.get("valence", 0.5),
                    arousal=item.get("arousal", 0.3),
                    name=item.get("name", ""),
                    created=conv_ts,
                    last_active=conv_ts,
                )
                stats["created"] += 1
                logger.info(f"    + {item.get('name', '?')} (V{item.get('valence', 0.5):.1f}/A{item.get('arousal', 0.3):.1f})")
        except Exception as e:
            logger.warning(f"    entry failed: {e}")


async def process_db(
    db_path: str,
    provider: str,
    model: Optional[str],
    dehydrator: Dehydrator,
    bucket_mgr: BucketManager,
    config: dict,
    batch_size: int,
    workers: int,
    session_filter: Optional[str],
    dry_run: bool,
) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Latest conversations first
    query = "SELECT * FROM conversations WHERE status = 'pending'"
    params = []
    if session_filter:
        query += " AND id LIKE ?"
        params.append(f"%{session_filter}%")
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(batch_size)

    rows = conn.execute(query, params).fetchall()
    total_pending = conn.execute("SELECT COUNT(*) FROM conversations WHERE status = 'pending'").fetchone()[0]

    logger.info(f"Pending: {total_pending}, processing batch: {len(rows)}, workers: {workers}")

    stats = {"processed": 0, "created": 0, "merged": 0, "failed": 0}
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=workers) if provider == "claude-cli" else None

    for row in rows:
        label = row["name"] or row["project"] or row["id"][:8]
        conv_ts = row["created_at"]

        msg_rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY position",
            (row["id"],)
        ).fetchall()
        transcript = "\n\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in msg_rows)
        chunks = chunk_transcript(transcript)

        logger.info(f"  [{row['source']}] {label} — {len(msg_rows)} msgs → {len(chunks)} chunks")

        if dry_run:
            for i, chunk in enumerate(chunks):
                logger.info(f"    chunk {i+1}: {len(chunk)} chars")
            conn.execute(
                "UPDATE conversations SET status = 'processed', processed_at = ? WHERE id = ?",
                (datetime.now().isoformat(), row["id"])
            )
            conn.commit()
            stats["processed"] += 1
            continue

        # Digest all chunks — parallel for claude-cli, async for openai
        chunk_results = []  # list of (chunk_index, items_or_error)
        chunk_failed = False

        if provider == "claude-cli" and workers > 1:
            # Parallel: run claude-cli calls in thread pool
            futures = []
            for i, chunk in enumerate(chunks):
                future = loop.run_in_executor(
                    executor, digest_with_claude_cli, chunk, model
                )
                futures.append((i, future))

            for i, future in futures:
                try:
                    items = await future
                    if items:
                        chunk_results.append(items)
                except subprocess.TimeoutExpired:
                    logger.warning(f"    chunk {i+1} timed out")
                    chunk_failed = True
                except Exception as e:
                    logger.error(f"    chunk {i+1} failed: {e}")
                    chunk_failed = True
        else:
            # Sequential
            for i, chunk in enumerate(chunks):
                try:
                    if provider == "claude-cli":
                        items = digest_with_claude_cli(chunk, model=model)
                    else:
                        items = await digest_with_api(chunk, dehydrator)
                    if items:
                        chunk_results.append(items)
                except subprocess.TimeoutExpired:
                    logger.warning(f"    chunk {i+1} timed out")
                    chunk_failed = True
                except Exception as e:
                    logger.error(f"    chunk {i+1} failed: {e}")
                    chunk_failed = True

        # Save all results sequentially (touches shared filesystem)
        for items in chunk_results:
            await save_items(items, conv_ts, dehydrator, bucket_mgr, config, stats)

        status = "failed" if chunk_failed else "processed"
        conn.execute(
            "UPDATE conversations SET status = ?, processed_at = ? WHERE id = ?",
            (status, datetime.now().isoformat(), row["id"])
        )
        conn.commit()
        stats["processed"] += 1
        if chunk_failed:
            stats["failed"] += 1

    if executor:
        executor.shutdown(wait=False)
    conn.close()
    return stats


# ── Main ─────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Process exported conversations into Ombre Brain memory buckets"
    )
    parser.add_argument("--db", required=True, help="Path to conversations.db")
    parser.add_argument("--provider", choices=["openai", "claude-cli"], default="openai")
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--batch", type=int, default=50, help="Conversations per run (default: 50)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel claude-cli workers (default: 4)")
    parser.add_argument("--session", default=None, help="Filter by conversation ID")
    parser.add_argument("--output-dir", default=None, help="Bucket output directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-decay", action="store_true", help="Skip decay sweep")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not os.path.exists(args.db):
        logger.error(f"Database not found: {args.db}")
        sys.exit(1)

    config = load_config()
    if args.output_dir:
        config["buckets_dir"] = args.output_dir

    dehydrator = Dehydrator(config)
    bucket_mgr = BucketManager(config)

    if args.provider == "openai" and not dehydrator.api_available:
        logger.error("OpenAI provider requires API key. Set OMBRE_API_KEY or configure config.yaml")
        sys.exit(1)

    stats = await process_db(
        args.db, args.provider, args.model,
        dehydrator, bucket_mgr, config,
        args.batch, args.workers, args.session, args.dry_run,
    )

    logger.info(
        f"\nDone. Processed: {stats['processed']}, "
        f"Created: {stats['created']}, Merged: {stats['merged']}, "
        f"Failed: {stats['failed']}"
    )

    if not args.dry_run and stats["created"] > 0 and not args.no_decay:
        logger.info("\nRunning decay sweep...")
        decay_engine = DecayEngine(config, bucket_mgr)
        result = await decay_engine.run_decay_cycle()
        logger.info(
            f"Decay complete. Checked: {result['checked']}, "
            f"Archived: {result['archived']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
