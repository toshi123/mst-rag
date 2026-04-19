#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


def safe_json_loads(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_text_blocks(result: Any) -> List[str]:
    texts: List[str] = []
    for block in getattr(result, "content", []) or []:
        if isinstance(block, types.TextContent):
            texts.append(block.text)
        else:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                texts.append(text)
    return texts


def normalize_call_result(result: Any) -> Any:
    structured = getattr(result, "structured_content", None)
    if structured not in (None, {}, []):
        return structured

    texts = extract_text_blocks(result)
    for text in texts:
        parsed = safe_json_loads(text)
        if parsed is not None:
            return parsed

    if texts:
        return texts[0]
    return None


def normalize_resource_result(resource_result: Any) -> Any:
    contents = getattr(resource_result, "contents", []) or []
    for content in contents:
        text = getattr(content, "text", None)
        if isinstance(text, str):
            parsed = safe_json_loads(text)
            if parsed is not None:
                return parsed
            return text
    return None


def extract_posts(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ["posts", "items", "results", "data"]:
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [payload]
    return []


async def main() -> None:
    if "ESA_ACCESS_TOKEN" not in os.environ:
        raise RuntimeError("ESA_ACCESS_TOKEN is not set")

    team = os.environ.get("ESA_TEAM")
    if not team:
        raise RuntimeError("ESA_TEAM is not set")

    env = {
        "ESA_ACCESS_TOKEN": os.environ["ESA_ACCESS_TOKEN"],
        "LANG": os.environ.get("LANG", "ja"),
    }

    server_params = StdioServerParameters(
        command="docker",
        args=[
            "run",
            "-i",
            "--rm",
            "-e", "ESA_ACCESS_TOKEN",
            "-e", "LANG",
            "ghcr.io/esaio/esa-mcp-server",
        ],
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("=== tools ===")
            for t in tools.tools:
                print("-", t.name)
            print()

            print("=== recent resource ===")
            rr = await session.read_resource(f"esa://teams/{team}/posts/recent")
            recent_payload = normalize_resource_result(rr)
            recent_posts = extract_posts(recent_payload)
            print("recent count =", len(recent_posts))
            for i, p in enumerate(recent_posts[:3], start=1):
                print(f"[recent {i}] keys =", sorted(p.keys()))
                print(f"  title =", p.get("name") or p.get("title"))
                print(f"  number =", p.get("number"))
                print(f"  url =", p.get("url"))
                body = p.get("body_md") or p.get("body") or ""
                print(f"  body_len =", len(body))
            print()

            print("=== search tool ===")
            search_result = await session.call_tool(
                "esa_search_posts",
                arguments={"teamName": team, "query": "", "page": 1, "perPage": 20},
            )
            search_payload = normalize_call_result(search_result)
            search_posts = extract_posts(search_payload)
            print("search count =", len(search_posts))
            for i, p in enumerate(search_posts[:3], start=1):
                print(f"[search {i}] keys =", sorted(p.keys()))
                print(f"  title =", p.get("name") or p.get("title"))
                print(f"  number =", p.get("number"))
                print(f"  url =", p.get("url"))
                body = p.get("body_md") or p.get("body") or ""
                print(f"  body_len =", len(body))
            print()

            # try get_post only if number exists
            candidate: Optional[Dict[str, Any]] = None
            for p in recent_posts + search_posts:
                if p.get("number") is not None:
                    candidate = p
                    break

            if candidate is None:
                print("No candidate with post number found.")
                return

            number = candidate["number"]
            print("=== get_post test ===")
            print("using number =", number)

            get_result = await session.call_tool(
                "esa_get_post",
                arguments={"teamName": team, "postNumber": int(number)},
            )
            get_payload = normalize_call_result(get_result)
            get_posts = extract_posts(get_payload)
            if get_posts:
                p = get_posts[0]
            elif isinstance(get_payload, dict):
                p = get_payload
            else:
                print("get_post payload =", repr(get_payload))
                return

            print("get_post keys =", sorted(p.keys()))
            print("title =", p.get("name") or p.get("title"))
            print("number =", p.get("number"))
            print("url =", p.get("url"))
            body = p.get("body_md") or p.get("body") or ""
            print("body_len =", len(body))


if __name__ == "__main__":
    asyncio.run(main())