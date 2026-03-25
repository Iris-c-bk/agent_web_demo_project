# agent_core.py — 循环体在 SmartAgent.chat() 内完成并 return；配置请在其他脚本注入 OpenAI 与 Tavily。
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI
from tavily import TavilyClient

import sqlite3

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "当用户询问新闻、实时事件、股票、天气或你不知道的事实时，必须使用此工具搜索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def _ensure_chats_schema(conn: sqlite3.Connection) -> None:
    """表 chats：session_id 主键；若存在旧表（无主键、可重复 session），迁移为每个 session 保留最新一条。"""
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chats'")
    if cur.fetchone() is None:
        cur.execute(
            """CREATE TABLE chats (
                session_id TEXT PRIMARY KEY,
                history_json TEXT NOT NULL
            )"""
        )
        conn.commit()
        return

    cur.execute("PRAGMA table_info(chats)")
    rows = cur.fetchall()
    pk_on_session = any(r[1] == "session_id" and r[5] >= 1 for r in rows)
    if pk_on_session:
        return

    cur.execute(
        """CREATE TABLE chats_migrated (
            session_id TEXT PRIMARY KEY,
            history_json TEXT NOT NULL
        )"""
    )
    cur.execute(
        """INSERT INTO chats_migrated (session_id, history_json)
           SELECT session_id, history_json FROM chats
           WHERE rowid IN (SELECT MAX(rowid) FROM chats GROUP BY session_id)"""
    )
    cur.execute("DROP TABLE chats")
    cur.execute("ALTER TABLE chats_migrated RENAME TO chats")
    conn.commit()


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif getattr(block, "type", None) == "text":
                parts.append(str(getattr(block, "text", "") or ""))
        return "".join(parts)
    return str(content)


def _tool_call_arguments(args: Any) -> str:
    if isinstance(args, str):
        return args
    return json.dumps(args, ensure_ascii=False)


def _assistant_to_dict(msg: Any, tool_calls_only: Optional[List[Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"role": "assistant"}
    text = _stringify_content(msg.content)
    tcs = tool_calls_only if tool_calls_only is not None else msg.tool_calls
    if tcs:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": _tool_call_arguments(tc.function.arguments),
                },
            }
            for tc in tcs
        ]
        # SiliconFlow 对 assistant 消息更严格：有 tool_calls 也要显式 content 字段
        out["content"] = text
    else:
        out["content"] = text
    return out


def _messages_for_siliconflow(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    open_tool_ids: set = set()
    open_assistant_index: Optional[int] = None
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            # 若上一段 assistant(tool_calls) 没有完整 tool 返回，直接丢弃该段，避免 messages 非法
            if open_tool_ids and open_assistant_index is not None:
                if 0 <= open_assistant_index < len(out):
                    out.pop(open_assistant_index)
                open_tool_ids.clear()
                open_assistant_index = None

            tcs = m.get("tool_calls")
            text = _stringify_content(m.get("content"))
            if not tcs and not text:
                continue
            cm: Dict[str, Any] = {"role": "assistant"}
            if tcs:
                valid_tcs = []
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    tc_id = tc.get("id")
                    fn = tc.get("function") or {}
                    if not tc_id or not isinstance(fn, dict):
                        continue
                    if not fn.get("name"):
                        continue
                    if "arguments" not in fn:
                        continue
                    valid_tcs.append(tc)
                if valid_tcs:
                    open_tool_ids = {tc["id"] for tc in valid_tcs if tc.get("id")}
                else:
                    open_tool_ids.clear()
                if not valid_tcs:
                    if not text:
                        continue
                else:
                    cm["tool_calls"] = valid_tcs
                cm["content"] = text
            else:
                cm["content"] = text
            out.append(cm)
            open_assistant_index = len(out) - 1 if open_tool_ids else None
        elif role == "tool":
            tool_call_id = m.get("tool_call_id")
            if not tool_call_id or tool_call_id not in open_tool_ids:
                continue
            open_tool_ids.discard(tool_call_id)
            if not open_tool_ids:
                open_assistant_index = None
            out.append(
                {
                    "role": "tool",
                    "content": str(m.get("content", "")),
                    "tool_call_id": tool_call_id,
                }
            )
        elif role in ("system", "user"):
            text = _stringify_content(m.get("content"))
            if not text:
                continue
            out.append({"role": role, "content": text})
        else:
            continue

    # 收尾：若最后仍有未闭合 tool_calls，移除对应 assistant(tool_calls) 片段
    if open_tool_ids and open_assistant_index is not None:
        if 0 <= open_assistant_index < len(out):
            out.pop(open_assistant_index)
    return out


def _sanitize_history(messages: Any) -> List[Dict[str, Any]]:
    """清洗历史消息，确保可被 SiliconFlow 接口稳定接受。"""
    if not isinstance(messages, list):
        return [{"role": "system", "content": "你是一个专业的智能助手。"}]

    cleaned = _messages_for_siliconflow(messages)
    has_system = any(m.get("role") == "system" for m in cleaned)
    if not has_system:
        cleaned.insert(0, {"role": "system", "content": "你是一个专业的智能助手。"})
    return cleaned


def _text_only_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """降级用：仅保留 system/user/assistant 的文本消息。"""
    out: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("system", "user", "assistant"):
            continue
        text = _stringify_content(m.get("content"))
        if not text:
            continue
        out.append({"role": role, "content": text})
    if not any(m.get("role") == "system" for m in out):
        out.insert(0, {"role": "system", "content": "你是一个专业的智能助手。"})
    return out


class SmartAgent:
    def __init__(self, api_key, base_url, model_name, tavily_key=None, memory_db_path="memory.db"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name
        self.tavily = TavilyClient(api_key=tavily_key) if tavily_key else None
        self._memory_db_path = memory_db_path

        self.messages = [
            {"role": "system", "content": "你是一个专业的智能助手。"}
        ]
        self.tools = self._define_tools()
        self.available_functions: Dict[str, Callable[..., str]] = self._map_functions()

    # 保存历史记忆
    def save_history(self, session_id="default_user") -> None:
        history_str = json.dumps(self.messages, ensure_ascii=False)
        with sqlite3.connect(self._memory_db_path) as conn:
            _ensure_chats_schema(conn)
            conn.execute(
                """INSERT INTO chats (session_id, history_json) VALUES (?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                   history_json = excluded.history_json""",
                (session_id, history_str),
            )
            conn.commit()
        print("💾 记忆已保存至 {} ({})".format(self._memory_db_path, session_id))

    # 加载历史记忆
    def load_history(self, session_id="default_user") -> None:
        with sqlite3.connect(self._memory_db_path) as conn:
            _ensure_chats_schema(conn)
            row = conn.execute(
                "SELECT history_json FROM chats WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()

        if row:
            self.messages = _sanitize_history(json.loads(row[0]))
            print("🧠 已加载历史记忆 ({})".format(session_id))
        else:
            print("✨ 新用户，开始新对话")

    # 定义工具
    def _define_tools(self):
        return tools

    # 映射函数
    def _map_functions(self):
        return {"search_web": self._search_web}

    # 搜索网络
    def _search_web(self, query: str) -> str:
        if not self.tavily:
            return "未配置搜索 Key"
        try:
            data = self.tavily.search(query, max_results=3)
            rows = data.get("results") or []
            lines = [
                "- {}: {}".format(r.get("title", ""), r.get("content", ""))
                for r in rows
            ]
            return "搜索结果:\n" + "\n".join(lines)
        except Exception as e:
            return "搜索失败: {}".format(e)

    # 对话
    def chat(self, user_input: str) -> Tuple[str, List[Dict[str, Any]]]:
        """一轮用户输入：内部可多步工具调用，返回(最终文本, 工具调用过程)。"""
        self.messages.append({"role": "user", "content": user_input})

        final_text = ""
        thought_process: List[Dict[str, Any]] = []
        step = 0
        while step < 10:
            step += 1
            try:
                kwargs = {
                    "model": self.model,
                    "messages": _messages_for_siliconflow(self.messages),
                    "tools": self.tools,
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                }

                def _create_once(call_kwargs: Dict[str, Any]):
                    try:
                        return self.client.chat.completions.create(**call_kwargs)
                    except TypeError:
                        call_kwargs = dict(call_kwargs)
                        call_kwargs.pop("parallel_tool_calls", None)
                        return self.client.chat.completions.create(**call_kwargs)

                try:
                    response = _create_once(kwargs)
                except Exception as api_err:
                    err_text = str(api_err).lower()
                    illegal_messages = "20015" in err_text or ("messages" in err_text and "illegal" in err_text)
                    if not illegal_messages:
                        raise

                    safe_messages = _text_only_messages(self.messages)
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["messages"] = safe_messages
                    response = _create_once(retry_kwargs)
                    self.messages = safe_messages

                msg = response.choices[0].message

                if msg.tool_calls and len(msg.tool_calls) > 1:
                    store_calls = msg.tool_calls[:1]
                else:
                    store_calls = msg.tool_calls

                self.messages.append(_assistant_to_dict(msg, tool_calls_only=store_calls))

                if not msg.tool_calls:
                    final_text = _stringify_content(msg.content)
                    break

                for tool_call in store_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    if func_name in self.available_functions:
                        result = self.available_functions[func_name](**args)
                        thought_process.append(
                            {
                                "step": len(thought_process) + 1,
                                "tool": func_name,
                                "args": args,
                                "result": str(result),
                            }
                        )
                        self.messages.append(
                            {
                                "role": "tool",
                                "content": str(result),
                                "tool_call_id": tool_call.id,
                            }
                        )
                    else:
                        error_text = "Error: Function not found."
                        thought_process.append(
                            {
                                "step": len(thought_process) + 1,
                                "tool": func_name,
                                "args": args,
                                "result": error_text,
                            }
                        )
                        self.messages.append(
                            {
                                "role": "tool",
                                "content": error_text,
                                "tool_call_id": tool_call.id,
                            }
                        )
            except Exception as e:
                return "出错了: {}".format(e), thought_process

        reply = final_text if final_text else "（未得到文本回复，可能超过最大步数）"
        return reply, thought_process

    # 获取历史记忆
    def get_history(self):
        return self.messages

    # 清空历史记忆
    def clear_history(self):
        self.messages = [{"role": "system", "content": "你是一个专业的智能助手。"}]
