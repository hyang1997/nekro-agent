"""
# Gmail

Lets the agent query the inbox with Gmail's native search syntax and read individual
messages. Read-only.

## Why IMAP rather than the Gmail API

The email adapter offers `TRANSPORT_TYPE: "gmail_api"`, but `GmailApiClient` is a stub
(`class GmailApiClient(_ApiClientBase): pass`, and every base method raises
NotImplementedError). Selecting it fails on the first fetch.

Gmail's IMAP, meanwhile, implements its own extensions. Measured against this mailbox
(2026-08-16, 8850 messages) they all work:

    X-GM-RAW "newer_than:7d"     -> 46
    X-GM-RAW "has:attachment"    -> 186
    X-GM-RAW "category:primary"  -> 3738     <- IMAP itself has no notion of categories
    X-GM-LABELS / X-GM-THRID     -> supported

That `category:` returns a sensible number proves Gmail is parsing the query rather than
falling back to a plain IMAP search. So an app password buys the full search-box syntax
with no Google Cloud project and no OAuth consent screen.

## Read-only

This plugin never sends, never changes labels, never marks anything read. Fetches always
use `BODY.PEEK[...]`; plain `BODY[...]` sets \\Seen, and against a mailbox with 3907
unread messages that is irreversible damage.

Not sending is deliberate. The agent's input includes untrusted content such as Discord
messages. Reading mail bounds the damage at "it learned something"; sending mail means
"something went out under your name". For sending, use the builtin `KroMiose.email_utils`
and turn on SEND_ENABLED for the account explicitly.

## Accounts come from the Email adapter

No credentials are stored here. The account list is read from the adapter's
`RECEIVE_ACCOUNTS`; accounts are still managed at `#/adapters/email/accounts`.

Note the adapter's per-account `RECEIVE_ENABLED` defaults to **on**, which makes
NekroAgent poll the inbox and inject every arriving message into the agent as chat —
anyone who knows the address could then write directly into its prompt. This plugin pulls
on demand and needs no polling, so leave that off.

## Method types

Both methods are **AGENT**: the return value has to re-enter context and trigger another
model step so the model can speak from the real result. As TOOL, the value would only
return into the sandbox script, forcing the model to write its commentary before seeing
the mail. An AGENT method halts the script, so only one runs per iteration.
"""

import email
import email.header
import email.utils
import re
from email.message import Message
from html import unescape
from typing import List, Optional, Tuple

from pydantic import Field

from nekro_agent.adapters import loaded_adapters
from nekro_agent.adapters.email.clients.imap_smtp_password import ImapSmtpPasswordClient
from nekro_agent.adapters.email.config import EmailAccount
from nekro_agent.api import core, i18n
from nekro_agent.api.plugin import (
    ConfigBase,
    ExtraField,
    NekroPlugin,
    SandboxMethodType,
)
from nekro_agent.api.schemas import AgentCtx

plugin = NekroPlugin(
    name="Gmail",
    module_name="gmail",
    description="用 Gmail 原生搜索语法查收件箱并读取邮件（只读）",
    version="0.1.0",
    author="Hao",
    url="https://github.com/hyang1997/nekro-agent",
    i18n_name=i18n.i18n_text(zh_CN="Gmail", en_US="Gmail"),
    i18n_description=i18n.i18n_text(
        zh_CN="用 Gmail 原生搜索语法查收件箱并读取邮件（只读）",
        en_US="Search and read Gmail with native query syntax (read-only)",
    ),
    allow_sleep=True,
    sleep_brief="搜索和阅读 Gmail 收件箱。聊到邮件、账单、订阅、快递、验证码，或早报要看有没有要紧事时激活。",
)


@plugin.mount_config()
class GmailConfig(ConfigBase):
    """Gmail plugin configuration."""

    ACCOUNT: str = Field(
        default="",
        title="邮箱账户",
        description="要使用的邮箱地址；留空则用 Email 适配器里第一个启用的账户",
        json_schema_extra=ExtraField(
            overridable=True,
            placeholder="user@gmail.com",
            i18n_title=i18n.i18n_text(zh_CN="邮箱账户", en_US="Email Account"),
        ).model_dump(),
    )
    MAX_RESULTS: int = Field(
        default=15,
        title="单次搜索最多返回",
        description="搜索结果条数上限。收件箱可能上万封，不设限会把上下文冲垮",
        json_schema_extra=ExtraField(
            overridable=True,
            i18n_title=i18n.i18n_text(zh_CN="单次搜索最多返回", en_US="Max Search Results"),
        ).model_dump(),
    )
    BODY_CHARS: int = Field(
        default=4000,
        title="正文最多返回字符",
        json_schema_extra=ExtraField(
            overridable=True,
            i18n_title=i18n.i18n_text(zh_CN="正文最多返回字符", en_US="Max Body Characters"),
        ).model_dump(),
    )
    IMAP_TIMEOUT: int = Field(
        default=60,
        title="IMAP 超时 (秒)",
        json_schema_extra=ExtraField(
            i18n_title=i18n.i18n_text(zh_CN="IMAP 超时 (秒)", en_US="IMAP Timeout (s)"),
        ).model_dump(),
    )


config: GmailConfig = plugin.get_config(GmailConfig)


# --------------------------------------------------------------------------- account


def _pick_account() -> EmailAccount:
    """Resolve which Email-adapter account to use."""
    adapter = loaded_adapters.get("email")
    accounts: List[EmailAccount] = list(getattr(getattr(adapter, "config", None), "RECEIVE_ACCOUNTS", []) or [])
    if not accounts:
        raise RuntimeError(
            "No account configured in the Email adapter. Add a Gmail account at "
            "#/adapters/email/accounts in the WebUI: pick the Gmail provider and use a "
            "Google app password.",
        )

    wanted = (config.ACCOUNT or "").strip().lower()
    if wanted:
        for acc in accounts:
            if acc.USERNAME.lower() == wanted:
                return acc
        available = ", ".join(a.USERNAME for a in accounts)
        raise RuntimeError(f"No account named {config.ACCOUNT}; available: {available}")

    for acc in accounts:
        if acc.ENABLED:
            return acc
    raise RuntimeError("Every account in the Email adapter is disabled")


# --------------------------------------------------------------------------- parsing


def _decode_header(raw: Optional[str]) -> str:
    """Decode an RFC 2047 header (CJK subjects arrive as =?UTF-8?B?...?=)."""
    if not raw:
        return ""
    out: List[str] = []
    for chunk, charset in email.header.decode_header(raw):
        if isinstance(chunk, bytes):
            out.append(chunk.decode(charset or "utf-8", errors="replace"))
        else:
            out.append(chunk)
    return " ".join("".join(out).split())


_TAG_RE = re.compile(r"<[^>]+>")
# Zero-width and other invisible characters. Marketing mail uses them as preheader
# padding so the inbox preview does not spill body text. Once tags are stripped those
# lines look empty but are not, so strip() leaves them, and the real content ends up
# buried under hundreds of invisible lines — unreadable, and a waste of context.
# Measured: the Garmin sign-in alert is one line of text paved over with U+200C.
_INVISIBLE_RE = re.compile(r"[​-‏  ⁠-⁤﻿­͏]")
_INLINE_WS_RE = re.compile(r"[ \t ]{2,}")


def _tidy_lines(text: str) -> str:
    """Drop invisible characters, strip each line, collapse blank runs to one."""
    text = _INVISIBLE_RE.sub("", text).replace(" ", " ")
    lines: List[str] = []
    for raw_line in text.splitlines():
        line = _INLINE_WS_RE.sub(" ", raw_line).strip()
        if line or (lines and lines[-1]):  # keep a single blank line so paragraphs survive
            lines.append(line)
    return "\n".join(lines).strip()


def _html_to_text(html: str) -> str:
    """Reduce an HTML message to readable plain text."""
    text = re.sub(r"(?is)<(script|style|head|title).*?</\1>", " ", html)
    text = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h[1-6]|table)>", "\n", text)
    text = _TAG_RE.sub(" ", text)
    return _tidy_lines(unescape(text))


def _extract_body(msg: Message) -> Tuple[str, List[str]]:
    """Return (plain-text body, attachment names). Prefer text/plain, fall back to HTML."""
    attachments: List[str] = []
    plain: List[str] = []
    html: List[str] = []

    for part in msg.walk() if msg.is_multipart() else [msg]:
        disposition = str(part.get("Content-Disposition") or "")
        filename = part.get_filename()
        if filename or "attachment" in disposition:
            attachments.append(_decode_header(filename) or "(unnamed attachment)")
            continue
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        try:
            payload = part.get_payload(decode=True)
        except Exception:
            continue
        if not payload:
            continue
        charset = part.get_content_charset() or "utf-8"
        decoded = payload.decode(charset, errors="replace") if isinstance(payload, bytes) else str(payload)
        (plain if ctype == "text/plain" else html).append(decoded)

    # Clean the plain part too: zero-width padding and nbsp show up in text/plain as well
    body = _tidy_lines("\n".join(plain)) or _html_to_text("\n".join(html))
    return body, attachments


def _parse_labels(raw: str) -> str:
    """Pull the X-GM-LABELS list out of a FETCH response."""
    match = re.search(r"X-GM-LABELS\s*\(([^)]*)\)", raw)
    if not match:
        return ""
    labels = [lbl.strip('"\\ ') for lbl in match.group(1).split()]
    # System labels arrive as \\Important; drop the prefix so they read cleanly
    cleaned = [lbl.lstrip("\\") for lbl in labels if lbl and lbl not in ("\\\\Inbox",)]
    return ", ".join(cleaned)


# --------------------------------------------------------------------------- IMAP


async def _with_inbox(fn):
    """Run fn against a connected INBOX, always disconnecting afterwards."""
    account = _pick_account()
    client = ImapSmtpPasswordClient(account, imap_timeout=config.IMAP_TIMEOUT)
    await client.connect()
    try:
        await client.select_mailbox("INBOX")
        return await fn(client, account)
    finally:
        try:
            await client.close()
        except Exception as e:
            core.logger.warning(f"[gmail] failed to close IMAP connection: {e}")


def _as_text(data) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


# --------------------------------------------------------------------------- prompt


@plugin.mount_prompt_inject_method(name="gmail_query_syntax")
async def inject_query_syntax(_ctx: AgentCtx) -> str:
    return (
        "`search_gmail` takes a raw Gmail search query — the exact syntax of the Gmail search box, "
        "not IMAP syntax. Useful operators: from: to: subject: label: has:attachment is:unread "
        "is:starred newer_than:3d older_than:1y after:2026/08/01 category:primary category:promotions "
        "filename:pdf larger:5M, quoted phrases, OR, and - to exclude. Combine them freely, e.g. "
        "`from:stripe newer_than:30d has:attachment` or `is:unread -category:promotions newer_than:2d`. "
        "Prefer a narrow query over a broad one: the mailbox is large and only the first "
        f"{config.MAX_RESULTS} results come back, newest first."
    )


# --------------------------------------------------------------------------- methods


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="搜索邮件",
    description="用 Gmail 原生搜索语法查收件箱，返回匹配邮件的日期/发件人/主题/标签与 uid",
)
async def search_gmail(_ctx: AgentCtx, query: str, limit: int = 0) -> str:
    """Search the mailbox using Gmail's native search syntax.

    Args:
        query: A Gmail search-box query, e.g. "from:stripe newer_than:7d has:attachment".
        limit: Max results to return; 0 uses the configured default.

    Returns:
        A formatted list of matching messages, newest first, each with a uid for read_gmail.

    Example:
        search_gmail("is:unread newer_than:2d -category:promotions")
    """
    query = (query or "").strip()
    if not query:
        return "[Gmail] Empty query. Pass a Gmail search expression, e.g. 'is:unread newer_than:2d'."

    cap = limit if limit and limit > 0 else config.MAX_RESULTS

    async def run(client: ImapSmtpPasswordClient, account: EmailAccount) -> str:
        status, data = await client.uid_command("SEARCH", "X-GM-RAW", f'"{query}"')
        if status != "OK":
            return f"[Gmail] Search failed (status={status}) for query: {query}"

        uids = (data[0] or b"").split() if data else []
        if not uids:
            return f"[Gmail] {account.USERNAME} has no messages matching `{query}`."

        total = len(uids)
        # UIDs ascend roughly with time, so take the tail and reverse for newest-first
        picked = uids[-cap:][::-1]

        header = f"[Gmail] {account.USERNAME} — `{query}` matched {total} message(s)"
        header += f", newest {len(picked)}:" if total > len(picked) else ":"
        lines = [header, ""]

        for uid in picked:
            uid_s = uid.decode()
            st, fd = await client.uid_command(
                "FETCH",
                uid_s,
                "(X-GM-LABELS BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])",
            )
            if st != "OK" or not fd:
                lines.append(f"- uid {uid_s} (fetch failed)")
                continue

            raw_meta = "".join(_as_text(p[0]) for p in fd if isinstance(p, tuple) and p)
            header_bytes = b"".join(p[1] for p in fd if isinstance(p, tuple) and len(p) > 1 and p[1])
            hdr = email.message_from_bytes(header_bytes)

            sender = _decode_header(hdr.get("From"))
            subject = _decode_header(hdr.get("Subject")) or "(no subject)"
            date_hdr = hdr.get("Date") or ""
            try:
                when = email.utils.parsedate_to_datetime(date_hdr).astimezone().strftime("%Y-%m-%d %H:%M")
            except Exception:
                when = date_hdr[:25]
            labels = _parse_labels(raw_meta)

            lines.append(f"- [{when}] {sender}")
            lines.append(f"  {subject}")
            lines.append(f"  uid={uid_s}" + (f" · {labels}" if labels else ""))

        lines.append("")
        lines.append("Use read_gmail(uid) for a full body. None of these were marked read.")
        return "\n".join(lines)

    try:
        return await _with_inbox(run)
    except Exception as e:
        core.logger.exception("[gmail] search failed")
        return f"[Gmail] Search error: {type(e).__name__}: {e}"


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="读取邮件",
    description="按 uid 读取一封邮件的完整正文（不会标记为已读）",
)
async def read_gmail(_ctx: AgentCtx, uid: str) -> str:
    """Read one message in full by its uid.

    Args:
        uid: The uid shown by search_gmail.

    Returns:
        Headers plus the plain-text body, truncated to the configured length.

    Example:
        read_gmail("48213")
    """
    uid = str(uid or "").strip()
    if not uid.isdigit():
        return f"[Gmail] Invalid uid {uid!r}. Use a uid returned by search_gmail."

    async def run(client: ImapSmtpPasswordClient, account: EmailAccount) -> str:
        status, data = await client.uid_command("FETCH", uid, "(BODY.PEEK[])")
        if status != "OK" or not data:
            return f"[Gmail] Could not read uid={uid} (status={status}); it may be deleted or not in INBOX."

        raw = b"".join(p[1] for p in data if isinstance(p, tuple) and len(p) > 1 and p[1])
        if not raw:
            return f"[Gmail] uid={uid} returned no content."

        msg = email.message_from_bytes(raw)
        body, attachments = _extract_body(msg)

        date_hdr = msg.get("Date") or ""
        try:
            when = email.utils.parsedate_to_datetime(date_hdr).astimezone().strftime("%Y-%m-%d %H:%M")
        except Exception:
            when = date_hdr[:25]

        head = [
            f"[Gmail] {account.USERNAME} — uid={uid}",
            f"From   : {_decode_header(msg.get('From'))}",
            f"To     : {_decode_header(msg.get('To'))}",
            f"Date   : {when}",
            f"Subject: {_decode_header(msg.get('Subject')) or '(no subject)'}",
        ]
        if attachments:
            head.append(f"Files  : {', '.join(attachments)}")
        head.append("")

        if not body:
            head.append("(No readable text body — this message may be image-only or attachment-only.)")
            return "\n".join(head)

        if len(body) > config.BODY_CHARS:
            hidden = len(body) - config.BODY_CHARS
            body = body[: config.BODY_CHARS] + f"\n\n...(body truncated, {hidden} characters omitted)"
        head.append(body)
        head.append("")
        head.append("(This read did not mark the message as read.)")
        return "\n".join(head)

    try:
        return await _with_inbox(run)
    except Exception as e:
        core.logger.exception("[gmail] read failed")
        return f"[Gmail] Read error: {type(e).__name__}: {e}"


@plugin.mount_cleanup_method()
async def clean_up():
    """Nothing persistent to release: IMAP connections close after each call."""
    return
