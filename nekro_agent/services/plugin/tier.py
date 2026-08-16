"""Per-channel capability scoping.

`PLUGIN_ENABLED` is global: `collector.py` flips `plugin._is_enabled` once at load time,
so every channel sees every plugin. That is fine while every channel is your own, and
wrong the moment the bot joins a guild you do not control — anyone who can post there can
make it read your mail, and everyone there reads the answer.

Two tiers, because input trust and output disclosure move together for the channels that
actually exist here:

    private   only you write, only you read
    public    anyone else can write or read  (the default)

Public is the default deliberately. A per-channel allowlist fails open — a new channel is
wide open until someone remembers to lock it, and the omission is silent. Defaulting to
untrusted means the failure mode is a tool that refuses, which is loud and easy to fix.

Plugins default the other way, and that asymmetry is intentional: a new *channel* may
contain people you never chose, whereas a new *plugin* is something you installed on
purpose. So plugins are public-safe unless listed in `PLUGIN_REQUIRE_PRIVATE`.

Enforcement lives in two places, because visibility is not authority:

  * `render_plugins_prompt` hides the block  — saves tokens, sets expectations
  * `routers/rpc.py` rejects the call        — the actual gate

Hiding alone would be theatre: predefined methods are RPC calls, so a sandbox can invoke
one by name whether or not the prompt advertised it. The RPC check is only trustworthy
because the gateway resolves the channel from the server-side session registry rather
than the caller's self-reported `from_chat_key`.
"""

from typing import Optional

from nekro_agent.core.config import CoreConfig, config
from nekro_agent.core.logger import get_sub_logger
from nekro_agent.services.plugin.collector import plugin_collector

logger = get_sub_logger("plugin_tier")

TIER_PRIVATE = "private"
TIER_PUBLIC = "public"


def normalize_tier(raw: Optional[str]) -> str:
    """Coerce a configured tier to a known value, treating anything unrecognised as public."""
    value = (raw or "").strip().lower()
    if value == TIER_PRIVATE:
        return TIER_PRIVATE
    if value and value != TIER_PUBLIC:
        logger.warning(f"未知的频道信任等级 {raw!r}，按 public 处理")
    return TIER_PUBLIC


def channel_tier(effective_config: CoreConfig) -> str:
    """Read the tier out of a channel's effective (override-merged) config."""
    return normalize_tier(getattr(effective_config, "CHANNEL_TIER", TIER_PUBLIC))


def plugin_requires_private(plugin_key: str, effective_config: Optional[CoreConfig] = None) -> bool:
    """Whether this plugin key is restricted to private channels."""
    cfg = effective_config or config
    return plugin_key in (getattr(cfg, "PLUGIN_REQUIRE_PRIVATE", None) or [])


def is_plugin_allowed(plugin_key: str, effective_config: CoreConfig) -> bool:
    """Whether a plugin may be used in the channel this config was resolved for."""
    if not plugin_requires_private(plugin_key, effective_config):
        return True
    return channel_tier(effective_config) == TIER_PRIVATE


def plugin_key_for_method(method_name: str) -> Optional[str]:
    """Map a sandbox method name back to its owning plugin key.

    `plugin_collector.get_method` returns the bare function, which is enough to call but
    not enough to authorize; the gate needs to know whose method it is.
    """
    for key, plugin in plugin_collector.loaded_plugins.items():
        for method in plugin.sandbox_methods:
            if method.func.__name__ == method_name:
                return key
    return None


def denial_message(plugin_key: str, method_name: str) -> str:
    """The refusal the model sees. Explicit on purpose.

    Silently omitting the method would leave the model retrying against a name it believes
    exists, burning iterations on a call that can never succeed. Saying so plainly lets it
    change approach or tell the user, and it names the fix for whoever reads the log.
    """
    return (
        f"`{method_name}` is not available in this channel. The plugin `{plugin_key}` is "
        f"restricted to private channels because it reads personal data or has external "
        f"side effects, and this channel is public. Do not retry it here — either continue "
        f"without it, or tell the user it is only available in a private channel."
    )
