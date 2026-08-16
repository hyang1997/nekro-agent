"""Per-channel capability scoping.

`PLUGIN_ENABLED` is global, so before this every channel saw every plugin. The tier gate
is what keeps a public channel from reaching plugins that read personal data.

The property that matters most: public is the *default*. A per-channel allowlist fails
open — a channel nobody configured is wide open and the omission is silent.
"""

import pytest

from nekro_agent.services.plugin.tier import (
    TIER_PRIVATE,
    TIER_PUBLIC,
    channel_tier,
    denial_message,
    is_plugin_allowed,
    normalize_tier,
    plugin_requires_private,
)


class _Cfg:
    """Stand-in for a channel's effective config."""

    def __init__(self, tier=None, private_plugins=None):
        if tier is not None:
            self.CHANNEL_TIER = tier
        self.PLUGIN_REQUIRE_PRIVATE = private_plugins if private_plugins is not None else []


class TestNormalizeTier:
    def test_private(self):
        assert normalize_tier("private") is TIER_PRIVATE

    def test_public(self):
        assert normalize_tier("public") is TIER_PUBLIC

    @pytest.mark.parametrize("raw", ["PRIVATE", " Private ", "pRiVaTe"])
    def test_case_and_whitespace_insensitive(self, raw: str):
        assert normalize_tier(raw) is TIER_PRIVATE

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_missing_is_public(self, raw):
        assert normalize_tier(raw) is TIER_PUBLIC

    @pytest.mark.parametrize("raw", ["secret", "dm", "trusted", "0", "true"])
    def test_unknown_value_is_public_not_private(self, raw: str):
        """A typo must fail closed. 'trusted' looking private-ish would be the bad outcome."""
        assert normalize_tier(raw) is TIER_PUBLIC


class TestChannelTier:
    def test_reads_configured_tier(self):
        assert channel_tier(_Cfg(tier="private")) is TIER_PRIVATE

    def test_absent_attribute_defaults_public(self):
        """A config predating this feature must not silently grant private access."""
        assert channel_tier(_Cfg()) is TIER_PUBLIC


class TestPluginRequiresPrivate:
    def test_listed_plugin(self):
        assert plugin_requires_private("Hao.gmail", _Cfg(private_plugins=["Hao.gmail"]))

    def test_unlisted_plugin(self):
        assert not plugin_requires_private("KroMiose.basic", _Cfg(private_plugins=["Hao.gmail"]))

    def test_empty_list(self):
        assert not plugin_requires_private("Hao.gmail", _Cfg(private_plugins=[]))

    def test_new_plugin_defaults_public_safe(self):
        """Plugins default permissive: you chose to install one, unlike who joins a channel."""
        assert not plugin_requires_private("Someone.brand_new", _Cfg(private_plugins=["Hao.gmail"]))


class TestIsPluginAllowed:
    PRIVATE_PLUGINS = ["Hao.gmail", "Hao.health", "KroMiose.cc_workspace"]

    def test_private_plugin_in_private_channel(self):
        cfg = _Cfg(tier="private", private_plugins=self.PRIVATE_PLUGINS)
        assert is_plugin_allowed("Hao.gmail", cfg)

    def test_private_plugin_in_public_channel(self):
        cfg = _Cfg(tier="public", private_plugins=self.PRIVATE_PLUGINS)
        assert not is_plugin_allowed("Hao.gmail", cfg)

    def test_ordinary_plugin_everywhere(self):
        for tier in ("private", "public"):
            cfg = _Cfg(tier=tier, private_plugins=self.PRIVATE_PLUGINS)
            assert is_plugin_allowed("KroMiose.basic", cfg)

    def test_unconfigured_channel_blocks_private_plugins(self):
        """The headline property: a channel nobody configured does not get personal data."""
        assert not is_plugin_allowed("Hao.gmail", _Cfg(private_plugins=self.PRIVATE_PLUGINS))

    def test_typo_in_tier_blocks_rather_than_grants(self):
        cfg = _Cfg(tier="privte", private_plugins=self.PRIVATE_PLUGINS)
        assert not is_plugin_allowed("Hao.gmail", cfg)

    @pytest.mark.parametrize("key", PRIVATE_PLUGINS)
    def test_every_listed_plugin_is_blocked_publicly(self, key: str):
        assert not is_plugin_allowed(key, _Cfg(tier="public", private_plugins=self.PRIVATE_PLUGINS))


class TestDenialMessage:
    def test_names_the_method_and_plugin(self):
        msg = denial_message("Hao.gmail", "search_gmail")
        assert "search_gmail" in msg
        assert "Hao.gmail" in msg

    def test_tells_the_model_not_to_retry(self):
        """Silent omission would leave it retrying a name it believes exists, burning iterations."""
        assert "not retry" in denial_message("Hao.gmail", "search_gmail").lower()

    def test_offers_a_way_forward(self):
        msg = denial_message("Hao.gmail", "search_gmail").lower()
        assert "without it" in msg or "tell the user" in msg

    def test_explains_the_reason(self):
        assert "private" in denial_message("Hao.gmail", "search_gmail").lower()
