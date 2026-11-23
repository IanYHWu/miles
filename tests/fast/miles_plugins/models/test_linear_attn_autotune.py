"""fla's size-bucketed GDN kernels are tuned once per shape, not once per packed length."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from miles_plugins.models import linear_attn


@pytest.fixture
def fake_fla(monkeypatch):
    kernels = {}
    for module_name, names in linear_attn._SIZE_KEYED_AUTOTUNE_KERNELS.items():
        module = ModuleType(module_name)
        for name in names:
            tuner = SimpleNamespace(keys=["D", "NB"])
            kernels[name] = tuner
            # the bwd kernels sit under @triton.heuristics, which wraps the autotuner as .fn
            setattr(module, name, SimpleNamespace(fn=tuner) if name.endswith("bwd_kernel") else tuner)
        monkeypatch.setitem(sys.modules, module_name, module)
    linear_attn._freeze_size_keyed_autotune.cache_clear()
    yield kernels
    linear_attn._freeze_size_keyed_autotune.cache_clear()


def test_the_size_bucket_is_dropped_from_every_autotune_key(fake_fla, monkeypatch):
    monkeypatch.delenv("MILES_FLA_SIZE_KEYED_AUTOTUNE", raising=False)
    linear_attn._freeze_size_keyed_autotune()
    linear_attn._freeze_size_keyed_autotune()  # once only
    assert all(tuner.keys == ["D"] for tuner in fake_fla.values())


def test_the_environment_keeps_fla_keys(fake_fla, monkeypatch):
    monkeypatch.setenv("MILES_FLA_SIZE_KEYED_AUTOTUNE", "1")
    linear_attn._freeze_size_keyed_autotune()
    assert all(tuner.keys == ["D", "NB"] for tuner in fake_fla.values())
