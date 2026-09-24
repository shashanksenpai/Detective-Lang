"""The Gemini judge is a placeholder (BACKLOG N-1): it must be off by default, say
exactly what is missing, and never return an invented verdict. Run: pytest test_judge.py
"""
import subprocess
import sys

import pytest

from judge import GeminiJudge, Judge, JudgeUnavailable, RELATIONS, Statement, StatementPair, Verdict


def _pair():
    a = Statement(1, "Yash", "Class Group", "group", None, "asleep by 11")
    b = Statement(2, "Yash", "Class Group", "group", None, "just checking one last thing")
    return StatementPair(a, b)


@pytest.fixture()
def clean_env(monkeypatch):
    for var in (GeminiJudge.ENV_CONSENT, GeminiJudge.ENV_KEY, GeminiJudge.ENV_MODEL):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_it_is_off_by_default_and_the_first_thing_it_asks_for_is_consent(clean_env):
    reason = GeminiJudge().unavailable_reason()
    assert GeminiJudge.ENV_CONSENT in reason
    assert "free tier" in reason                    # the privacy warning is part of the message


def test_each_missing_setting_is_named_in_turn(clean_env):
    clean_env.setenv(GeminiJudge.ENV_CONSENT, "1")
    assert GeminiJudge.ENV_KEY in GeminiJudge().unavailable_reason()
    clean_env.setenv(GeminiJudge.ENV_KEY, "not-a-real-key")
    assert GeminiJudge.ENV_MODEL in GeminiJudge().unavailable_reason()
    clean_env.setenv(GeminiJudge.ENV_MODEL, "some-model")
    assert GeminiJudge().unavailable_reason() is None


@pytest.mark.parametrize("value", ["", "0", "true", "yes", "on"])
def test_only_an_explicit_1_switches_external_use_on(clean_env, value):
    clean_env.setenv(GeminiJudge.ENV_CONSENT, value)
    clean_env.setenv(GeminiJudge.ENV_KEY, "not-a-real-key")
    clean_env.setenv(GeminiJudge.ENV_MODEL, "some-model")
    assert GeminiJudge.ENV_CONSENT in GeminiJudge().unavailable_reason()


def test_unconfigured_it_raises_judge_unavailable_with_the_reason(clean_env):
    with pytest.raises(JudgeUnavailable) as err:
        GeminiJudge().judge(_pair())
    assert GeminiJudge.ENV_CONSENT in str(err.value)


def test_fully_configured_it_is_still_a_placeholder_and_never_invents_a_verdict(clean_env):
    clean_env.setenv(GeminiJudge.ENV_CONSENT, "1")
    clean_env.setenv(GeminiJudge.ENV_KEY, "not-a-real-key")
    clean_env.setenv(GeminiJudge.ENV_MODEL, "some-model")
    with pytest.raises(NotImplementedError):
        GeminiJudge().judge(_pair())


def test_a_verdict_can_only_be_one_of_the_allowed_relations():
    for relation in RELATIONS:
        assert Verdict(relation, "why", "gemini:some-model").relation == relation
    for bad in ("guilty", "lying", "", "CONFLICT"):
        with pytest.raises(ValueError):
            Verdict(bad, "why", "gemini:some-model")


def test_uncertain_is_a_first_class_answer():
    assert "uncertain" in RELATIONS


def test_the_placeholder_satisfies_the_judge_interface():
    judge: Judge = GeminiJudge()
    assert judge.name == "gemini"
    assert callable(judge.unavailable_reason) and callable(judge.judge)


def test_importing_the_module_does_not_pull_in_the_google_sdk_or_the_ml_stack():
    code = "import sys, judge; print(sorted(m for m in ('google.genai', 'torch', 'sentence_transformers') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
