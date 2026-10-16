"""Smoke tests for the Streamlit page using Streamlit's headless AppTest runner."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from churn import client as client_module

APP = str(Path(__file__).parents[1] / "app" / "streamlit_app.py")


@pytest.fixture(autouse=True)
def clear_streamlit_caches():
    # cache_resource/cache_data persist across AppTest runs in one process
    st.cache_resource.clear()
    st.cache_data.clear()
    yield


@pytest.fixture
def app_with_api(api_session, monkeypatch):
    """Route the app's HTTP calls to the in-process API instead of the network."""

    class InProcessClient(client_module.ChurnApiClient):
        def __init__(self, base_url, **kwargs):
            super().__init__("http://testserver", session=api_session)

    monkeypatch.setattr(client_module, "ChurnApiClient", InProcessClient)
    return AppTest.from_file(APP, default_timeout=60)


def test_renders_when_api_is_down(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Could not reach the API" in e.value for e in at.error)


def test_sample_data_flow(app_with_api):
    at = app_with_api.run()
    assert not at.exception
    assert any("API connected" in s.value for s in at.success)

    at.button[0].click().run()  # "Try with sample data"
    assert not at.exception
    scored = {m.label: m.value for m in at.metric}
    assert scored["Customers scored"] == "200"


def test_single_customer_flow(app_with_api):
    at = app_with_api.run()
    at.button[1].click().run()  # "Score customer"
    assert not at.exception
    assert "Churn probability" in [m.label for m in at.metric]
