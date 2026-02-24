import base64
import json
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="TruthShield - AI Misinformation Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
                max-width: 1400px;
            }
            .verdict-real {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 12px;
                text-align: center;
                font-size: 1.5rem;
                font-weight: 700;
                margin: 1rem 0;
                box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);
            }
            .verdict-fake {
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 12px;
                text-align: center;
                font-size: 1.5rem;
                font-weight: 700;
                margin: 1rem 0;
                box-shadow: 0 4px 6px rgba(239, 68, 68, 0.3);
            }
            .confidence-label {
                font-size: 0.9rem;
                opacity: 0.9;
                margin-bottom: 0.5rem;
            }
            .stat-card {
                background: rgba(99, 102, 241, 0.1);
                border-left: 4px solid #6366f1;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
            }
            .stat-value {
                font-size: 2rem;
                font-weight: 700;
                color: #6366f1;
            }
            .stat-label {
                font-size: 0.875rem;
                opacity: 0.8;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults: dict[str, Any] = {
        "base_url": "http://localhost:8000",
        "auth_value": "",
        "auth_kind": "API Key",
        "last_job_id": "",
        "last_request_id": "",
        "developer_mode": False,
        "logged_in": False,
        "username": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def auth_headers() -> dict[str, str]:
    if not st.session_state.auth_value.strip():
        return {}
    return {"Authorization": f"Bearer {st.session_state.auth_value.strip()}"}


def api_call(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    with_auth: bool = True,
) -> tuple[bool, Any, int]:
    url = f"{st.session_state.base_url.rstrip('/')}{path}"
    headers = {"Content-Type": "application/json"}
    if with_auth:
        headers.update(auth_headers())

    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            json=payload,
            params=params,
            timeout=45,
        )
        try:
            body = response.json()
        except Exception:
            body = {"raw": response.text}
        return response.ok, body, response.status_code
    except requests.RequestException as exc:
        return False, {"error": str(exc)}, 0


def show_result(ok: bool, body: Any, status_code: int) -> None:
    if ok:
        st.success(f"Success ({status_code})")
        st.json(body)
    else:
        st.error(f"Request failed ({status_code})")
        st.json(body)


def show_detection_result(ok: bool, body: Any, status_code: int) -> None:
    """User-friendly detection result display"""
    if not ok:
        st.error(f"❌ Detection failed ({status_code})")
        if isinstance(body, dict) and "detail" in body:
            st.warning(body["detail"])
        if st.session_state.developer_mode:
            with st.expander("🔧 Debug Info"):
                st.json(body)
        return
    
    if not isinstance(body, dict):
        st.error("Invalid response format")
        return
    
    verdict = body.get("verdict", "Unknown")
    confidence = body.get("confidence", 0.0)
    probabilities = body.get("probabilities", {})
    processing_time = body.get("processing_time_ms", 0)
    model = body.get("model", "unknown")
    
    # Verdict display
    if verdict == "Real":
        st.markdown(
            f"<div class='verdict-real'>✅ Likely REAL</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='verdict-fake'>⚠️ Likely FAKE</div>",
            unsafe_allow_html=True,
        )
    
    # Confidence meter
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<div class='confidence-label'>Confidence Level</div>", unsafe_allow_html=True)
        st.progress(confidence, text=f"{confidence:.1%}")
    with col2:
        st.metric("Model", model.upper())
    
    # Probabilities breakdown
    if probabilities:
        st.subheader("Detailed Probabilities")
        prob_col1, prob_col2 = st.columns(2)
        prob_col1.metric("Real", f"{probabilities.get('Real', 0):.1%}", delta=None)
        prob_col2.metric("Fake", f"{probabilities.get('Fake', 0):.1%}", delta=None)
    
    # Processing info
    st.caption(f"⚡ Analyzed in {processing_time:.0f}ms")
    
    # Store request ID for history/feedback
    if body.get("request_id"):
        st.session_state.last_request_id = body["request_id"]
        st.info(f"🔖 Request ID: `{body['request_id']}` (saved for feedback)")
    
    # Developer mode JSON
    if st.session_state.developer_mode:
        with st.expander("🔧 Raw Response"):
            st.json(body)


def to_base64(uploaded_file) -> str:
    content = uploaded_file.read()
    return base64.b64encode(content).decode("utf-8")


def section_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def render_sidebar() -> str:
    with st.sidebar:
        st.title("🛡️ TruthShield")
        st.caption("AI-Powered Misinformation Detection")

        # Auth section
        if not st.session_state.logged_in:
            st.info("👤 Login to access your history & stats")
            with st.expander("🔐 Login / Register", expanded=False):
                # Tabs for Login vs Register
                tab_login, tab_register = st.tabs(["Login", "Register"])
                
                with tab_login:
                    st.caption("Login with username/password or API key")
                    
                    login_method = st.radio(
                        "Method",
                        options=["Username & Password", "API Key"],
                        key="login_method",
                        label_visibility="collapsed",
                    )
                    
                    if login_method == "Username & Password":
                        login_username = st.text_input("Username", key="login_user")
                        login_password = st.text_input("Password", type="password", key="login_pass")
                        
                        if st.button("🔑 Login", use_container_width=True, type="primary"):
                            if not login_username.strip() or not login_password.strip():
                                st.error("❌ Enter username and password")
                            else:
                                with st.spinner("Logging in..."):
                                    ok, body, _ = api_call(
                                        "POST",
                                        "/api/v1/auth/login",
                                        payload={"username": login_username, "password": login_password},
                                        with_auth=False,
                                    )
                                    if ok and isinstance(body, dict) and body.get("access_token"):
                                        st.session_state.auth_kind = "JWT Token"
                                        st.session_state.auth_value = body["access_token"]
                                        st.session_state.logged_in = True
                                        st.session_state.username = login_username
                                        st.success(f"✅ Welcome back, {login_username}!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Invalid username or password")
                    
                    else:  # API Key
                        api_key_input = st.text_input(
                            "API Key",
                            type="password",
                            placeholder="ts_xxx...",
                            key="api_key_login",
                        )
                        
                        if st.button("🔑 Login with Key", use_container_width=True, type="primary"):
                            if not api_key_input.strip():
                                st.error("❌ Enter your API key")
                            else:
                                with st.spinner("Verifying key..."):
                                    st.session_state.auth_kind = "API Key"
                                    st.session_state.auth_value = api_key_input.strip()
                                    ok, body, _ = api_call("GET", "/api/v1/auth/users/me")
                                    if ok and isinstance(body, dict):
                                        st.session_state.logged_in = True
                                        st.session_state.username = body.get("username", "User")
                                        st.success(f"✅ Logged in as {st.session_state.username}")
                                        st.rerun()
                                    else:
                                        st.session_state.auth_value = ""
                                        st.error("❌ Invalid API key")
                
                with tab_register:
                    st.caption("Create a new account")
                    
                    reg_username = st.text_input("Username", key="reg_user", placeholder="Choose a username")
                    reg_email = st.text_input("Email", key="reg_email", placeholder="your@email.com")
                    reg_password = st.text_input("Password", type="password", key="reg_pass", placeholder="Min 6 characters")
                    
                    if st.button("✨ Create Account", use_container_width=True, type="primary"):
                        if not reg_username.strip() or not reg_email.strip() or not reg_password.strip():
                            st.error("❌ All fields are required")
                        elif len(reg_password) < 6:
                            st.error("❌ Password must be at least 6 characters")
                        else:
                            with st.spinner("Creating account..."):
                                ok, body, _ = api_call(
                                    "POST",
                                    "/api/v1/auth/register",
                                    payload={
                                        "username": reg_username,
                                        "email": reg_email,
                                        "password": reg_password,
                                    },
                                    with_auth=False,
                                )
                                if ok and isinstance(body, dict):
                                    # Auto-login with the API key from registration
                                    if body.get("api_key"):
                                        st.session_state.auth_kind = "API Key"
                                        st.session_state.auth_value = body["api_key"]
                                        st.session_state.logged_in = True
                                        st.session_state.username = reg_username
                                        st.success(f"🎉 Account created! Welcome, {reg_username}!")
                                        st.rerun()
                                    else:
                                        st.success("✅ Account created! Please login.")
                                else:
                                    error_msg = body.get("detail", "Registration failed") if isinstance(body, dict) else "Registration failed"
                                    st.error(f"❌ {error_msg}")
                
                # Advanced settings
                with st.expander("⚙️ Advanced", expanded=False):
                    st.session_state.base_url = st.text_input(
                        "API URL",
                        value=st.session_state.base_url,
                        help="Backend API endpoint",
                    )
        
        else:
            # Logged in state
            st.success(f"👤 **{st.session_state.username}**")
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.auth_value = ""
                st.session_state.username = ""
                st.rerun()

        st.markdown("---")
        
        # Developer mode toggle
        st.session_state.developer_mode = st.checkbox(
            "🔧 Developer Mode",
            value=st.session_state.developer_mode,
            help="Show raw JSON, admin tools, and advanced features"
        )
        
        st.markdown("---")
        
        # Navigation
        if st.session_state.developer_mode:
            return st.selectbox(
                "Navigate",
                [
                    "🏠 Home",
                    "🔍 Detect Content",
                    "📊 My Dashboard",
                    "📜 History",
                    "💬 Give Feedback",
                    "---",
                    "🔐 Auth (Dev)",
                    "⚙️ Jobs & APIs (Dev)",
                    "🧠 Explainability (Dev)",
                    "🎛️ Admin Panel (Dev)",
                ],
            )
        else:
            return st.selectbox(
                "Navigate",
                [
                    "🏠 Home",
                    "🔍 Detect Content",
                    "📊 My Dashboard",
                    "📜 History",
                    "💬 Give Feedback",
                ],
            )


def render_home() -> None:
    st.title("🛡️ Welcome to TruthShield")
    st.caption("AI-powered misinformation detection for news, messages, and social media")
    
    st.markdown("---")
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📰 News Detection")
        st.write("Analyze news articles and headlines for credibility using advanced AI models.")
        
    with col2:
        st.subheader("📱 SMS/Email Phishing")
        st.write("Detect phishing attempts and spam messages with high accuracy.")
        
    with col3:
        st.subheader("🖼️ Multimodal Analysis")
        st.write("Verify claims combining both text and images for comprehensive fact-checking.")
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("System Status")
    col1, col2, col3, col4 = st.columns(4)
    
    ok, health_body, _ = api_call("GET", "/api/v1/models/health", with_auth=False)
    if ok and isinstance(health_body, dict):
        models_loaded = health_body.get("models_loaded", {})
        status = health_body.get("status", "unknown")
        
        with col1:
            status_color = "🟢" if status == "healthy" else "🟡"
            st.markdown(f"<div class='stat-card'><div class='stat-value'>{status_color}</div><div class='stat-label'>System Status</div></div>", unsafe_allow_html=True)
        
        with col2:
            news_status = "✅" if models_loaded.get("news") else "❌"
            st.markdown(f"<div class='stat-card'><div class='stat-value'>{news_status}</div><div class='stat-label'>News Model</div></div>", unsafe_allow_html=True)
        
        with col3:
            sms_status = "✅" if models_loaded.get("sms") else "❌"
            st.markdown(f"<div class='stat-card'><div class='stat-value'>{sms_status}</div><div class='stat-label'>SMS Model</div></div>", unsafe_allow_html=True)
        
        with col4:
            multi_status = "✅" if models_loaded.get("multimodal") else "❌"
            st.markdown(f"<div class='stat-card'><div class='stat-value'>{multi_status}</div><div class='stat-label'>Multimodal</div></div>", unsafe_allow_html=True)
    
    # User stats if logged in
    if st.session_state.logged_in:
        st.markdown("---")
        st.subheader("Your Activity")
        ok, stats_body, _ = api_call("GET", "/api/v1/stats")
        if ok and isinstance(stats_body, dict):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total = stats_body.get("total_requests", 0)
                st.markdown(f"<div class='stat-card'><div class='stat-value'>{total}</div><div class='stat-label'>Total Checks</div></div>", unsafe_allow_html=True)
            
            with col2:
                remaining = stats_body.get("quota_remaining", 0)
                st.markdown(f"<div class='stat-card'><div class='stat-value'>{remaining}</div><div class='stat-label'>Quota Remaining</div></div>", unsafe_allow_html=True)
            
            with col3:
                limit = stats_body.get("quota_limit", 0)
                st.markdown(f"<div class='stat-card'><div class='stat-value'>{limit}</div><div class='stat-label'>Daily Limit</div></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("💡 **Get Started:** Use the 🔍 Detect Content page to analyze any text, message, or article.")


def render_dashboard() -> None:
    section_header("Control Center", "Overview of API health, models, and your account stats.")

    col1, col2, col3 = st.columns(3)

    if col1.button("System Health", use_container_width=True):
        ok, body, status = api_call("GET", "/api/v1/models/health", with_auth=False)
        show_result(ok, body, status)

    if col2.button("List Models", use_container_width=True):
        ok, body, status = api_call("GET", "/api/v1/models")
        show_result(ok, body, status)

    if col3.button("My Stats", use_container_width=True):
        ok, body, status = api_call("GET", "/api/v1/stats")
        show_result(ok, body, status)



def render_auth() -> None:
    section_header("Authentication", "Register users, login, refresh tokens, revoke key, and view current user.")

    tab_register, tab_login, tab_refresh, tab_revoke, tab_me = st.tabs(
        ["Register", "Login", "Refresh", "Revoke API Key", "Users Me"]
    )

    with tab_register:
        with st.form("register_form"):
            username = st.text_input("Username", key="reg_username")
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_password")
            submit = st.form_submit_button("Register")
        if submit:
            ok, body, status = api_call(
                "POST",
                "/api/v1/auth/register",
                payload={"username": username, "email": email, "password": password},
                with_auth=False,
            )
            show_result(ok, body, status)

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Login")
        if submit:
            ok, body, status = api_call(
                "POST",
                "/api/v1/auth/login",
                payload={"username": username, "password": password},
                with_auth=False,
            )
            show_result(ok, body, status)
            if ok and isinstance(body, dict) and body.get("access_token"):
                st.session_state.auth_kind = "JWT Access Token"
                st.session_state.auth_value = body["access_token"]
                st.info("Access token loaded into session credential field")

    with tab_refresh:
        if st.button("Refresh Access Token"):
            ok, body, status = api_call("POST", "/api/v1/auth/refresh")
            show_result(ok, body, status)
            if ok and isinstance(body, dict) and body.get("access_token"):
                st.session_state.auth_kind = "JWT Access Token"
                st.session_state.auth_value = body["access_token"]
                st.info("Refreshed token loaded into session")

    with tab_revoke:
        if st.button("Revoke and Regenerate API Key"):
            ok, body, status = api_call("DELETE", "/api/v1/auth/revoke")
            show_result(ok, body, status)

    with tab_me:
        if st.button("Get Current User"):
            ok, body, status = api_call("GET", "/api/v1/auth/users/me")
            show_result(ok, body, status)


def render_detection() -> None:
    st.title("🔍 Detect Content")
    st.caption("Analyze news, messages, or social media content for misinformation")
    
    # Model selector
    detection_type = st.radio(
        "What would you like to check?",
        options=["📰 News Article", "📱 SMS / Email", "🖼️ Post with Image", "🤖 Auto-detect"],
        horizontal=True,
    )
    
    st.markdown("---")
    
    if detection_type == "📰 News Article":
        st.subheader("News Article Detection")
        st.caption("Paste a news article, headline, or claim to verify")
        
        content = st.text_area(
            "Article Content",
            height=220,
            placeholder="Enter the news article text here...",
            help="Paste the full article text or headline you want to verify"
        )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            threshold = st.slider("Sensitivity", 0.5, 0.95, 0.7, 0.05, help="Higher = stricter fake detection")
        
        if st.button("🔍 Analyze News", type="primary", use_container_width=True):
            if not content.strip():
                st.warning("Please enter some content to analyze")
            else:
                with st.spinner("Analyzing article..."):
                    ok, body, status = api_call("POST", "/api/v1/detect/news", payload={"content": content, "threshold": threshold})
                    show_detection_result(ok, body, status)
    
    elif detection_type == "📱 SMS / Email":
        st.subheader("SMS / Email Phishing Detection")
        st.caption("Check if a message is legitimate or a phishing attempt")
        
        content = st.text_area(
            "Message Content",
            height=180,
            placeholder="Paste the SMS or email text here...",
            help="Enter the full message you received"
        )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            threshold = st.slider("Sensitivity", 0.5, 0.95, 0.7, 0.05)
        
        if st.button("🔍 Analyze Message", type="primary", use_container_width=True):
            if not content.strip():
                st.warning("Please enter a message to analyze")
            else:
                with st.spinner("Checking for phishing..."):
                    ok, body, status = api_call("POST", "/api/v1/detect/sms", payload={"content": content, "threshold": threshold})
                    show_detection_result(ok, body, status)
    
    elif detection_type == "🖼️ Post with Image":
        st.subheader("Multimodal Detection (Text + Image)")
        st.caption("Verify posts that combine text claims with images")
        
        content = st.text_area(
            "Post Text",
            height=140,
            placeholder="Enter the text from the post...",
        )
        
        image_source = st.radio("Image Source", ["URL", "Upload File"], horizontal=True)
        
        image_url = ""
        uploaded = None
        
        if image_source == "URL":
            image_url = st.text_input("Image URL", placeholder="https://example.com/image.jpg")
        else:
            uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"])
        
        col1, col2 = st.columns([3, 1])
        with col2:
            threshold = st.slider("Sensitivity", 0.5, 0.95, 0.7, 0.05)
        
        if st.button("🔍 Analyze Post", type="primary", use_container_width=True):
            if not content.strip():
                st.warning("Please enter post text")
            elif not image_url.strip() and not uploaded:
                st.warning("Please provide an image (URL or upload)")
            else:
                with st.spinner("Analyzing text and image..."):
                    payload: dict[str, Any] = {"content": content, "threshold": threshold}
                    if image_url.strip():
                        payload["image_url"] = image_url.strip()
                    elif uploaded:
                        payload["image_base64"] = to_base64(uploaded)
                    
                    ok, body, status = api_call("POST", "/api/v1/detect/multimodal", payload=payload)
                    show_detection_result(ok, body, status)
    
    else:  # Auto-detect
        st.subheader("Auto-Detection")
        st.caption("Let the AI automatically choose the best model for your content")
        
        content = st.text_area(
            "Content to Analyze",
            height=200,
            placeholder="Paste any text content here...",
            help="Enter any text and we'll automatically select the best detection model"
        )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            threshold = st.slider("Sensitivity", 0.5, 0.95, 0.7, 0.05)
        
        if st.button("🔍 Auto-Analyze", type="primary", use_container_width=True):
            if not content.strip():
                st.warning("Please enter some content to analyze")
            else:
                with st.spinner("Auto-detecting and analyzing..."):
                    ok, body, status = api_call("POST", "/api/v1/detect", payload={"content": content, "threshold": threshold})
                    show_detection_result(ok, body, status)
    
    # Developer mode: Show advanced options
    if st.session_state.developer_mode:
        with st.expander("🔧 Advanced: Batch Detection"):
            st.caption("Submit multiple items at once (JSON format)")
            default_items = [
                {"content": "Breaking: miracle cure discovered", "model_type": "news", "threshold": 0.7},
                {"content": "Win free cash now, click http://scam.com", "model_type": "sms", "threshold": 0.7},
            ]
            batch_text = st.text_area(
                "Batch Items JSON",
                value=json.dumps(default_items, indent=2),
                height=200,
            )
            if st.button("Run Batch Detection"):
                try:
                    items = json.loads(batch_text)
                    ok, body, status = api_call("POST", "/api/v1/detect/batch", payload={"items": items})
                    show_result(ok, body, status)
                except json.JSONDecodeError as exc:
                    st.error(f"Invalid JSON: {exc}")


def render_my_dashboard() -> None:
    st.title("📊 My Dashboard")
    st.caption("Your detection activity and statistics")
    
    if not st.session_state.logged_in:
        st.warning("🔐 Please login to view your dashboard")
        return
    
    # Fetch user stats
    ok, stats_body, _ = api_call("GET", "/api/v1/stats")
    if ok and isinstance(stats_body, dict):
        st.subheader("Usage Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total = stats_body.get("total_requests", 0)
            st.markdown(f"<div class='stat-card'><div class='stat-value'>{total}</div><div class='stat-label'>Total Checks</div></div>", unsafe_allow_html=True)
        
        with col2:
            remaining = stats_body.get("quota_remaining", 0)
            st.markdown(f"<div class='stat-card'><div class='stat-value'>{remaining}</div><div class='stat-label'>Quota Remaining</div></div>", unsafe_allow_html=True)
        
        with col3:
            quota_usage = (stats_body.get("total_requests", 0) / max(stats_body.get("quota_limit", 1), 1)) * 100
            st.markdown(f"<div class='stat-card'><div class='stat-value'>{quota_usage:.0f}%</div><div class='stat-label'>Quota Used</div></div>", unsafe_allow_html=True)
        
        # Model usage breakdown
        st.subheader("Detection by Model Type")
        requests_by_model = stats_body.get("requests_by_model", {})
        if requests_by_model:
            model_df = pd.DataFrame(list(requests_by_model.items()), columns=["Model", "Count"])
            st.bar_chart(model_df.set_index("Model"))
        else:
            st.info("No detections yet. Start analyzing content!")
    else:
        st.error("Failed to load stats")


def render_user_history() -> None:
    st.title("📜 History")
    st.caption("Browse your detection history")
    
    if not st.session_state.logged_in:
        st.warning("🔐 Please login to view your history")
        return
    
    # Pagination controls
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        page = st.number_input("Page", min_value=1, value=1, step=1)
    with col2:
        page_size = st.selectbox("Items per page", [10, 20, 50], index=1)
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Load history
    ok, body, status = api_call(
        "GET",
        "/api/v1/history",
        params={"page": int(page), "page_size": int(page_size)},
    )
    
    if ok and isinstance(body, dict):
        items = body.get("items", [])
        total = body.get("total", 0)
        
        st.caption(f"Showing {len(items)} of {total} total detections")
        
        if items:
            for item in items:
                with st.expander(
                    f"{item.get('model_type', '?').upper()} - {item.get('verdict', '?')} "
                    f"({item.get('confidence', 0):.0%}) - {item.get('created_at', '')}"
                ):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Request ID:**", item.get("request_id", "N/A"))
                        st.write("**Model:**", item.get("model_type", "N/A"))
                        st.write("**Verdict:**", item.get("verdict", "N/A"))
                    
                    with col2:
                        st.write("**Confidence:**", f"{item.get('confidence', 0):.1%}")
                        st.write("**Timestamp:**", item.get("created_at", "N/A"))
                    
                    # Show request ID for feedback
                    if st.button(f"📝 Give Feedback", key=f"fb_{item.get('request_id')}"):
                        st.session_state.last_request_id = item.get("request_id", "")
                        st.info("Request ID saved. Go to 'Give Feedback' page.")
        else:
            st.info("No history found. Start analyzing content!")
    else:
        st.error("Failed to load history")


def render_user_feedback() -> None:
    st.title("💬 Give Feedback")
    st.caption("Help improve TruthShield by reporting incorrect predictions")
    
    if not st.session_state.logged_in:
        st.warning("🔐 Please login to submit feedback")
        return
    
    st.info("💡 Found a detection error? Let us know the correct label to help improve the model!")
    
    request_id = st.text_input(
        "Request ID",
        value=st.session_state.last_request_id,
        help="The request ID from your detection (check History page)"
    )
    
    true_label = st.radio(
        "What is the correct label?",
        options=["Real", "Fake"],
        horizontal=True,
    )
    
    if st.button("📤 Submit Feedback", type="primary", use_container_width=True):
        if not request_id.strip():
            st.warning("Please enter a request ID")
        else:
            ok, body, status = api_call(
                "POST",
                "/api/v1/feedback",
                payload={"request_id": request_id.strip(), "true_label": true_label},
            )
            
            if ok:
                st.success("✅ Thank you! Your feedback will help improve TruthShield.")
                st.balloons()
            else:
                st.error(f"Failed to submit feedback: {body.get('detail', 'Unknown error')}")
                if st.session_state.developer_mode:
                    st.json(body)


def render_jobs() -> None:
    section_header("Async Jobs", "Submit detection jobs, poll status, fetch results, and cancel jobs.")

    tab_submit, tab_status, tab_result, tab_cancel = st.tabs(["Submit", "Status", "Result", "Cancel"])

    with tab_submit:
        with st.form("submit_job_form"):
            content = st.text_area("Content", height=140, key="job_content")
            model_type = st.selectbox("Model Type", ["auto", "news", "sms", "multimodal"], key="job_model_type")
            image_url = st.text_input("Image URL (optional)", key="job_image_url")
            threshold = st.slider("Threshold", 0.0, 1.0, 0.7, 0.01, key="job_threshold")
            submit = st.form_submit_button("Submit Job")
        if submit:
            payload: dict[str, Any] = {"content": content, "threshold": threshold}
            if model_type != "auto":
                payload["model_type"] = model_type
            if image_url.strip():
                payload["image_url"] = image_url.strip()

            ok, body, status = api_call("POST", "/api/v1/jobs/detect", payload=payload)
            show_result(ok, body, status)
            if ok and isinstance(body, dict):
                st.session_state.last_job_id = body.get("job_id", "")

    with tab_status:
        job_id = st.text_input("Job ID", value=st.session_state.last_job_id, key="status_job_id")
        if st.button("Check Job Status") and job_id.strip():
            ok, body, status = api_call("GET", f"/api/v1/jobs/{job_id.strip()}")
            show_result(ok, body, status)

    with tab_result:
        job_id = st.text_input("Job ID", value=st.session_state.last_job_id, key="result_job_id")
        if st.button("Get Job Result") and job_id.strip():
            ok, body, status = api_call("GET", f"/api/v1/jobs/{job_id.strip()}/result")
            show_result(ok, body, status)

    with tab_cancel:
        job_id = st.text_input("Job ID", value=st.session_state.last_job_id, key="cancel_job_id")
        if st.button("Cancel Job") and job_id.strip():
            ok, body, status = api_call("DELETE", f"/api/v1/jobs/{job_id.strip()}")
            show_result(ok, body, status)


def render_explainability() -> None:
    section_header("Explainability", "Inspect news/SMS explanations and test multimodal explain endpoint.")

    tab_news, tab_sms, tab_multi = st.tabs(["Explain News", "Explain SMS", "Explain Multimodal"])

    with tab_news:
        content = st.text_area("News Content", height=180, key="exp_news_content")
        if st.button("Run News Explanation"):
            ok, body, status = api_call("POST", "/api/v1/explain/news", payload={"content": content})
            show_result(ok, body, status)

    with tab_sms:
        content = st.text_area("SMS Content", height=130, key="exp_sms_content")
        if st.button("Run SMS Explanation"):
            ok, body, status = api_call("POST", "/api/v1/explain/sms", payload={"content": content})
            show_result(ok, body, status)

    with tab_multi:
        content = st.text_area("Multimodal Content", height=120, key="exp_multi_content")
        if st.button("Run Multimodal Explanation"):
            ok, body, status = api_call("POST", "/api/v1/explain/multimodal", payload={"content": content})
            show_result(ok, body, status)


def render_history() -> None:
    section_header("History & Analytics", "Browse request history, request details, and user/global stats.")

    tab_history, tab_detail, tab_stats, tab_global = st.tabs(
        ["History", "History Detail", "My Stats", "Global Stats"]
    )

    with tab_history:
        col1, col2 = st.columns(2)
        page = col1.number_input("Page", min_value=1, value=1)
        page_size = col2.number_input("Page Size", min_value=1, max_value=100, value=20)
        if st.button("Load History"):
            ok, body, status = api_call(
                "GET",
                "/api/v1/history",
                params={"page": int(page), "page_size": int(page_size)},
            )
            show_result(ok, body, status)
            if ok and isinstance(body, dict) and isinstance(body.get("items"), list):
                items = body.get("items", [])
                if items:
                    df = pd.DataFrame(items)
                    st.dataframe(df, use_container_width=True)

    with tab_detail:
        request_id = st.text_input("Request ID", value=st.session_state.last_request_id)
        if st.button("Get History Detail") and request_id.strip():
            ok, body, status = api_call("GET", f"/api/v1/history/{request_id.strip()}")
            show_result(ok, body, status)

    with tab_stats:
        if st.button("Get My Stats"):
            ok, body, status = api_call("GET", "/api/v1/stats")
            show_result(ok, body, status)

    with tab_global:
        if st.button("Get Global Stats (Admin)"):
            ok, body, status = api_call("GET", "/api/v1/stats/global")
            show_result(ok, body, status)


def render_feedback() -> None:
    section_header("Feedback & Retraining", "Submit label corrections and use admin retraining tools.")

    tab_submit, tab_queue, tab_trigger, tab_status = st.tabs(
        ["Submit Feedback", "Feedback Queue", "Trigger Retrain", "Retrain Status"]
    )

    with tab_submit:
        request_id = st.text_input("Request ID", value=st.session_state.last_request_id, key="fb_request_id")
        true_label = st.selectbox("True Label", ["Real", "Fake"], key="fb_true_label")
        if st.button("Submit Feedback") and request_id.strip():
            ok, body, status = api_call(
                "POST",
                "/api/v1/feedback",
                payload={"request_id": request_id.strip(), "true_label": true_label},
            )
            show_result(ok, body, status)

    with tab_queue:
        processed = st.checkbox("Processed", value=False)
        limit = st.number_input("Limit", min_value=1, max_value=1000, value=100)
        if st.button("Load Feedback Queue (Admin)"):
            ok, body, status = api_call(
                "GET",
                "/api/v1/feedback",
                params={"processed": str(processed).lower(), "limit": int(limit)},
            )
            show_result(ok, body, status)
            if ok and isinstance(body, list) and body:
                st.dataframe(pd.DataFrame(body), use_container_width=True)

    with tab_trigger:
        if st.button("Trigger Retraining (Admin)"):
            ok, body, status = api_call("POST", "/api/v1/feedback/retrain/trigger")
            show_result(ok, body, status)

    with tab_status:
        if st.button("Get Retraining Status (Admin)"):
            ok, body, status = api_call("GET", "/api/v1/feedback/retrain/status")
            show_result(ok, body, status)


def render_models() -> None:
    section_header("Model Management", "Inspect model status, details, metrics, health, and reload models.")

    tab_list, tab_detail, tab_health, tab_reload, tab_metrics = st.tabs(
        ["Models List", "Model Detail", "Health", "Reload", "Metrics"]
    )

    with tab_list:
        if st.button("List Models"):
            ok, body, status = api_call("GET", "/api/v1/models")
            show_result(ok, body, status)
            if ok and isinstance(body, dict) and isinstance(body.get("models"), list):
                st.dataframe(pd.DataFrame(body["models"]), use_container_width=True)

    with tab_detail:
        model_name = st.selectbox("Model", ["news", "sms", "multimodal"])
        if st.button("Get Model Info"):
            ok, body, status = api_call("GET", f"/api/v1/models/{model_name}")
            show_result(ok, body, status)

    with tab_health:
        if st.button("Check Health"):
            ok, body, status = api_call("GET", "/api/v1/models/health", with_auth=False)
            show_result(ok, body, status)

    with tab_reload:
        if st.button("Reload Models (Admin)"):
            ok, body, status = api_call("POST", "/api/v1/models/reload")
            show_result(ok, body, status)

    with tab_metrics:
        if st.button("Get API Metrics (Admin)"):
            ok, body, status = api_call("GET", "/api/v1/models/metrics")
            show_result(ok, body, status)


def render_utilities() -> None:
    section_header("Utilities", "Version endpoint and quick links to API docs and OpenAPI schema.")

    col1, col2, col3 = st.columns(3)
    if col1.button("Get Version"):
        ok, body, status = api_call("GET", "/api/v1/version", with_auth=False)
        show_result(ok, body, status)

    docs_url = f"{st.session_state.base_url.rstrip('/')}/docs"
    openapi_url = f"{st.session_state.base_url.rstrip('/')}/openapi.json"
    redoc_url = f"{st.session_state.base_url.rstrip('/')}/redoc"

    col2.link_button("Open Swagger Docs", docs_url, use_container_width=True)
    col3.link_button("Open OpenAPI JSON", openapi_url, use_container_width=True)
    st.link_button("Open ReDoc", redoc_url, use_container_width=False)


def main() -> None:
    inject_styles()
    init_state()

    current_page = render_sidebar()

    # User-facing pages
    if current_page == "🏠 Home":
        render_home()
    elif current_page == "🔍 Detect Content":
        render_detection()
    elif current_page == "📊 My Dashboard":
        render_my_dashboard()
    elif current_page == "📜 History":
        render_user_history()
    elif current_page == "💬 Give Feedback":
        render_user_feedback()
    
    # Developer/admin pages
    elif current_page == "🔐 Auth (Dev)":
        render_auth()
    elif current_page == "⚙️ Jobs & APIs (Dev)":
        st.title("Developer Tools")
        tab1, tab2, tab3 = st.tabs(["Async Jobs", "Model Management", "Utilities"])
        with tab1:
            render_jobs()
        with tab2:
            render_models()
        with tab3:
            render_utilities()
    elif current_page == "🧠 Explainability (Dev)":
        render_explainability()
    elif current_page == "🎛️ Admin Panel (Dev)":
        st.title("Admin Panel")
        tab1, tab2 = st.tabs(["History & Analytics", "Feedback & Retraining"])
        with tab1:
            render_history()
        with tab2:
            render_feedback()
    
    # Legacy fallback for old navigation
    elif current_page == "Dashboard":
        render_dashboard()
    elif current_page == "Authentication":
        render_auth()
    elif current_page == "Detection":
        render_detection()
    elif current_page == "Async Jobs":
        render_jobs()
    elif current_page == "Explainability":
        render_explainability()
    elif current_page == "History & Analytics":
        render_history()
    elif current_page == "Feedback & Retraining":
        render_feedback()
    elif current_page == "Model Management":
        render_models()
    elif current_page == "Utilities":
        render_utilities()


if __name__ == "__main__":
    main()