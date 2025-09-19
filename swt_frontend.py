import streamlit as st
from swt_backend import app, segment_info
from langchain_core.messages import HumanMessage
import base64
import time
from PIL import Image
import pandas as pd
import numpy as np
from datetime import date, timedelta
import base64
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import base64
import uuid
import io
from v_backend import plot_engagement_histogram,engagement_histogram_insights
import pickle
import streamlit_authenticator as stauth

# ------------------ PAGE CONFIG ------------------

st.set_page_config(page_title="RETAIL KPIs | AI Assistant", page_icon="🏦", layout="wide")

# ----------------------------------------------------------------------------------------------- NWP ----------------------------------------------------------------
# # --- Stronger CSS to style the login box ---
# st.markdown(
#     """
#     <style>
#     /* Common Streamlit form containers */
#     div[data-testid="stForm"],
#     form[role="form"],
#     div.stForm {
#         background-color: #e6f2ff !important;  /* change to any color */
#         padding: 20px !important;
#         border-radius: 12px !important;
#         box-shadow: 0 6px 18px rgba(0,0,0,0.08) !important;
#     }

#     /* Target the first login form if multiple forms exist */
#     main div[data-testid="stForm"]:first-of-type,
#     main form[role="form"]:first-of-type {
#         background-color: #e6f2ff !important;
#     }

#     /* Inputs inside the form */
#     div[data-testid="stForm"] input,
#     form[role="form"] input {
#         background-color: #ffffff !important;
#         border-radius: 8px !important;
#         padding: 8px !important;
#     }

#     /* Buttons inside the form */
#     div[data-testid="stForm"] button,
#     form[role="form"] button {
#         background-color: #2E86C1 !important;
#         color: #fff !important;
#         border-radius: 8px !important;
#         padding: 8px 14px !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # ------------------ USER AUTHENTICATION ------------
# names = ["Robert Parker","Harry Miller"]
# usernames = ['rparker','hmiller']

# file_path = Path(__file__).parent / "hashed_pw.pkl"
# with file_path.open('rb') as file:
#     hashed_passwords = pickle.load(file)

# # --- Build credentials dictionary ---
# credentials = {
#     "usernames": {
#         usernames[i]: {
#             "name": names[i],
#             "password": hashed_passwords[i]
#         }
#         for i in range(len(usernames))
#     }
# }

# authenticator = stauth.Authenticate(credentials,"swt_segment",'abcdef',cookie_expiry_days=30)

# # --- Login widget ---
# name, authentication_status, username = authenticator.login(
#     fields={'Form name': 'Login'},
#     location='main'
# )

# if authentication_status == False:
#     st.error('Username/Password is incorrect')
# if authentication_status == None:
#     st.warning("Please enter your username and password")
# if authentication_status:
#------------------------------------------------------------------------------------------------------- NWP ---------------------------------------------------------------
    # --------------------------------------------

DARK_BLUE_1 = "#0B1F4B"
DARK_BLUE_2 = "#1E3A8A"
TEXT_DARK   = "#111827"

SNOW_BLUE_1 = "#00AEEF"
SNOW_BLUE_2 = "#1F5AA6"
INK_900     = "#0F172A"
INK_700     = "#334155"
BG_SOFT     = "#F6F9FC"

# ------------------ Config ------------------

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# ------------------ GLOBAL STYLES ------------------
st.markdown(f"""
<style>
.block-container {{ padding-top: 1rem; }}
#MainMenu, footer {{ visibility: hidden; }}
/* Header: dark blue gradient + white text */
.hdr {{
    background: linear-gradient(135deg, {SNOW_BLUE_1} 0%, {SNOW_BLUE_2} 100%);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 16px;
    box-shadow: 0 10px 28px rgba(0,0,0,.18);
    color: #ffffff !important;
}}
.hdr h1 {{
    margin: 0;
    font-size: 22px;
    letter-spacing: .2px;
    color: #ffffff !important;
    text-shadow: 0 1px 0 rgba(0,0,0,.15);
}}
.filter-label {{ font-weight: 800; color: #6b7280; margin-bottom: 6px }}
.metric-card {{ border:1px solid #e6e8ef; border-radius:12px; padding:16px 18px; background:#fff }}
.metric-title {{ font-size:12px; color:#6b7280; font-weight:800; text-transform:uppercase; letter-spacing:.04em }}
.metric-value {{ font-size:28px; font-weight:900; color:{TEXT_DARK}; margin-top:4px }}
.chat-wrap {{ border:1px solid #e6e8ef; border-radius:12px; background:#fff }}
.chat-head {{ padding:10px 14px; border-bottom:1px solid #eef0f5; font-weight:900; color:#111827 }}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='hdr'><h1> RETAIL Marketing Insight Assistant </h1></div>", unsafe_allow_html=True)

# ===== STATE =====
def _init_state():
    ss = st.session_state
    ss.setdefault("logged_in", False)
    ss.setdefault("user", None)
    ss.setdefault("chats", [])          # [{id,title,messages:[{role,content,ts,meta}]}]
    ss.setdefault("current_chat", None)
    ss.setdefault("compact_bubbles", True)
    ss.setdefault("show_ts", False)
    ss.setdefault("show_dict", False)   # Data Dictionary view toggle
    ss.setdefault("font_scale", 1.08)   # Dynamic font scale (0.90–1.30)
_init_state()

# -------------------------------------------------------------------------------------------------------

# # ------------------ DATA ------------------
# DATA_PATH = Path("bfsi_gold_sample_1000.csv")  # keep CSV next to app.py

# @st.cache_data(show_spinner=False, ttl=600)
# def load_data(p: Path) -> pd.DataFrame:
#     if not p.exists():
#         st.error(f"CSV not found: {p}. Place bfsi_gold_sample_1000.csv next to app.py.")
#         st.stop()
#     df = pd.read_csv(p)
#     # normalize dtypes / names
#     for c in ["RESPONSE_STATUS", "REGION", "PRODUCT", "CAMPAIGN_NAME", "CHANNEL"]:
#         if c in df.columns:
#             df[c] = df[c].astype(str)
#     for c in ["REVENUE", "SPEND"]:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
#     return df

# df = load_data(DATA_PATH)

# # ------------------ FILTERS ------------------
# f1, f2, f3 = st.columns([1.2, 1.6, 1.2])
# #, vertical_alignment="center"
# with f1:
#     st.markdown("<div class='filter-label'>🧩 Product</div>", unsafe_allow_html=True)
#     prod_opts = sorted(df["PRODUCT"].dropna().unique().tolist()) if "PRODUCT" in df.columns else []
#     sel_products = st.multiselect("", prod_opts, default=prod_opts, key="f_prod")

# with f2:
#     st.markdown("<div class='filter-label'>🎯 Campaign</div>", unsafe_allow_html=True)
#     if "CAMPAIGN_NAME" in df.columns:
#         camp_opts = sorted(df["CAMPAIGN_NAME"].dropna().unique().tolist())
#         sel_campaigns = st.multiselect("", camp_opts, default=camp_opts, key="f_camp")
#     else:
#         sel_campaigns = []
#         st.selectbox("", ["No campaign field found"], index=0, disabled=True)

# with f3:
#     st.markdown("<div class='filter-label'>🌍 Region</div>", unsafe_allow_html=True)
#     region_opts = sorted(df["REGION"].dropna().unique().tolist()) if "REGION" in df.columns else []
#     sel_regions = st.multiselect("", region_opts, default=region_opts, key="f_region")

# mask = pd.Series(True, index=df.index)
# if sel_products:   mask &= df["PRODUCT"].isin(sel_products)
# if sel_regions:    mask &= df["REGION"].isin(sel_regions)
# if sel_campaigns:  mask &= df["CAMPAIGN_NAME"].isin(sel_campaigns)
# dff = df.loc[mask].copy()

# st.markdown("---")  # (no records chip by request)

# # ------------------ HELPERS ------------------
# def pct(n, d): return 0.0 if d == 0 else (n / d) * 100.0
# def fmt_money_myr(v: float) -> str: return f"RM {v:,.0f}"

# def format_roi_from_roas(roas: float) -> str:
#     """Display ROI as +X.X% based on ROAS (revenue/spend)."""
#     if roas is None or np.isnan(roas) or np.isinf(roas):
#         return "—"
#     roi = (roas - 1.0) * 100.0
#     sign = "+" if roi >= 0 else ""
#     return f"{sign}{roi:.1f}% "

# # ------------------ KPIs ------------------

# total = len(dff)
# resp = dff["RESPONSE_STATUS"].str.lower() if "RESPONSE_STATUS" in dff.columns else pd.Series("", index=dff.index)
# engaged   = int((resp == "engaged").sum())
# converted = int((resp == "converted").sum())

# leads_pct = pct(engaged, total)
# conv_pct  = pct(converted, total)
# revenue   = float(dff["REVENUE"].sum()) if "REVENUE" in dff.columns else 0.0
# spend     = float(dff["SPEND"].sum()) if "SPEND" in dff.columns else 0.0
# roas_val  = (revenue / spend) if spend > 0 else None  # for ROI formatter

# c1, c2, c3, c4 = st.columns(4)
# with c1:
#     st.markdown(
#         f"<div class='metric-card'><div class='metric-title'>Leads %</div>"
#         f"<div class='metric-value'>{leads_pct:.2f}%</div></div>", unsafe_allow_html=True)
# with c2:
#     st.markdown(
#         f"<div class='metric-card'><div class='metric-title'>Conversion %</div>"
#         f"<div class='metric-value'>{conv_pct:.2f}%</div></div>", unsafe_allow_html=True)
# with c3:
#     st.markdown(
#         f"<div class='metric-card'><div class='metric-title'>Revenue (RM)</div>"
#         f"<div class='metric-value'>{fmt_money_myr(revenue)}</div></div>", unsafe_allow_html=True)
# with c4:
#     st.markdown(
#         f"<div class='metric-card'><div class='metric-title'>ROI</div>"
#         f"<div class='metric-value'>{format_roi_from_roas(roas_val)}</div></div>", unsafe_allow_html=True)

# st.markdown("")

# ------------------------------------------------------------------------------------------

def _new_chat(title="Untitled"):
    cid = str(uuid.uuid4())[:8]
    st.session_state.chats.insert(0, {"id": cid, "title": title, "messages": []})
    st.session_state.current_chat = cid
    return cid

# ===== CHAT MGMT (delete + active highlight) =====
def _delete_chat(chat_id: str):
    chats = st.session_state.chats
    st.session_state.chats = [c for c in chats if c["id"] != chat_id]
    if st.session_state.current_chat == chat_id:
        if st.session_state.chats:
            st.session_state.current_chat = st.session_state.chats[0]["id"]
        else:
            _new_chat("Untitled")

# ====== CUSTOMER 360 DATA DICTIONARY (business-friendly) ======
C360_DICTIONARY = {
    "Identity": [
        {"Attribute": "CUSTOMER_ID", "Definition": "Unique identifier for the customer across all systems.", "Notes/Example": "e.g., 10023891"},
        {"Attribute": "CUSTOMER_NAME", "Definition": "Full name as captured in KYC/CRM.", "Notes/Example": "e.g., Rehman Aziz"},
        {"Attribute": "SEGMENT_BADGE", "Definition": "High-level customer segment tag used for audience selection.", "Notes/Example": "e.g., Platinum, Mass Affluent"},
    ],
    "Demographics": [
        {"Attribute": "GENDER", "Definition": "Self-declared or inferred gender.", "Notes/Example": "Male / Female / Unknown"},
        {"Attribute": "DOB", "Definition": "Date of birth used for age-based rules and KYC.", "Notes/Example": "YYYY-MM-DD"},
        {"Attribute": "CITY", "Definition": "Primary city of residence.", "Notes/Example": "Kuala Lumpur"},
        {"Attribute": "STATE", "Definition": "State/province of residence.", "Notes/Example": "Selangor"},
        {"Attribute": "REGION", "Definition": "Macro geography grouping for reporting.", "Notes/Example": "Central / North / South"},
        {"Attribute": "INCOME", "Definition": "Declared or inferred monthly income band.", "Notes/Example": "e.g., 6,000–10,000 MYR"},
        {"Attribute": "OCCUPATION", "Definition": "Primary occupation for profiling.", "Notes/Example": "e.g., Engineer"},
    ],
    "Contact & Channel": [
        {"Attribute": "PHONE_NO", "Definition": "Primary contact number for service and OTP.", "Notes/Example": "+60-XXX-XXXX"},
        {"Attribute": "EMAIL_ID", "Definition": "Primary email for communication.", "Notes/Example": "name@example.com"},
        {"Attribute": "PREFERRED_CHANNEL", "Definition": "Customer’s most responsive outreach channel.", "Notes/Example": "Email / SMS / App Push"},
        {"Attribute": "most_recent_device_locale", "Definition": "Locale of recent device used.", "Notes/Example": "en_MY"},
        {"Attribute": "time_zone", "Definition": "Customer’s time zone for scheduling outreach.", "Notes/Example": "Asia/Kuala_Lumpur"},
    ],
    "orders": [
    {"Attribute": "TOTAL_ORDERS", "Definition": "Total number of completed orders by the customer.", "Notes/Example": "Numeric (count)"},
    {"Attribute": "TOTAL_ORDER_VALUE", "Definition": "Cumulative monetary value of all completed orders.", "Notes/Example": "MYR value"},
    {"Attribute": "AVG_ORDER_VALUE", "Definition": "Average value per order.", "Notes/Example": "MYR value"},
    {"Attribute": "LAST_ORDER_DATE", "Definition": "Date of the most recent completed order.", "Notes/Example": "YYYY-MM-DD"},
    {"Attribute": "DAYS_SINCE_LAST_ORDER", "Definition": "Number of days since most recent order.", "Notes/Example": "Numeric (days)"},
    {"Attribute": "CANCELLED_ORDERS", "Definition": "Total number of orders cancelled by the customer.", "Notes/Example": "Numeric (count)"},
    {"Attribute": "RETURNED_ORDERS", "Definition": "Total number of orders returned by the customer.", "Notes/Example": "Numeric (count)"},
    {"Attribute": "MOST_FREQUENT_CATEGORY", "Definition": "Category in which the customer has placed maximum orders.", "Notes/Example": "e.g., Electronics, Apparel"},
    {"Attribute": "PREFERRED_CHANNEL", "Definition": "Channel used most often for placing orders.", "Notes/Example": "e.g., App, Web, Store"}
    ],
    "Engagement & Digital": [
        {"Attribute": "AVG_MONTHLY_LOGINS", "Definition": "Average number of app/web logins per month.", "Notes/Example": "Numeric"},
        {"Attribute": "MOBILE_APP_LOGIN_LAST_30D", "Definition": "Count of app logins in last 30 days.", "Notes/Example": "0–N"},
        {"Attribute": "sessions_in_past_30_days", "Definition": "Total sessions (web+app) in last 30 days.", "Notes/Example": "0–N"},
        {"Attribute": "most_preferred_prod_category", "Definition": "Commonly browsed product category.", "Notes/Example": "e.g., Jeans / Loans"},
    ],
    "Support & Risk": [
        {"Attribute": "SERVICE_TICKET_COUNT_LAST_90D", "Definition": "Support tickets created in last 90 days.", "Notes/Example": "0–N"},
        {"Attribute": "AVG_RESOLUTION_TIME", "Definition": "Average time to close tickets.", "Notes/Example": "Hours"},
        {"Attribute": "SENTIMENT_SCORE", "Definition": "Aggregated sentiment from calls/emails/chats.", "Notes/Example": "-1.0 to +1.0"},
        {"Attribute": "RFM_SEGMENT", "Definition": "Recency-Frequency-Monetary segment label.", "Notes/Example": "Champions, At Risk, etc."},
    ],
    "Campaign & CRM": [
        {"Attribute": "CAMPAIGN_NAME", "Definition": "Latest campaign targeted to the customer.", "Notes/Example": "e.g., PRS Exclusive Reward"},
        {"Attribute": "CAMPAIGN_PRODUCT", "Definition": "Product promoted in latest campaign.", "Notes/Example": "e.g., Personal Loan"},
        {"Attribute": "CAMPAIGN_RESPONSE_STATUS", "Definition": "Last response captured for campaign.", "Notes/Example": "Opened / Clicked / Purchased / No Action"},
        {"Attribute": "discounted_campaign_engagement_rate", "Definition": "Share of discount campaigns engaged with.", "Notes/Example": "0–100%"},
        {"Attribute": "rfm_ml_cluster", "Definition": "ML-driven cluster combining RFM and behavior.", "Notes/Example": "e.g., Promising"},
    ],
}

def render_data_dictionary():
    """Main panel view for the Customer 360 data dictionary."""
    import pandas as pd

    st.markdown("### 📚 Customer 360 Attributes")
    colA, colB = st.columns([0.7, 0.3])
    with colA:
        q = st.text_input("Search attributes or definitions…", placeholder="Try: login, sentiment, credit score", key="dict_search")
    with colB:
        if st.button("Close", use_container_width=True):
            st.session_state.show_dict = False
            st.rerun()

    categories = list(C360_DICTIONARY.keys())
    tabs = st.tabs(categories)

    ql = (q or "").lower().strip()

    for tab, cat in zip(tabs, categories):
        with tab:
            rows = C360_DICTIONARY[cat]
            if ql:
                rows = [
                    r for r in rows
                    if (ql in r["Attribute"].lower()) or (ql in r["Definition"].lower()) or (ql in str(r.get("Notes/Example","")).lower())
                ]
            if not rows:
                st.info("No matches in this category.")
            else:
                df = pd.DataFrame(rows, columns=["Attribute", "Definition", "Notes/Example"])
                st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Field conventions / tips", expanded=False):
        st.markdown(
            "- **Flags** are 0/1 booleans.  \n"
            "- **Counts** are integers over the lookback window in the field name.  \n"
            "- **Scores** typically map to 0–1, -1–1, or standardized scales (300–900 for credit).  \n"
            "- **Segments/Badges** are categorical labels used for targeting."
        )


# ===== MAIN PANEL ROUTING =====
if st.session_state.get("show_dict"):
    render_data_dictionary()

# --- Custom CSS for sidebar ---
st.markdown(
    """
    <style>
        /* Sidebar background color */
        [data-testid="stSidebar"] {
            background-color: #f5f5f5; /* Light gray */
        }

        /* Sidebar text color */
        [data-testid="stSidebar"] * {
            color: #333333; /* Dark text */
        }

        /* Optional: add border or shadow */
        [data-testid="stSidebar"] {
            border-right: 2px solid #ddd;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
    )

# --- Custom CSS for sidebar buttons ---
st.markdown(
    """
    <style>
        /* Style all sidebar buttons */
        [data-testid="stSidebar"] button {
            background-color: #f0f0f0; /* Green background */
            color: #333;              /* dark text */
            border: 1px solid #ccc;
            border-radius: 6px;        /* Rounded corners */
            font-weight: 500;
        }
        
        /* Ensure inner text (span) respects color */
        [data-testid="stSidebar"] button * {
            color: #333 !important;  
        }
        
        /* Hover effect */
        [data-testid="stSidebar"] button:hover {
            background: linear-gradient(90deg, #ff9800, #f57c00); /* Deeper orange on hover */
            color: #fff;   /* White text */     
        }
        
        /* Force white text inside on hover */
        [data-testid="stSidebar"] button:hover * {
            color: #fff !important;
        }

        /* Optional: spacing between buttons */
        [data-testid="stSidebar"] button {
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
    )

# --------------------- Side Bar Section -------------------------
st.sidebar.title(f"Welcome, Robert Parker!")
authenticator.logout("Logout",'sidebar')
# Sidebar — conversations + dictionary button
with st.sidebar:
    # st.markdown("#### CHAT HISTORY")
    # if st.button("➕ New chat", use_container_width=True):
    #     st.session_state.show_dict = False
    #     _new_chat("Untitled"); st.rerun()

    # for chat in st.session_state.chats:
    #     is_active = (chat['id'] == st.session_state.current_chat) and not st.session_state.get("show_dict")
    #     row = st.container()
    #     with row:
    #         c_open, c_del = st.columns([0.88, 0.12], vertical_alignment="center")
    #         btn_container = c_open.container()
    #         if is_active:
    #             btn_container.markdown("<div class='active-chat'>", unsafe_allow_html=True)
    #         open_clicked = btn_container.button(
    #             f"💬 {chat['title'] or 'Conversation'}",
    #             key=f"open_{chat['id']}",
    #             use_container_width=True,
    #             type=("primary" if is_active else "secondary")
    #         )
    #         if is_active:
    #             btn_container.markdown("</div>", unsafe_allow_html=True)
    #         if open_clicked:
    #             st.session_state.show_dict = False
    #             st.session_state.current_chat = chat["id"]; st.rerun()

    #         del_clicked = c_del.button("🗑️", key=f"del_{chat['id']}", help="Delete chat")
    #         if del_clicked:
    #             _delete_chat(chat["id"]); st.rerun()

    #     st.markdown("<div class='row-sep'></div>", unsafe_allow_html=True)
    
    # ---------- Reference ---------
    st.markdown("<h2 style='font-size:20px;'>📖 REFERENCES</h2>", unsafe_allow_html=True)
    if st.button("📖 Customer 360 Attributes", use_container_width=True):
        st.session_state.show_dict = True
        st.rerun()
    # ------------------------------
    
    # ----------- 
    st.markdown("<h2 style='font-size:20px;'>📊 Retail Insights</h2>", unsafe_allow_html=True)
    st.write("💰 Total Sales (Today): S$18,500")
    st.write("👥 Active Shoppers (Today): 245")
    
    # ----- Predefined Segments (buttons only in sidebar) -------
    st.markdown("<h2 style='font-size:20px;'>📂 Pre-defined Segments</h2>", unsafe_allow_html=True)
    for segment_name, segment_data in segment_info.items():
        button_label = f"{segment_name} : {segment_data['customer_count']} Customers"
        if st.button(button_label, key=segment_name):
            st.session_state["selected_segment"] = segment_name
            st.session_state["segment_description"] = segment_data["description"]
    
    # ------ About -------------
    st.markdown("<h2 style='font-size:20px;'>ℹ️ About</h2>", unsafe_allow_html=True)
    st.markdown(
    """
    <p style="color:#333333; font-size:14px;">
        Built for retail marketing teams to analyze customer segmentation using AI.
    </p>
    <p style="color:#333333; font-size:14px;">
        App v1.0 | Powered by Snowflake + LangChain + Streamlit
    </p>
    """,
    unsafe_allow_html=True)
    # st.markdown("<h2 style='font-size:20px;'>📂 Pre-defined Segments</h2>", unsafe_allow_html=True)
    # for segment_name in segment_info.keys():
    #     if st.button(segment_name, key=segment_name):
    #         st.session_state["selected_segment"] = segment_name
            
# ---------------- Segment Info Diaplyed --------------

# -------------------------
# Main Content (right side)
# -------------------------
if "selected_segment" in st.session_state:
    seg = st.session_state["selected_segment"]
    
    # Top row with title + clear button
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.markdown(f"## ✨ {seg}")
    with col2:
        if st.button("❌ Clear", use_container_width=True):
            del st.session_state["selected_segment"]
            st.rerun()

    # Segment details
    st.markdown(segment_info[seg]["description"])
    st.markdown("**SQL Query to retrieve this segment:**")
    st.code(segment_info[seg]["sql"], language="sql")
    

# --------------- Chat bot section -------------------- 

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if "message_history" not in st.session_state:
        st.session_state.message_history = [{"role":"assistant","content":"👋 Welcome! Ask about customer segments, campaign targeting, engagement metrics, segment size, or final activation rules."}]

with st.expander("🤖 AI Financial Assistant", expanded=True):
    #st.markdown("<div class='section-title'>🤖 AI Financial Assistant</div>", unsafe_allow_html=True)
    
    if 'message_history' not in st.session_state:
        st.session_state['message_history'] = []

    if 'Predefine_history' not in st.session_state:
        st.session_state['Predefine_history'] = []
        
    # --- Scrollable message box ---
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    
    # loading the conversation history
    for message in st.session_state['message_history']:
        with st.chat_message(message['role'], avatar="🧑" if message['role']=="user" else "🤖"):
            st.markdown(message['content'])

    st.markdown("</div>", unsafe_allow_html=True)

        
    # ----------------- V1 ------------------------
    # # loading the conversation history
    # for message in st.session_state['message_history']:
    #     with st.chat_message(message['role'], avatar="🧑" if message['role']=="user" else "🤖"):
    #         if message.get("is_image", False):
    #             st.image(message['content'], caption="Campaign Engagement Histogram", use_column_width=True)
    #         else:
    #             st.markdown(message['content'])

    # -------------------- V1 ------------------------

    # loading the conversation history
    # for message in st.session_state['message_history']:
    #     with st.chat_message(message['role'], avatar="🧑" if message['role']=="user" else "🤖"):
    #         st.markdown(message['content'])

    st.markdown("</div>", unsafe_allow_html=True)

# st.image('campaign.png', caption="This is an example image", use_column_width=True)
    
user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user',avatar="🧑"):
        st.markdown(f"<div class='user-msg'>{user_input}</div>", unsafe_allow_html=True)
        # st.markdown(f"<div style='font-size:28px;'>{user_input}</div>", unsafe_allow_html=True)
        #st.text(user_input)

    # response = app.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    # ai_message = response['messages'][-1].content
    
    # Check if the user asks for a histogram
    if "histogram" in user_input.lower() or "plot" in user_input.lower():
        fig = plot_engagement_histogram()

        # Append only a placeholder text for AI in history, don't send Figure object
        st.session_state['message_history'].append({
            'role': 'assistant',
            'content': "Here is your histogram:",
            'is_image': True
        })

        # Render histogram in the chat
        col1, col2 = st.columns([1,1])  # left small, right large
        with col1:
            with st.chat_message('assistant', avatar="🤖"):
                with st.spinner("Thinking..."):
                    time.sleep(3.8)
                    st.markdown("Here is your histogram:")
                    st.pyplot(fig)
        
        st.markdown(engagement_histogram_insights())

    else:
        # Normal AI text response
        response = app.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
        ai_message = response['messages'][-1].content

        st.session_state['message_history'].append({
            'role': 'assistant',
            'content': ai_message,
            'is_image': False
        })

        with st.chat_message('assistant', avatar="🤖"):
            with st.spinner("Thinking..."):
                time.sleep(1.8)
                st.markdown(f"<div class='assistant-msg'>{ai_message}</div>", unsafe_allow_html=True)

    
    # Append AI message
    # st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    
    # with st.chat_message('assistant',avatar="🤖"):
    #     with st.spinner("Thinking..."):
    #         time.sleep(1.8)
    #         st.markdown(f"<div class='assistant-msg'>{ai_message}</div>", unsafe_allow_html=True)

st.markdown("""
        <style>
        .user-msg {
            background-color: #C8FACC;
            padding: 12px;
            border-radius: 15px;
            margin: 8px 0px;
            font-size: 20px;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)        

            

