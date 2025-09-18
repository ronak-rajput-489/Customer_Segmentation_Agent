from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from openai import OpenAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict
from sqlalchemy import create_engine, text
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import quote_plus
from langchain.embeddings import OpenAIEmbeddings  # Or HuggingFaceEmbeddings
# from dbutils import schema, table_info
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
import io
import base64
import os
load_dotenv()

###########################################################################################################

snowflake_user = os.getenv("SNOWFLAKE_USER")
snowflake_pass = os.getenv("SNOWFLAKE_PASSWORD")
snowflake_acct = os.getenv("SNOWFLAKE_ACCOUNT")  # e.g. ab12345.ap-southeast-1
snowflake_db   = os.getenv("SNOWFLAKE_DATABASE")
snowflake_schema = os.getenv("SNOWFLAKE_SCHEMA")
snowflake_wh = os.getenv("SNOWFLAKE_WAREHOUSE")
snowflake_role = os.getenv("SNOWFLAKE_ROLE")

# 🧠 1. Create a LangGraph-compatible state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Snowflake Engine:
engine = create_engine(
    f"snowflake://{snowflake_user}:{snowflake_pass}@{snowflake_acct}/"
    f"{snowflake_db}/{snowflake_schema}?warehouse={snowflake_wh}&role={snowflake_role}"
)
# 🧠 State
class AgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]

##########################################################################################################

@tool
def run_raw_sql(query: str) -> str:
    """Run any Snowflake SQL command like CREATE, INSERT, UPDATE, DELETE."""
    try:
        with engine.begin() as conn:
            conn.execute(text(query))
        return "✅ Snowflake SQL command executed successfully."
    except Exception as e:
        return f"❌ Error executing SQL: {e}"

#############################################################################################################

@tool
def run_select_sql(query: str) -> str:
    """Run SELECT queries against the snowflake and return results."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            if not rows:
                return "✅ Query executed, but returned no data."
            return "\n".join(str(row) for row in rows)
    except Exception as e:
        return f"❌ Error executing SQL: {e}"

###############################################################################################################

## RAG Agent
## --> Load PDF
# loader = PyPDFLoader("pd_description.pdf")
# documents = loader.load()

# ## --> Chunking
# splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
# chunks = splitter.split_documents(documents)

# ## --> Embeddings
# embeddings = OpenAIEmbeddings()  # Or HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ## --> Create Vector Store
# vectorstore = FAISS.from_documents(chunks, embeddings)

## --> Add RAG Tool

# retriever = vectorstore.as_retriever()
# rag_chain = RetrievalQA.from_chain_type(
#     llm=ChatOpenAI(model="gpt-4.1-nano"),
#     retriever=retriever
# )

# @tool
# def rag_lookup(query: str) -> str:
#     """Answer questions from knowledge documents using RAG."""
#     return rag_chain.run(query)

##################################################################################################################

# Example function to categorize
def categorize_risk(score):
    if score <= 10:
        return 'Low'
    elif 10 < score <= 25:
        return 'Moderate'
    else:
        return 'High'
    
@tool
def segment_customers(table_name: str) -> str:
    """
    Segments customers into Low, Moderate, High based on their risk_appetite score.
    Args:
        table_name: The name of the database table containing the customer data.
    Returns:
        A message confirming segmentation and showing sample output.
    """
    # 🔹 Load table from schema (example: using SQLAlchemy)
    #from sqlalchemy import create_engine
    # engine = create_engine(
    # f"snowflake://{snowflake_user}:{snowflake_pass}@{snowflake_acct}/"
    # f"{snowflake_db}/{snowflake_schema}?warehouse={snowflake_wh}&role={snowflake_role}"
    # )
    query = f"SELECT * FROM {table_name}"
    df_final = pd.read_sql(query, engine)

    # Apply categorization
    df_final['risk_appetite_class'] = df_final['risk_appetite'].apply(categorize_risk)

    # Optionally save back to DB
    df_final.to_sql(table_name + "_segmented", engine, if_exists="replace", index=False)

    return f"Segmentation done ✅. Saved as {table_name}_segmented. Sample:\n{df_final[['customer_id','risk_appetite','risk_appetite_class']].head(5)}"

################################################################################################################

@tool
def recommend_products(table_name: str) -> str:
    """
    Recommends investment products for customers based on their risk_appetite_class.
    Args:
        table_name: The name of the database table containing the segmented customer data.
    Returns:
        A message confirming recommendations and showing sample output.
    """

    # 🔹 Load table from schema
    query = f"SELECT * FROM {table_name}"
    df_final = pd.read_sql(query, engine)

    # Check if segmentation already exists
    if 'risk_appetite_class' not in df_final.columns:
        raise ValueError("Table must already contain 'risk_appetite_class'. Run segment_customers first.")

    # Mapping dictionary (you can extend this easily)
    product_mapping = {
        "Low": "A – Investlink",
        "Moderate": "A - Life Wealth Premier",
        "High": "A - Life Infinite"
    }

    # Apply mapping
    df_final['recommended_product'] = df_final['risk_appetite_class'].map(product_mapping)

    # Save back to DB (new table with recommendations)
    new_table = table_name + "_with_recommendations"
    df_final.to_sql(new_table, engine, if_exists="replace", index=False)

    return (
        f"✅ Product recommendations added based on risk appetite. "
        f"Saved as {new_table}. Sample:\n"
        f"{df_final[['customer_id','risk_appetite_class','recommended_product']].head(5)}"
    )

## ------------------- Image Rendering Tool -----------------

@tool
def show_histogram() -> str:
    """
    Return the path to the pre-generated campaign engagement histogram image.
    Only called when user explicitly requests a histogram.
    """
    return "campaign.png"

## ---------------------Image Rendering Tool ----------------------

@tool
def show_histogram() -> plt.Figure:
    """
    Dynamically generate a histogram of simulated campaign engagement rates.
    Returns a matplotlib figure object.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10,6))

    code_to_run = """
np.random.seed(42)
right_skewed = np.random.beta(2, 5, 1500) * 100
ax.hist(right_skewed, bins=20, edgecolor='black', alpha=0.7, color='lightblue')
ax.set_title("Simulated Right-Skewed Histogram of Campaign Engagement Rate", fontsize=14)
ax.set_xlabel("Engagement Rate (%)", fontsize=12)
ax.set_ylabel("Number of Customers", fontsize=12)
ax.grid(alpha=0.3)
"""
    exec(code_to_run, {"np": np, "plt": plt, "ax": ax})
    return fig

#################################################################################################################

tools = [run_raw_sql, run_select_sql, segment_customers, recommend_products]

#################################################################################################################

segment_info = {
    "Discount Seekers": {
        "description": """
**Discount Seekers** are customers who are highly motivated by promotions and discounts. 
They wait for sales, stockpile low-priced goods, and are less brand loyal. 
They often respond actively to coupon codes, cashback, and bundle deals.
""",
    "customer_count": "11,700",
    "Rules": """
            "discount_used = TRUE in majority of purchases",
            "avg_discount_percent >= 20",
            "cart_abandonment_rate > overall_avg (wait for discount before checkout)",
            "gender = 'Female' OR 'Male' (demographic split possible)",
            "preferred_channel = 'online' (higher promo exposure)""",
            
        "sql": """
SELECT customer_id
FROM transactions
WHERE discount_used = TRUE
GROUP BY customer_id
HAVING AVG(discount_percent) >= 20
   AND SUM(cart_abandoned) > (SELECT AVG(cart_abandoned) FROM transactions)
   AND preferred_channel = 'online';
"""
    },

    "Churn Risk Customers": {
        "description": """
**Churn Risk Customers** show declining engagement and reduced purchase activity. 
They may have shifted to competitors or lost interest. 
They often need reactivation campaigns like win-back emails, loyalty points, or personalized offers.
""",
    "customer_count": "4,753",
        "sql": """
SELECT customer_id
FROM customer_summary
WHERE last_purchase_date <= DATEADD(DAY, -90, CURRENT_DATE)
  AND purchase_frequency < (SELECT AVG(purchase_frequency) FROM customer_summary)
  AND total_spent_last_12m < (SELECT MEDIAN(total_spent_last_12m) FROM customer_summary)
  AND email_open_rate < 0.2
  AND customer_tenure > 12;
"""
    },

    "Seasonal Buyers": {
        "description": """
**Seasonal Buyers** purchase mainly during festive or seasonal periods (e.g., holidays, cultural events). 
They show spikes in purchase frequency during peak months and remain inactive otherwise. 
They can be nurtured with timely seasonal reminders and early previews.
""",
    "customer_count": "16,200",
        "sql": """
SELECT customer_id
FROM transactions
WHERE MONTH(purchase_date) IN (11, 12) -- Nov, Dec
   OR purchase_frequency >= 3 * (
        SELECT AVG(purchase_frequency) 
        FROM transactions t2
        WHERE t2.customer_id = transactions.customer_id
          AND MONTH(t2.purchase_date) NOT IN (11, 12)
   )
  AND product_category IN ('gifts', 'festive-pack');
"""
    }
}



#################################################################################################################
    
# Bind tools
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

selected_columns_info = """
Table: LOOMIO_C360
-   PRED_ATTR_RFM_SEGMENT: The customer’s segment label derived from RFM (Recency, Frequency, Monetary) analysis.
-   DEM_ATTR_GENDER: Customer’s gender attribute (e.g., Male, Female, Other). Used to personalize campaigns and segment customers by demographic.
-   ORDR_ATTR_PREFERRED_PROD_CATEGORY: The product category that the customer most frequently purchases or shows preference for (e.g., Jeans, Dresses, Shoes). Derived from transactional order history.
-   CAMP_ENG_ATTR_EMAIL_CAMPAIGN_ENGAGEMENT_RATE: Percentage of marketing email campaigns (with or without discounts) that the customer has engaged with (opened, clicked, redeemed). A key measure of marketing responsiveness.
-   ONLINE_BEHVR_ATTR_TOTAL_NO_OF_SESSIONS_LAST_30D: Total number of times the customer visited the retailer’s website or app in the last 30 days. Indicates digital engagement and activeness.

-   AVG_APP_SESSION_DURATION: The average time (usually in minutes or seconds) a user spends per session in the app.
-   TOTAL_APP_ENGAGEMENT_TIME: The total cumulative time users spend in the app over a given period.
-   PUSH_NOTIFICATION_OPENS: The total number of times users opened or interacted with push notifications sent by the app.
-   PUSH_NOTIFICATION_CTR: Click-through rate for push notifications, calculated as (Number of Clicks on Notification / Number of Notifications Sent) × 100.
-   NO_OF_EMAILS_OPENED: The total number of emails opened by users from marketing or transactional campaigns.
-   NO_OF_EMAIL_CLICKS: The total number of times users clicked on links within emails.
-   PAID_MEDIA_IMPRESSIONS: The total number of times paid ads (social, search, or display) were shown to users.
-   PAID_MEDIA_ENGAGEMENTS: The total number of user interactions (clicks, likes, shares, comments) with paid media ads.
-   
"""

# First model node — decides on tool usage
def model_node(state: AgentState) -> AgentState:
    sys_prompt = SystemMessage(
        content=(
            
            f"""
            You are an AI assistant helping the marketing team of a Retail company. 
            Your task is to design customer segments for campaigns (using Snowflake data).

            Use the following schema as your reference:
            {selected_columns_info}

            Follow this conversational flow:
            1. If the user asks about creating a campaign segment for "overstock female jeans":
                - Focus on filters → Cluster = Promising, Gender = Female, Most Preferred Product Category = Jeans (from transactional data).
                - Always explain *why* these filters are relevant.
                - Respond with these filters as a starting segment.

            2. If the user asks: "How many customers with these filters?" 
                - Run count query and return total size.
                - Provide a short interpretation
            
            3. If the user asks to "reduce the segment size" or "use engagement metrics":
            
                - Highlight available KPIs (e.g., CAMP_ENG_ATTR_EMAIL_CAMPAIGN_ENGAGEMENT_RATE, AVG_APP_SESSION_DURATION, TOTAL_APP_ENGAGEMENT_TIME, PUSH_NOTIFICATION_OPENS, PUSH_NOTIFICATION_CTR, NO_OF_EMAILS_OPENED, PAID_MEDIA_ENGAGEMENTS ).
                - Recommend CAMP_ENG_ATTR_EMAIL_CAMPAIGN_ENGAGEMENT_RATE as the most important metric compared to others, since it directly measures how actively customers interact with marketing campaigns (open rates, click-throughs, and participation). A higher engagement rate indicates stronger responsiveness to marketing efforts, making it a reliable predictor of conversion.
                - Provide this explanation as supporting proof when suggesting the metric, while still allowing the user to select an alternative KPI if desired.
                
            4. If user applies filter (e.g., 40%):
                - Return updated segment size.
                - Explain what this filter means in business terms, e.g., “This cutoff excludes disengaged customers and keeps only those with medium-to-high interaction levels.”

            5. If user asks "how many visited web/app >5 in L30d":
                - Use SESSIONS_IN_PAST_30_DAYS > 5 as filter and return result.
                - Briefly explain why this is useful:

            6. If user asks for final "segment rules to apply in Braze":
                - Always summarize filters clearly like this:

                    1. rfm_ml_cluster = 'Promising'
                    2. gender = 'Female'
                    3. most_preferred_prod_category = 'Jeans'
                    4. discounted_campaign_engagement_rate >= 40% 
                    5. sessions_in_past_30_days > 5
                - Add a one-line conclusion: “This final rule set ensures your campaign targets medium-to-highly engaged female customers who prefer jeans and are active in the last 30 days.”


            Always make your answers **clear, actionable, and tied back to data filters**,  while giving the marketing team enough context to understand *why* these filters are meaningful.
            Your explaination always in business term friendly.
        """
        )
    )
    
    # - Highlight available KPIs (e.g., CAMP_ENG_ATTR_EMAIL_CAMPAIGN_ENGAGEMENT_RATE among others).
    
    # 4. If the user requests a histogram or plot of any KPI/metric, 
    #             - you must call the `show_histogram` tool.
    #             - Do not try to describe the histogram in text — only return the tool call so the frontend can render the image.
    #             - Also provide a clear summary of Image 
    
    
    # 4. If the user selects "discount participation KPI":
    #             - Plot histogram of CAMP_ENG_ATTR_EMAIL_CAMPAIGN_ENGAGEMENT_RATE.
    #             - Add key observations (e.g., most customers cluster between 20–40%, tail at 80%+).

    # - First, explain the rationale for the segment by highlighting the filters being applied (Cluster = Promising, Gender = Female, Most Preferred Product Category = Jeans from transactional data) and why these filters are relevant for targeting.
    #            - After providing this explanation, run the count query and return the total size of the segment that meets these criteria.

    # - Focus on filters → Cluster = Promising, Gender = Female, Most Preferred Product Category = Jeans (from transactional data).
    #             - Respond with these filters as a starting segment.
    #             - Run count query and return total size.
    # - Highlight available KPIs (e.g., CAMP_ENG_ATTR_EMAIL_CAMPAIGN_ENGAGEMENT_RATE among others).
    #            - Allow user to select one.
    # 2. If the user asks: "How many customers with these filters?" → Run count query and return total size.
    # - prefer 'CUSTOMER_segmented' table as a knowledge base.
    # - Suggest products based on the customer’s risk appetite classification (High, Low, High).  
    
    # - Then ask the human user:  
    #                 "Do you want me to recommend specific products for these customers?"  
    #             - If the user say "I would like to recommend products." → **only then call the 'recommend_products' tool** on the 'CUSTOMER_segmented' table.

    # Let the LLM decide next step
    response = model.invoke([sys_prompt] + state["messages"])
    return {"messages": [response]}

###############################################################################################

# Tool execution node — corrected
def tool_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    tool_messages = []

    for call in tool_calls:
        tool_name = call["name"]
        args = call["args"]

        # Find the matching tool
        for t in tools:
            if t.name == tool_name:
                # Run the tool
                result = t.run(args)
                
                # # If result is a matplotlib figure, mark it as an image
                # if isinstance(result, plt.Figure):
                #     tool_messages.append(
                #         ToolMessage(
                #             content=[{"type": "image", "figure": result}],
                #             tool_call_id=call["id"]
                #         )
                #     )
                # else:
                #     # Otherwise, wrap as text
                #     tool_messages.append(
                #         ToolMessage(
                #             content=[{"type": "text", "text": str(result)}],
                #             tool_call_id=call["id"]
                #         )
                #     )
                
                # --------------------- V2 ---------------------------
                
                # # Convert result to string if it's not already
                # if isinstance(result, (dict, list)):
                #     import json
                #     result = json.dumps(result, indent=2)
                    
                # # In tool_node
                # if tool_name == "show_histogram":
                #     # Return raw image path
                #     tool_messages.append(
                #         ToolMessage(
                #             content=[{"type": "text", "text": result}],  # result is just "campaign.png"
                #             tool_call_id=call["id"]
                #         )
                #     )
                # else:
                #     # Keep wrapping for other tools
                #     tool_messages.append(
                #         ToolMessage(
                #             content=[{"type": "text", "text": f"Tool {tool_name} executed. Result:\n{result}"}],
                #             tool_call_id=call["id"]
                #         )
                #     )
                
                #  --------------------- V2 -----------------------
                
                # ------------ Original ------------
                # Wrap result in the expected content structure
                # tool_messages.append(
                #     ToolMessage(
                #         content=[{"type": "text", "text": f"Tool {tool_name} executed. Result:\n{result}"}],
                #         tool_call_id=call["id"]
                #     )
                # )
                # --------------- Original -------------
                
                tool_messages.append(
                    ToolMessage(
                        content=[{"type": "text", "text": result}],  # raw matplotlib code
                        tool_call_id=call["id"]
                    )
                )

    # Return new state with messages
    return {"messages": tool_messages}

##############################################################################################

# Conditional — do we need another round?
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    # Continue only if the last message actually contains a tool call
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "continue"
    return "end"

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", model_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "agent")
app = graph.compile()

############################################################################################### 