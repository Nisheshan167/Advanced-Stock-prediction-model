from openai import OpenAI

# ===============================
# OpenAI init + reporter
# ===============================
def _init_openai_client():
    key = None
    try:
        if "OPENAI_API_KEY" in st.secrets:   # Streamlit Cloud secrets
            key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    if not key:
        key = os.getenv("OPENAI_API_KEY")    # Fallback: env variable

    if not key:
        st.warning("⚠️ OPENAI_API_KEY not found in st.secrets or environment. GenAI analysis will be skipped.")
        return None
    return OpenAI(api_key=key)

# Create the client once, at startup
client = _init_openai_client()

def generate_report(forecast_summary: str, indicators: str, recommendation: str) -> str:
    """LLM commentary on model outputs."""
    if client is None:
        return "ℹ️ GenAI commentary disabled (no API key found)."
    prompt = f"""You are a financial analyst.
Given the forecast summary: {forecast_summary},
indicators: {indicators},
and recommendation: {recommendation},
write a clear, professional 2–3 paragraph market commentary for an investor audience."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",   # can also use "gpt-4o" or "gpt-4-turbo"
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error generating report: {e}"
