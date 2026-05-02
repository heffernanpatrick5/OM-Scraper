"""
Data Center Offering Memorandum Extractor — Streamlit Version
=============================================================
Install dependencies:
    pip install streamlit anthropic pdfplumber pdf2image Pillow

Run locally:
    streamlit run datacenter_om_extractor_streamlit.py

Deploy free:
    1. Push this file + requirements.txt to a GitHub repo
    2. Go to share.streamlit.io → New app → select your repo
"""

import streamlit as st
import json
import base64
import io
import os

# ── Optional imports (graceful degradation) ───────────────────────────────────
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from PIL import Image
except ImportError:
    Image = None


# ── Field Definitions ─────────────────────────────────────────────────────────

SHARED_FIELDS = [
    "property_name",
    "address",
    "city",
    "state",
    "zip_code",
    "market_submarket",
    "site_acreage",
    "asking_price",
    "price_per_acre",
    "seller_sponsor",
    "broker",
    "om_date",
    "zoning",
    "fiber_connectivity",
    "notes",
]

POWERED_LAND_FIELDS = [
    "total_mw_capacity_site",
    "permitted_mw",
    "critical_it_load_mw",
    "power_utility_provider",
    "substation_name_distance",
    "available_power_mw",
    "estimated_delivery_timeline",
    "power_redundancy",
    "shovel_ready",
    "estimated_construction_cost_per_mw",
    "proposed_building_sf",
    "proposed_num_buildings",
    "cooling_type",
    "water_rights_availability",
    "incentives_tax_abatements",
    "opportunity_zone",
    "environmental_permitting_issues",
]

STABILIZED_FIELDS = [
    "total_building_sf",
    "total_data_hall_sf",
    "total_installed_mw",
    "critical_it_load_mw",
    "power_redundancy",
    "cooling_type",
    "year_built",
    "year_renovated",
    "occupancy_rate",
    "number_of_tenants",
    "anchor_tenants",
    "walt_years",
    "lease_expiration_schedule",
    "gross_revenue",
    "noi",
    "ebitda",
    "cap_rate",
    "price_per_mw",
    "price_per_sf",
    "pue",
    "tier_classification",
    "colocation_vs_hyperscale",
    "pct_hyperscale_tenancy",
    "remaining_expansion_capacity_mw",
    "debt_existing_financing",
]


# ── PDF Processing ────────────────────────────────────────────────────────────

def is_garbled(text: str) -> bool:
    if not text or len(text.strip()) < 20:
        return True
    words = text.split()
    if not words:
        return True
    real_words = [w for w in words if any(c in "aeiouAEIOU" for c in w)]
    return len(real_words) / len(words) < 0.25


def image_coverage_ratio(page) -> float:
    page_area = page.width * page.height
    if page_area == 0:
        return 0.0
    img_area = sum((i.get("width", 0) * i.get("height", 0)) for i in page.images)
    return min(img_area / page_area, 1.0)


def classify_page(page) -> str:
    text = page.extract_text() or ""
    word_count = len(text.split())
    if word_count == 0:
        return "vision"
    if is_garbled(text):
        return "vision"
    if image_coverage_ratio(page) > 0.60:
        return "vision"
    if word_count < 40 and len(page.images) > 1:
        return "vision"
    return "text"


def page_to_base64(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def extract_pdf_content(pdf_bytes: bytes) -> dict:
    """
    Hybrid extraction from raw PDF bytes.
    Returns text pages, base64 image pages, and strategy map.
    """
    result = {"text_pages": [], "image_pages": [], "strategy_map": {}}

    if pdfplumber is None:
        st.error("pdfplumber not installed. Run: pip install pdfplumber")
        return result

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        vision_page_numbers = []
        total = len(pdf.pages)

        progress = st.progress(0, text="Classifying pages…")

        for i, page in enumerate(pdf.pages):
            strategy = classify_page(page)
            result["strategy_map"][i + 1] = strategy

            if strategy == "text":
                text = page.extract_text() or ""
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if row:
                            text += "\n" + "\t".join(
                                [str(c) if c else "" for c in row]
                            )
                result["text_pages"].append((i + 1, text))
            else:
                vision_page_numbers.append(i + 1)

            progress.progress(
                int((i + 1) / total * 40),
                text=f"Classifying pages… {i+1}/{total}",
            )

        # Rasterize vision pages
        if vision_page_numbers and convert_from_bytes is not None:
            progress.progress(45, text=f"Rasterizing {len(vision_page_numbers)} image page(s)…")
            images = convert_from_bytes(pdf_bytes, dpi=150)
            for page_num in vision_page_numbers:
                if page_num - 1 < len(images):
                    b64 = page_to_base64(images[page_num - 1])
                    result["image_pages"].append((page_num, b64))

        progress.progress(50, text="PDF processing complete.")
        progress.empty()

    return result


# ── Claude API ────────────────────────────────────────────────────────────────

def build_prompt(deal_type: str) -> str:
    if deal_type == "Powered Land / Development":
        fields = SHARED_FIELDS + POWERED_LAND_FIELDS
    else:
        fields = SHARED_FIELDS + STABILIZED_FIELDS

    field_list = "\n".join(f"  - {f}" for f in fields)

    return f"""You are an expert commercial real estate analyst specializing in data center assets.

Extract the following fields from the offering memorandum content provided.
Return ONLY a valid JSON object — no markdown, no preamble, no backticks.
Use null for any field you cannot find or confidently extract.
For numeric fields, return numbers only (no $ signs, commas, or units).
For 'power_redundancy', return the tier string (e.g. "2N", "N+1").
For boolean fields like 'shovel_ready' and 'opportunity_zone', return true/false/null.

Deal type: {deal_type}

Fields to extract:
{field_list}

Offering Memorandum content follows:
"""


def call_claude(api_key: str, deal_type: str, pdf_content: dict) -> dict:
    if anthropic is None:
        st.error("anthropic package not installed. Run: pip install anthropic")
        return {}

    client = anthropic.Anthropic(api_key=api_key)
    prompt_prefix = build_prompt(deal_type)
    content = []

    # Text pages
    if pdf_content["text_pages"]:
        combined_text = ""
        for page_num, text in pdf_content["text_pages"]:
            combined_text += f"\n\n--- Page {page_num} (text) ---\n{text}"
        content.append({"type": "text", "text": prompt_prefix + combined_text})
    else:
        content.append({"type": "text", "text": prompt_prefix})

    # Vision pages
    if pdf_content["image_pages"]:
        content.append({
            "type": "text",
            "text": "\n\nThe following pages were image-based:",
        })
        for page_num, b64 in pdf_content["image_pages"]:
            content.append({"type": "text", "text": f"\n--- Page {page_num} (image) ---"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })

    progress = st.progress(70, text="Sending to Claude API…")

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": content}],
    )

    progress.progress(90, text="Parsing response…")

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    progress.progress(100, text="Done!")
    progress.empty()

    return json.loads(raw)


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_field_name(field: str) -> str:
    """Convert snake_case to Title Case for display."""
    return field.replace("_", " ").title()


def render_value(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Data Center OM Extractor",
    page_icon="⚡",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f17; }
    .block-container { padding-top: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e2e;
        color: #a6adc8;
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #313244 !important;
        color: #89b4fa !important;
    }
    div[data-testid="metric-container"] {
        background-color: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .field-row-found { color: #a6e3a1; }
    .field-row-null  { color: #6c7086; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ── API Key — read from Streamlit Secrets / environment only ──────────────────
api_key = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## ⚡ Data Center OM Extractor")
st.markdown("Extract structured fields from data center offering memorandums using AI.")
st.divider()

# ── Deal Type Toggle ──────────────────────────────────────────────────────────
deal_type = st.radio(
    "Deal Type",
    options=["Stabilized", "Powered Land / Development"],
    horizontal=True,
    help="Determines which fields are extracted.",
)

st.markdown("")

# ── Main — Upload ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Offering Memorandum (PDF)",
    type=["pdf"],
    help="Supports both digital and scanned PDFs.",
)

if uploaded_file:
    col1, col2, col3 = st.columns(3)
    col1.metric("File", uploaded_file.name[:28] + ("…" if len(uploaded_file.name) > 28 else ""))
    col2.metric("Size", f"{uploaded_file.size / 1024:.0f} KB")
    col3.metric("Deal Type", deal_type.split("/")[0].strip())

    st.divider()

    if not api_key:
        st.error("⚠️ ANTHROPIC_API_KEY is not set. Add it to Streamlit Secrets.")
    else:
        if st.button("⚡ Extract Fields", type="primary", use_container_width=True):

            with st.status("Processing offering memorandum…", expanded=True) as status:

                try:
                    # Step 1 — PDF extraction
                    st.write("📄 Reading and classifying PDF pages…")
                    pdf_bytes = uploaded_file.read()
                    pdf_content = extract_pdf_content(pdf_bytes)

                    text_count   = sum(1 for s in pdf_content["strategy_map"].values() if s == "text")
                    vision_count = sum(1 for s in pdf_content["strategy_map"].values() if s == "vision")
                    total_pages  = len(pdf_content["strategy_map"])

                    st.write(f"✅ {total_pages} pages classified — "
                             f"{text_count} text, {vision_count} image")

                    # Step 2 — Claude API
                    st.write("🤖 Sending to Claude for extraction…")
                    result = call_claude(api_key, deal_type, pdf_content)

                    # Store in session state
                    st.session_state["result"] = result
                    st.session_state["strategy_map"] = pdf_content["strategy_map"]
                    st.session_state["deal_type"] = deal_type
                    st.session_state["filename"] = uploaded_file.name

                    found = sum(1 for v in result.values() if v is not None)
                    status.update(
                        label=f"✅ Extraction complete — {found}/{len(result)} fields found",
                        state="complete",
                    )

                except json.JSONDecodeError as e:
                    status.update(label="❌ Failed to parse Claude response", state="error")
                    st.error(f"Claude returned non-JSON output: {e}")
                except Exception as e:
                    status.update(label="❌ Error during extraction", state="error")
                    st.error(str(e))

# ── Results ───────────────────────────────────────────────────────────────────
if "result" in st.session_state:
    result       = st.session_state["result"]
    strategy_map = st.session_state["strategy_map"]
    filename     = st.session_state.get("filename", "")

    st.divider()
    st.markdown(f"### 📊 Results — {filename}")

    # Summary metrics
    found      = sum(1 for v in result.values() if v is not None)
    total      = len(result)
    text_pages = sum(1 for s in strategy_map.values() if s == "text")
    vis_pages  = sum(1 for s in strategy_map.values() if s == "vision")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fields Found",    f"{found}/{total}")
    m2.metric("Coverage",        f"{found/total*100:.0f}%")
    m3.metric("Text Pages",      text_pages)
    m4.metric("Vision Pages",    vis_pages)

    st.divider()

    # Output tabs
    tab1, tab2, tab3 = st.tabs(["📋 Field Table", "{ } JSON", "🗺️ Page Strategy"])

    # ── Tab 1: Field Table ──
    with tab1:
        saved_deal_type = st.session_state.get("deal_type", "Stabilized")
        if saved_deal_type == "Powered Land / Development":
            sections = {
                "Shared Fields": SHARED_FIELDS,
                "Powered Land / Development Fields": POWERED_LAND_FIELDS,
            }
        else:
            sections = {
                "Shared Fields": SHARED_FIELDS,
                "Stabilized Fields": STABILIZED_FIELDS,
            }

        for section_name, field_list in sections.items():
            st.markdown(f"**{section_name}**")
            rows = []
            for field in field_list:
                value = result.get(field)
                rows.append({
                    "Field": format_field_name(field),
                    "Value": render_value(value),
                    "Status": "✅" if value is not None else "—",
                })
            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Status": st.column_config.TextColumn(width="small"),
                    "Field":  st.column_config.TextColumn(width="medium"),
                    "Value":  st.column_config.TextColumn(width="large"),
                },
            )
            st.markdown("")

    # ── Tab 2: JSON ──
    with tab2:
        pretty_json = json.dumps(result, indent=2)
        st.code(pretty_json, language="json")

        st.download_button(
            label="⬇️ Download JSON",
            data=pretty_json,
            file_name=f"{filename.replace('.pdf','')}_extracted.json",
            mime="application/json",
        )

    # ── Tab 3: Page Strategy ──
    with tab3:
        st.markdown("Shows which extraction method was used for each page.")
        strategy_rows = [
            {
                "Page":     p,
                "Strategy": "🔤 Text" if s == "text" else "🖼️ Vision",
                "Method":   "pdfplumber (cheaper)" if s == "text" else "Claude Vision (image-based)",
            }
            for p, s in sorted(strategy_map.items())
        ]
        st.dataframe(strategy_rows, use_container_width=True, hide_index=True)

elif not uploaded_file:
    st.info("👆 Upload a PDF offering memorandum to get started.")
