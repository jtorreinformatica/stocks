"""
Stock & Crypto Pattern Detection App
=====================================
Streamlit application for monitoring chart patterns in stocks and crypto.
"""

import json
import streamlit as st
from pathlib import Path

from data_fetcher import fetch_ohlcv
from chart_renderer import render_chart
from alerts import filter_today_patterns, get_alert_summary
from patterns import get_all_detectors, get_all_detector_names, get_detector_by_name

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="📈 Pattern Detector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Dark premium theme */
    .stApp {
        background-color: #0e1117;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Alert banner */
    .alert-banner {
        background: linear-gradient(135deg, #1a237e 0%, #4a148c 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        color: #e0e0e0;
        font-size: 14px;
        line-height: 1.6;
    }

    .alert-banner h3 {
        margin: 0 0 8px 0;
        color: #ffab40;
        font-size: 18px;
    }

    /* Pattern card */
    .pattern-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #252540 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .pattern-card .pattern-name {
        font-size: 15px;
        font-weight: 600;
        color: #bb86fc;
    }

    .pattern-card .pattern-detail {
        font-size: 13px;
        color: #9e9e9e;
        margin-top: 4px;
    }

    .confidence-high { color: #66bb6a; }
    .confidence-mid { color: #ffa726; }
    .confidence-low { color: #ef5350; }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #0a0a14;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 10px 0 20px 0;
    }

    .main-header h1 {
        background: linear-gradient(135deg, #42a5f5, #ab47bc, #ff7043);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }

    .main-header p {
        color: #9e9e9e;
        font-size: 14px;
        margin: 4px 0 0 0;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .status-ok { background: rgba(102,187,106,0.2); color: #66bb6a; }
    .status-warn { background: rgba(255,167,38,0.2); color: #ffa726; }
</style>
""", unsafe_allow_html=True)

# ── Config ───────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"assets": ["AAPL", "BTC-USD"], "period": "1y", "patterns_enabled": []}


def save_config(config: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


config = load_config()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.markdown("---")

    # Asset management
    st.markdown("### 📋 Listado de Activos")
    
    # Bulk input
    current_assets_str = ", ".join(config.get("assets", []))
    asset_input = st.text_area(
        "Introduce los tickers separados por coma o espacios",
        value=current_assets_str,
        placeholder="Ej: AAPL, BTC-USD, MSFT, TSLA, NVDA...",
        help="Introduce una lista larga de activos para escanear patrones.",
        height=150
    )
    
    # Process input
    new_assets = [a.strip().upper() for a in asset_input.replace("\n", ",").split(",") if a.strip()]
    if new_assets != config.get("assets"):
        config["assets"] = new_assets
        save_config(config)
        # We don't rerun here immediately to let the user finish typing multiple assets if they use text_area, 
        # but since Streamlit handles text_area updates on focus out/ctrl+enter, it's fine.

    st.markdown("---")

    # Period selector
    st.markdown("### 📅 Periodo")
    period = st.selectbox(
        "Periodo de datos",
        options=["3mo", "6mo", "1y", "2y", "5y"],
        index=["3mo", "6mo", "1y", "2y", "5y"].index(config.get("period", "1y")),
        label_visibility="collapsed",
    )
    if period != config.get("period"):
        config["period"] = period
        save_config(config)

    st.markdown("---")

    # Timeframe selector
    st.markdown("### ⏱️ Timeframe")
    timeframe_map = {
        "1 Día (1D)": "1d",
        "1 Semana (1W)": "1wk",
        "1 Mes (1M)": "1mo"
    }
    timeframe_label = st.selectbox(
        "Timeframe de búsqueda",
        options=list(timeframe_map.keys()),
        index=list(timeframe_map.values()).index(config.get("interval", "1d")),
        label_visibility="collapsed",
    )
    interval = timeframe_map[timeframe_label]
    if interval != config.get("interval"):
        config["interval"] = interval
        save_config(config)

    st.markdown("---")

    # Pattern selector
    st.markdown("### 🔍 Patrones")
    all_pattern_names = get_all_detector_names()
    
    # Default patterns: VCP and Cup and Handle
    default_patterns = ["VCP", "Cup and Handle"]
    current_defaults = config.get("patterns_enabled")
    if current_defaults is None:
        current_defaults = [p for p in default_patterns if p in all_pattern_names]
    else:
        current_defaults = [p for p in current_defaults if p in all_pattern_names]

    enabled_patterns = st.multiselect(
        "Patrones activos",
        options=all_pattern_names,
        default=current_defaults,
        label_visibility="collapsed",
    )
    config["patterns_enabled"] = enabled_patterns
    save_config(config)

    st.markdown("---")

    # Scan button
    scan_btn = st.button("🔎 Escanear patrones", use_container_width=True, type="primary")

    st.markdown("---")

    # Info
    st.markdown(
        f"<div style='text-align:center; color:#616161; font-size:12px;'>"
        f"📊 {len(config['assets'])} activos · "
        f"🔍 {len(enabled_patterns)} patrones"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="main-header">
        <h1>Pattern Detector</h1>
        <p>Detección de patrones chartistas en acciones y criptomonedas</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Main Content ─────────────────────────────────────────────────────────────

if not config["assets"]:
    st.info("👈 Añade activos en la barra lateral para comenzar.")
    st.stop()

if not enabled_patterns:
    st.warning("👈 Selecciona al menos un patrón en la barra lateral.")
    st.stop()

# Run analysis
all_matches: dict[str, list] = {}
high_confidence_symbols = []

# Filtering threshold
CONFIDENCE_THRESHOLD = 0.7

with st.spinner(f"📡 Escaneando {len(config['assets'])} activos..."):
    progress_bar = st.progress(0)
    for i, symbol in enumerate(config["assets"]):
        df = fetch_ohlcv(
            symbol, 
            period=config.get("period", "1y"), 
            interval=config.get("interval", "1d")
        )
        progress_bar.progress((i + 1) / len(config["assets"]))
        
        if df.empty:
            continue

        # Detect patterns
        symbol_matches = []
        for pattern_name in enabled_patterns:
            detector = get_detector_by_name(pattern_name)
            if detector:
                try:
                    found = detector.detect(df)
                    # Filter for high confidence and active today
                    filtered_found = [
                        m for m in found 
                        if m.confidence >= CONFIDENCE_THRESHOLD and m.is_active_today
                    ]
                    symbol_matches.extend(filtered_found)
                except Exception as e:
                    # Log error silently or show minimal warning
                    pass

        if symbol_matches:
            all_matches[symbol] = symbol_matches
            high_confidence_symbols.append(symbol)
    
    progress_bar.empty()

# ── Alerts ───────────────────────────────────────────────────────────────────

today_matches = filter_today_patterns(all_matches)
alerts = get_alert_summary(today_matches)

if alerts:
    alert_html = "<div class='alert-banner'><h3>🔔 Alertas de Hoy</h3>"
    for alert in alerts:
        alert_html += f"<div>{alert}</div>"
    alert_html += "</div>"
    st.markdown(alert_html, unsafe_allow_html=True)

    # Visual notification
    st.toast(f"🚨 {len(alerts)} patrón(es) detectado(s) hoy!", icon="🔔")

# ── Scanning Results ──────────────────────────────────────────────────────────

if not high_confidence_symbols:
    st.info("🔎 No se detectaron patrones de alta confianza hoy en la lista proporcionada.")
    st.stop()

st.success(f"🎯 Se han detectado **{len(high_confidence_symbols)}** activos con patrones de alta confianza para revisar hoy.")

# ── Charts ───────────────────────────────────────────────────────────────────

for symbol in high_confidence_symbols:
    df = fetch_ohlcv(
        symbol, 
        period=config.get("period", "1y"), 
        interval=config.get("interval", "1d")
    )
    if df.empty:
        continue

    matches = all_matches.get(symbol, [])

    # Symbol header
    col_title, col_stats = st.columns([3, 2])
    with col_title:
        price = df["Close"].iloc[-1]
        prev_price = df["Close"].iloc[-2] if len(df) > 1 else price
        change = (price - prev_price) / prev_price * 100
        change_icon = "🟢" if change >= 0 else "🔴"
        st.markdown(
            f"### {symbol} — ${price:.2f} {change_icon} {change:+.2f}%"
        )
    with col_stats:
        badge_text = f"🔥 {len(matches)} patrón(es) de ALTA CONFIANZA"
        st.markdown(
            f"<div style='text-align:right; padding-top:12px;'>"
            f"<span class='status-badge status-ok' style='background: rgba(187,134,252,0.2); color: #bb86fc;'>"
            f"{badge_text}"
            f"</span></div>",
            unsafe_allow_html=True,
        )

    # Chart
    fig = render_chart(df, symbol, matches)
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{symbol}")

    # Pattern details
    with st.expander(f"📋 Detalles de patrones en {symbol}", expanded=True):
        for m in matches:
            st.markdown(
                f"""<div class="pattern-card">
                    <div class="pattern-name">{m.pattern_name} 🔔 HOY</div>
                    <div class="pattern-detail">
                        📅 {m.start_date.strftime('%d/%m/%Y')} → {m.end_date.strftime('%d/%m/%Y')}
                        &nbsp;·&nbsp;
                        Confianza: <span class="confidence-high">{m.confidence:.0%}</span>
                        &nbsp;·&nbsp;
                        {m.description}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="text-align:center; color:#616161; font-size:12px; padding:20px 0;">
        Pattern Detector · Los patrones detectados son orientativos, no constituyen asesoría financiera.
    </div>
    """,
    unsafe_allow_html=True,
)
