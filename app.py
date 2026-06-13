import streamlit as st
from ultralytics import RTDETR
from PIL import Image
import numpy as np
import time
import io
import os
import gdown

st.set_page_config(
    page_title="AI Demonstrator — Classroom Engagement",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Serif+Display:ital@0;1&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

[data-testid="stAppViewContainer"] {
    background: #F7F4EF;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] { display: none; }
[data-testid="stFileUploadDropzone"] { display: none; }
[data-testid="stSidebar"] { display: none; }

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: 20px 0 16px;
    border-bottom: 1px solid #E2DDD7;
    margin-bottom: 20px;
}
.topbar-title {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #1C1916;
    letter-spacing: -0.3px;
}
.topbar-title em {
    font-style: italic;
    color: #7A6A5A;
}
.topbar-meta {
    font-size: 12px;
    color: #9E9187;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* ── Upload zone ── */
.upload-zone {
    border: 1px dashed #C9C2B8;
    border-radius: 10px;
    background: #FDFAF7;
    padding: 32px 24px;
    text-align: center;
    margin-bottom: 0;
    cursor: pointer;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #9E9187; }
.upload-zone-title {
    font-family: 'DM Serif Display', serif;
    font-size: 15px;
    color: #1C1916;
    margin-bottom: 4px;
}
.upload-zone-sub {
    font-size: 12px;
    color: #9E9187;
    font-weight: 300;
}

/* ── Image cards ── */
.img-card {
    background: #FDFAF7;
    border: 1px solid #E2DDD7;
    border-radius: 10px;
    overflow: hidden;
    height: 100%;
}
.img-card-header {
    padding: 8px 14px;
    background: #F2EDE7;
    border-bottom: 1px solid #E2DDD7;
    font-size: 11px;
    font-weight: 500;
    color: #7A6A5A;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 14px 0;
}
.metric-card {
    background: #FDFAF7;
    border: 1px solid #E2DDD7;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: left;
}
.metric-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9E9187;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    line-height: 1;
    margin-bottom: 2px;
}
.metric-sub {
    font-size: 11px;
    color: #9E9187;
    font-weight: 300;
}
.c-green  { color: #3B6D11; }
.c-amber  { color: #854F0B; }
.c-red    { color: #A32D2D; }
.c-muted  { color: #7A6A5A; font-size: 22px !important; }

/* ── Breakdown bars ── */
.breakdown {
    background: #FDFAF7;
    border: 1px solid #E2DDD7;
    border-radius: 10px;
    padding: 16px;
}
.breakdown-title {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9E9187;
    margin-bottom: 14px;
}
.bar-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.bar-row:last-of-type { margin-bottom: 0; }
.bar-name {
    font-size: 12px;
    color: #3D2F24;
    width: 68px;
    flex-shrink: 0;
    font-weight: 400;
}
.bar-track {
    flex: 1;
    height: 6px;
    background: #EDE8E2;
    border-radius: 99px;
    overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 99px; }
.bar-pct {
    font-size: 11px;
    color: #9E9187;
    width: 32px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Alerts ── */
.alert {
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    line-height: 1.5;
    margin-top: 10px;
    font-weight: 400;
}
.alert-ok   { background: #EAF3DE; color: #27500A; border: 1px solid #C0DD97; }
.alert-warn { background: #FAEEDA; color: #633806; border: 1px solid #EFC97A; }
.alert-bad  { background: #FCEBEB; color: #791F1F; border: 1px solid #F0A8A8; }

/* ── Right panel cards ── */
.panel-card {
    background: #FDFAF7;
    border: 1px solid #E2DDD7;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
}
.panel-card-title {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9E9187;
    margin-bottom: 12px;
}

/* ── Settings sliders ── */
.setting-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    font-size: 12px;
    color: #7A6A5A;
}
.setting-label { width: 70px; flex-shrink: 0; }

/* ── Session history ── */
.history-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #EDE8E2;
    font-size: 12px;
}
.history-row:last-child { border-bottom: none; padding-bottom: 0; }
.history-time { color: #9E9187; font-weight: 300; }
.history-rate { font-weight: 500; }

/* ── Legend ── */
.legend-row {
    display: flex;
    gap: 16px;
    font-size: 11px;
    color: #7A6A5A;
    margin-top: 8px;
}
.legend-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}

/* ── Download button ── */
.stDownloadButton > button {
    width: 100%;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    color: #3D2F24 !important;
    background: transparent !important;
    border: 1px solid #C9C2B8 !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    letter-spacing: 0.02em !important;
    transition: background 0.15s !important;
}
.stDownloadButton > button:hover {
    background: #F2EDE7 !important;
    border-color: #9E9187 !important;
}

/* ── Override streamlit defaults ── */
div[data-testid="column"] { padding: 0 !important; }
.block-container { padding: 0 24px 24px !important; max-width: 100% !important; }
section[data-testid="stFileUploadDropzone"] { display: none !important; }
.stProgress > div > div { background: #3B6D11 !important; }
[data-testid="stProgressBar"] > div { border-radius: 99px !important; }
</style>
""", unsafe_allow_html=True)

# ── Model ──────────────────────────────────────────────────────
MODEL_PATH  = "rtdetr_best.pt"
CLASS_NAMES = ["Attentive", "Distracted", "Sleepy"]

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... (first run only)"):
        gdown.download("https://drive.google.com/uc?id=1lxcXwQSNLEz1Pg1gTdHdz9m7hG4BtBw9", MODEL_PATH, quiet=False)

@st.cache_resource
def load_model(path):
    return RTDETR(path)

# ── Session state ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "conf" not in st.session_state:
    st.session_state.conf = 0.40
if "iou" not in st.session_state:
    st.session_state.iou = 0.50

model = load_model(MODEL_PATH)

# ── Top bar ────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-title">AI Demonstrator <em>— Classroom Engagement</em></div>
    <div class="topbar-meta">RT-DETR &nbsp;·&nbsp; real-time detection transformer</div>
</div>
""", unsafe_allow_html=True)

# ── Layout: left (content) + right (panel) ─────────────────────
left, right = st.columns([3, 1], gap="medium")

with right:
    # Settings card
    st.markdown('<div class="panel-card"><div class="panel-card-title">Detection settings</div>', unsafe_allow_html=True)
    st.session_state.conf = st.slider("Confidence", 0.10, 1.0, st.session_state.conf, 0.05, label_visibility="collapsed")
    st.markdown(f'<div class="setting-row"><span class="setting-label">Confidence</span><span style="font-size:12px;color:#3D2F24;font-weight:500">{st.session_state.conf:.2f}</span></div>', unsafe_allow_html=True)
    st.session_state.iou = st.slider("IoU", 0.10, 1.0, st.session_state.iou, 0.05, label_visibility="collapsed")
    st.markdown(f'<div class="setting-row"><span class="setting-label">IoU</span><span style="font-size:12px;color:#3D2F24;font-weight:500">{st.session_state.iou:.2f}</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="legend-row">
        <span><span class="legend-dot" style="background:#3B6D11"></span>Attentive</span>
        <span><span class="legend-dot" style="background:#BA7517"></span>Distracted</span>
        <span><span class="legend-dot" style="background:#A32D2D"></span>Sleepy</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Session history
    st.markdown('<div class="panel-card"><div class="panel-card-title">Session history</div>', unsafe_allow_html=True)
    if st.session_state.history:
        for entry in reversed(st.session_state.history[-6:]):
            rate = entry["eng_rate"]
            color = "#3B6D11" if rate >= 0.7 else "#854F0B" if rate >= 0.5 else "#A32D2D"
            st.markdown(f"""
            <div class="history-row">
                <span class="history-time">{entry["time"]} — snap {entry["snap"]}</span>
                <span class="history-rate" style="color:{color}">{round(rate * 100)}%</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:12px;color:#9E9187;font-weight:300">No snapshots yet this session.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with left:
    # Upload zone
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-zone-title">Drop a classroom image here</div>
        <div class="upload-zone-sub">JPG or PNG — annotated output appears alongside</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        img       = Image.open(uploaded).convert("RGB")
        img_array = np.array(img)

        with st.spinner("Running detection..."):
            t0      = time.time()
            results = model.predict(img_array, conf=st.session_state.conf, iou=st.session_state.iou, verbose=False)[0]
            elapsed = (time.time() - t0) * 1000

        counts      = {c: 0 for c in CLASS_NAMES}
        conf_scores = {c: [] for c in CLASS_NAMES}
        for box in results.boxes:
            cls_id = int(box.cls.item())
            name   = results.names[cls_id]
            counts[name] += 1
            conf_scores[name].append(float(box.conf.item()))

        total    = sum(counts.values()) or 1
        eng_rate = counts["Attentive"] / total
        a_pct    = round(counts["Attentive"]  / total * 100)
        d_pct    = round(counts["Distracted"] / total * 100)
        s_pct    = round(counts["Sleepy"]     / total * 100)

        annotated     = results.plot()
        annotated_rgb = annotated[:, :, ::-1]

        # Save to session history
        snap_num = len(st.session_state.history) + 1
        st.session_state.history.append({
            "time":     time.strftime("%H:%M"),
            "snap":     snap_num,
            "eng_rate": eng_rate,
            "counts":   dict(counts),
        })

        # Images
        col_orig, col_det = st.columns(2, gap="small")
        with col_orig:
            st.markdown('<div class="img-card"><div class="img-card-header">Original</div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_det:
            st.markdown('<div class="img-card"><div class="img-card-header">Detection result</div>', unsafe_allow_html=True)
            st.image(annotated_rgb, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Metric cards
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Attentive</div>
                <div class="metric-value c-green">{counts['Attentive']}</div>
                <div class="metric-sub">{a_pct}% of class</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Distracted</div>
                <div class="metric-value c-amber">{counts['Distracted']}</div>
                <div class="metric-sub">{d_pct}% of class</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sleepy</div>
                <div class="metric-value c-red">{counts['Sleepy']}</div>
                <div class="metric-sub">{s_pct}% of class</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Inference</div>
                <div class="metric-value c-muted">{elapsed:.0f}ms</div>
                <div class="metric-sub">conf {st.session_state.conf:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Breakdown bars
        avg_conf_a = round(np.mean(conf_scores['Attentive'])  * 100) if conf_scores['Attentive']  else 0
        avg_conf_d = round(np.mean(conf_scores['Distracted']) * 100) if conf_scores['Distracted'] else 0
        avg_conf_s = round(np.mean(conf_scores['Sleepy'])     * 100) if conf_scores['Sleepy']     else 0

        st.markdown(f"""
        <div class="breakdown">
            <div class="breakdown-title">Engagement breakdown</div>
            <div class="bar-row">
                <div class="bar-name">Attentive</div>
                <div class="bar-track"><div class="bar-fill" style="width:{a_pct}%;background:#3B6D11;"></div></div>
                <div class="bar-pct">{a_pct}%</div>
            </div>
            <div class="bar-row">
                <div class="bar-name">Distracted</div>
                <div class="bar-track"><div class="bar-fill" style="width:{d_pct}%;background:#BA7517;"></div></div>
                <div class="bar-pct">{d_pct}%</div>
            </div>
            <div class="bar-row">
                <div class="bar-name">Sleepy</div>
                <div class="bar-track"><div class="bar-fill" style="width:{s_pct}%;background:#A32D2D;"></div></div>
                <div class="bar-pct">{s_pct}%</div>
            </div>
            <div style="margin-top:12px;font-size:11px;color:#9E9187;font-weight:300;">
                Avg detection confidence — attentive {avg_conf_a}% &nbsp;·&nbsp; distracted {avg_conf_d}% &nbsp;·&nbsp; sleepy {avg_conf_s}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Engagement progress
        st.markdown("<div style='margin-top:12px;font-size:11px;color:#9E9187;font-weight:300;letter-spacing:0.04em;text-transform:uppercase;margin-bottom:6px;'>Overall engagement rate</div>", unsafe_allow_html=True)
        st.progress(eng_rate)

        # Alerts
        alerts_html = ""
        if eng_rate >= 0.7:
            alerts_html += f'<div class="alert alert-ok">Good engagement — {round(eng_rate*100)}% of students appear attentive.</div>'
        if counts["Distracted"] / total > 0.4:
            alerts_html += f'<div class="alert alert-warn">High distraction rate ({d_pct}%) — consider varying your teaching approach.</div>'
        if counts["Sleepy"] > 0:
            alerts_html += f'<div class="alert alert-bad">{counts["Sleepy"]} student(s) detected as sleepy — consider a short break or activity change.</div>'
        if alerts_html:
            st.markdown(alerts_html, unsafe_allow_html=True)

        # Download
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        buf = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="PNG")
        st.download_button(
            label="Download annotated image",
            data=buf.getvalue(),
            file_name="engagement_result.png",
            mime="image/png",
        )

    else:
        st.markdown("""
        <div style="margin-top:24px;text-align:center;color:#9E9187;font-size:13px;font-weight:300;">
            Upload an image above to begin analysis.
        </div>
        """, unsafe_allow_html=True)
