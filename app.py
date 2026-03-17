# app.py — Streamlit application สำหรับทำนายโอกาสไวรัลของคลิป YouTube
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px

# ===== การตั้งค่าหน้าเว็บ =====
st.set_page_config(
    page_title="ระบบทำนายโอกาสไวรัล YouTube",
    page_icon="🚀",          
    layout="centered",        
    initial_sidebar_state="expanded"
)

# ===== โหลดโมเดล =====
@st.cache_resource
def load_model():
    """โหลด pipeline ของ Random Forest — ทำครั้งเดียวตอนเริ่ม app เพื่อความรวดเร็ว"""
    # โหลดโมเดล Random Forest (ใช้ตัวนี้บน Windows จะไม่มีปัญหา DLL crash)
    pipeline = joblib.load("viral_rf_model.pkl")
    return pipeline

# โหลดโมเดลพร้อมแสดง Spinner
with st.spinner("กำลังเตรียมระบบ AI..."):
    pipeline = load_model()

# ===== Sidebar: ข้อมูลเกี่ยวกับโมเดล =====
with st.sidebar:
    st.header("ℹ️ เกี่ยวกับโมเดลนี้")
    st.write("**ประเภทโมเดล:** Random Forest Classifier")
    st.write("**ความแม่นยำ (F1-Score):** 76.0%")
    st.write("**ข้อมูล Train:** 800,000 คลิป")
    st.write("**เกณฑ์ไวรัล:** ยอดวิวติด Top 10% (มากกว่า 23,058 วิว)")

    st.divider()

    st.subheader("⚠️ ข้อควรระวัง")
    st.warning(
        "ผลลัพธ์นี้เป็นการประเมินความน่าจะเป็นจากสถิติในอดีตเท่านั้น "
        "โอกาสไวรัลจริงอาจขึ้นอยู่กับปัจจัยภายนอก เช่น อัลกอริทึมล่าสุดของแพลตฟอร์ม หรือกระแสสังคม ณ เวลานั้น"
    )

# ===== ส่วนหลัก: Header =====
st.title("🚀 ระบบประเมินโอกาสไวรัล YouTube")
st.markdown("""
กรอกสถิติของคลิปวิดีโอที่คุณคาดหวังด้านล่าง ระบบจะประเมินโอกาสที่คลิปของคุณจะมียอดวิวทะลุเข้าสู่ **Top 10%** โดยใช้โมเดล AI ที่เรียนรู้จากฐานข้อมูลผู้สร้างคอนเทนต์กว่า 1 ล้านรายการ
""")

st.divider()

# ===== ส่วนรับ Input =====
st.subheader("📋 กำหนดเป้าหมายคลิปของคุณ")

col1, col2 = st.columns(2)

with col1:
    duration_sec = st.number_input(
        "ความยาววิดีโอ (วินาที)",
        min_value=1, max_value=36000,
        value=600, step=10,
        help="ตัวอย่าง: คลิป 10 นาที = 600 วินาที"
    )

    likes = st.number_input(
        "เป้ายอด Likes",
        min_value=0, max_value=10000000,
        value=1000, step=100
    )

    sentiment_score = st.slider(
        "โทนอารมณ์ของคลิป (Sentiment)",
        min_value=-1.0, max_value=1.0,
        value=0.0, step=0.1,
        help="ค่า -1.0 คือเชิงลบ/ดราม่า และ 1.0 คือเชิงบวก/สนุกสนาน"
    )

with col2:
    shares = st.number_input(
        "เป้ายอด Shares",
        min_value=0, max_value=1000000,
        value=50, step=10,
        help="จำนวนครั้งที่คลิปถูกแชร์ต่อ (มีผลต่อความไวรัลสูงมาก)"
    )

    comments = st.number_input(
        "เป้ายอด Comments",
        min_value=0, max_value=1000000,
        value=100, step=10
    )

st.divider()

# ===== ปุ่มทำนายและแสดงผล =====
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_button = st.button(
        "🔍 ประเมินโอกาสไวรัล",
        use_container_width=True,
        type="primary"
    )

if predict_button:
    # รวบรวมข้อมูล ต้องเรียงลำดับให้ตรงกับตอนที่เทรนโมเดล
    input_data = pd.DataFrame([[
        duration_sec, likes, shares, comments, sentiment_score
    ]], columns=['duration_sec', 'likes', 'shares', 'comments', 'sentiment_score'])

    with st.spinner("กำลังวิเคราะห์ข้อมูล..."):
        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]

    prob_negative = probabilities[0]
    prob_positive = probabilities[1]

    st.subheader("🎯 ผลการประเมิน")

    if prediction == 1:
        st.success(f"""
        ### 🔥 มีโอกาสเป็นคลิปไวรัล!
        สถิติที่คุณตั้งเป้าไว้มีแนวโน้มสูงที่จะผลักดันให้คลิปนี้ติด Top 10%
        **ความน่าจะเป็น: {prob_positive*100:.1f}%**
        """)
    else:
        st.warning(f"""
        ### 🐢 ยังเข้าไม่ถึงเกณฑ์ไวรัล
        สถิติปัจจุบันอาจยังไม่พอที่จะผลักดันคลิปให้ติด Top 10% แนะนำให้เน้นเนื้อหาที่กระตุ้นยอดแชร์
        **ความน่าจะเป็น: {prob_positive*100:.1f}%**
        """)

    # แสดง progress bar ของความน่าจะเป็น
    st.write("**ระดับความสำเร็จ:**")
    st.progress(
        float(prob_positive),
        text=f"โอกาสเป็นไวรัล: {prob_positive*100:.1f}%"
    )

    # --- โบนัส: กราฟ Interactive Plotly ---
    st.markdown("---")
    st.subheader("📊 ปัจจัยใดมีผลต่อโมเดลนี้มากที่สุด?")
    
    # ดึงค่า Importance จาก Random Forest ใน Pipeline
    rf_model = pipeline.named_steps['rf_model']
    importance = rf_model.feature_importances_
    
    df_imp = pd.DataFrame({'ปัจจัย': input_data.columns, 'น้ำหนักความสำคัญ': importance})
    df_imp = df_imp.sort_values(by='น้ำหนักความสำคัญ', ascending=True)
    
    fig = px.bar(
        df_imp, x='น้ำหนักความสำคัญ', y='ปัจจัย', orientation='h', 
        color='น้ำหนักความสำคัญ', color_continuous_scale='Teal'
    )
    st.plotly_chart(fig, use_container_width=True)

    # แสดงข้อมูลสรุป
    with st.expander("📋 ดูข้อมูลที่กรอก"):
        summary = {
            "ความยาววิดีโอ (วินาที)": duration_sec,
            "ยอด Likes": likes,
            "ยอด Shares": shares,
            "ยอด Comments": comments,
            "โทนอารมณ์": sentiment_score
        }
        st.dataframe(
            pd.DataFrame.from_dict(summary, orient="index", columns=["ค่าเป้าหมาย"]),
            use_container_width=True
        )