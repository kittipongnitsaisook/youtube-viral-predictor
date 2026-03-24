import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------
# 1. โหลดโมเดล (อย่าลืมเปลี่ยนชื่อไฟล์ให้ตรงกับโมเดลตัวใหม่ที่เทรนด้วย 3 ตัวแปร)
# ------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load('rf_model_3features.pkl') # <--- ใส่ชื่อไฟล์โมเดลใหม่ตรงนี้

pipeline = load_model()

# ------------------------------------------------
# 2. ส่วนหัวของหน้าเว็บ (Header)
# ------------------------------------------------
st.title("🚀 YouTube Viral Predictor")
st.markdown("ระบบ AI วิเคราะห์โอกาสความไวรัลของคลิปวิดีโอ (เป้าหมาย: Top 10% ยอดวิวสูงสุด)")
st.divider()

# ------------------------------------------------
# 3. ส่วนรับข้อมูลจากผู้ใช้ (Inputs) - เหลือแค่ 3 ตัวแปร
# ------------------------------------------------
st.subheader("📊 กรอกสถิติที่คาดหวังของคลิปคุณ")

col1, col2, col3 = st.columns(3)

with col1:
    shares = st.number_input("จำนวนยอดแชร์ (Shares)", min_value=0, value=1000, step=100)
with col2:
    likes = st.number_input("จำนวนยอดไลก์ (Likes)", min_value=0, value=5000, step=100)
with col3:
    comments = st.number_input("จำนวนคอมเมนต์ (Comments)", min_value=0, value=100, step=10)

# ------------------------------------------------
# 4. ส่วนประมวลผลและแสดงผลลัพธ์
# ------------------------------------------------
st.divider()

if st.button("🔮 ทำนายโอกาสไวรัล", type="primary", use_container_width=True):
    
    # สร้าง DataFrame ให้ตรงกับ 3 คอลัมน์ที่โมเดลรอรับ
    input_data = pd.DataFrame({
        'shares': [shares],
        'likes': [likes],
        'comments': [comments]
    })
    
    # ให้โมเดลทำนาย (ดึงค่า Probability ของคลาส 1)
    probabilities = pipeline.predict_proba(input_data)[0]
    viral_prob = probabilities[1] * 100 
    
    # แสดงผลลัพธ์
    st.subheader("ผลการวิเคราะห์จาก AI 🤖")
    
    # ทำ Progress bar สวยๆ
    st.progress(int(viral_prob))
    st.metric(label="โอกาสเป็นคลิปไวรัล (ติด Top 10%)", value=f"{viral_prob:.2f}%")
    
    # แสดงข้อความตามเกณฑ์ 50%
    if viral_prob >= 50:
        st.success("🔥 ยอดเยี่ยม! สถิติระดับนี้มีโอกาสสูงมากที่จะกลายเป็นคลิปไวรัล")
    else:
        st.warning("🐢 ยังไม่ถึงเกณฑ์ไวรัล แนะนำให้เน้นการทำคอนเทนต์ที่กระตุ้นให้คนกด 'แชร์' มากขึ้นครับ")

    st.caption("*หมายเหตุ: AI ประเมินจากข้อมูล YouTube 1 ล้านคลิป โดยให้น้ำหนักกับ 'ยอดแชร์' เป็นปัจจัยสำคัญที่สุด")
