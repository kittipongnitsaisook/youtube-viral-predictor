# 🚀 YouTube Viral Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://youtube-viral-predictor-ctekckwdgcwzypawmzgmpp.streamlit.app/#39c3d699)

เว็บแอปพลิเคชันสำหรับทำนายโอกาสที่คลิปวิดีโอบน YouTube จะกลายเป็นคลิป "ไวรัล" (ติดอันดับ Top 10% ของแพลตฟอร์ม) โดยใช้โมเดล Machine Learning วิเคราะห์จากสถิติและปัจจัยแวดล้อมต่างๆ ของวิดีโอ

## 📌 ที่มาและความสำคัญของปัญหา (Problem Statement)
ในปัจจุบัน การแข่งขันบนแพลตฟอร์ม YouTube มีสูงมาก ครีเอเตอร์และนักการตลาดจำเป็นต้องรู้ว่าปัจจัยใดที่ส่งผลให้คลิปวิดีโอประสบความสำเร็จ โปรเจคนี้จึงนำข้อมูล **YouTube 1M Global Creator Analytics** มาวิเคราะห์เพื่อตอบคำถามทางธุรกิจว่า *"สถิติแบบไหนที่ผลักดันให้คลิปมียอดวิวทะลุเกณฑ์ Top 10% (มากกว่า 23,058 วิว) ได้สำเร็จ?"* ซึ่งข้อมูลนี้จะช่วยให้ครีเอเตอร์สามารถวางกลยุทธ์การทำคอนเทนต์ได้อย่างแม่นยำและลดความเสี่ยงในการลงทุน

## 📊 ข้อมูลที่ใช้ (Dataset & Features)
ใช้ข้อมูลวิดีโอจำนวนกว่า 1 ล้านรายการ โดยดึงปัจจัย (Features) ที่สำคัญมาใช้ในการทำนาย ได้แก่:
- `duration_sec`: ความยาวของวิดีโอ (วินาที)
- `likes`: จำนวนการกดถูกใจ
- `comments`: จำนวนความคิดเห็น
- `shares`: จำนวนการแชร์ (ปัจจัยที่มีน้ำหนักสูงสุด)
- `sentiment_score`: โทนอารมณ์ของคลิป (-1.0 ถึง 1.0)

## 🤖 โมเดล Machine Learning (Model Approach)
- **Algorithm:** Random Forest Classifier (เลือกใช้เนื่องจากมีความแม่นยำสูงและมีความเสถียรในการ Deploy ข้ามระบบปฏิบัติการ)
- **Target Variable:** `is_viral` (1 = Viral [Top 10%], 0 = Not Viral)
- **Evaluation Metric:** เลือกใช้ **F1-Score** แทน Accuracy เนื่องจากชุดข้อมูลมีลักษณะ Imbalanced Data (90:10)
- **Performance:** โมเดลสามารถทำคะแนน F1-Score ได้ถึง **0.76** (76.0%)

## ✨ ฟีเจอร์หลักของเว็บแอป (App Features)
1. **Interactive Prediction:** ผู้ใช้สามารถกรอกตัวเลขเป้าหมายที่คาดหวัง เพื่อดูความน่าจะเป็น (Probability) ในการเป็นไวรัล
2. **Feature Importance Chart:** กราฟแบบ Interactive (Plotly) ที่แสดงให้ผู้ใช้เห็นว่าปัจจัยใดมีผลต่อการตัดสินใจของ AI มากที่สุด เพื่อนำไปปรับปรุงคอนเทนต์
3. **Responsive UI:** ออกแบบหน้าตาให้ใช้งานง่าย รองรับทั้งบนคอมพิวเตอร์และสมาร์ทโฟน

## 🛠️ เครื่องมือที่ใช้ (Tech Stack)
- **Data Science:** Python, Pandas, Numpy
- **Machine Learning:** Scikit-Learn, Joblib
- **Data Visualization:** Matplotlib, Seaborn, Plotly
- **Web Deployment:** Streamlit, Streamlit Community Cloud

## 💻 วิธีการติดตั้งและรันโปรแกรมบนเครื่อง (Local Installation)
หากต้องการนำโปรเจคนี้ไปรันบนเครื่องคอมพิวเตอร์ของคุณเอง สามารถทำตามขั้นตอนต่อไปนี้ได้:

1. Clone repository นี้ลงเครื่อง:
   ```bash
   git clone [https://github.com/](https://github.com/)[kittipongnitsaisook]/youtube-viral-predictor.git
   cd youtube-viral-predictor

2. ติดตั้งไลบรารีที่จำเป็น:
   ```bash
   pip install -r requirements.txt

4. สั่งรันแอปพลิเคชัน:
   ```bash
   streamlit run app.py
