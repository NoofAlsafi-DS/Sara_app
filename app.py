import streamlit as st
import pickle
from scipy.sparse import hstack
from urllib.parse import urlparse
from pathlib import Path

st.set_page_config(page_title="URL Malware Detector", page_icon="🔒", layout="centered")
st.title("🔒 URL Malware Detector")

# ====== تحميل النموذج والمحولات ======
@st.cache_resource
def load_artifacts():
    clf = pickle.load(open("model.pkl","rb"))
    tfidf = pickle.load(open("tfidf.pkl","rb"))
    man = pickle.load(open("manual.pkl","rb"))
    return clf, tfidf, man

clf, tfidf, man = load_artifacts()

# ====== إعدادات الواجهة ======
colA, colB = st.columns([2,1])
with colB:
    threshold = st.slider("عتبة الحكم على (ضار)", 0.5, 0.99, 0.9, 0.01)

with colA:
    url = st.text_input("أدخل الرابط هنا:", placeholder="https://example.com/login")

# صور الحالة (ملفّات محلية اختيارية)
MAL_IMG = Path("malicious.png")
BEN_IMG = Path("benign.png")

def show_status(label_str: str, prob: float):
    """عرض صورة/رمز + معلومات النتيجة."""
    col1, col2 = st.columns([1,2])
    with col1:
        if label_str == "ضار":
            if MAL_IMG.exists():
                st.image(str(MAL_IMG), use_container_width=True)
            else:
                st.markdown("### 🚨")
        else:
            if BEN_IMG.exists():
                st.image(str(BEN_IMG), use_container_width=True)
            else:
                st.markdown("### 🛡️")
    with col2:
        st.markdown(f"### النتيجة: **{label_str}**")
        if prob is not None:
            st.metric("احتمال الضار (model)", f"{prob:.3f}")
            st.caption(f"العتبة الحالية: {threshold:.2f} — إذا كان الاحتمال ≥ العتبة ➜ نصنّف «ضار»")

# ====== زر أمثلة سريعة ======
with st.expander("🧪 أمثلة سريعة"):
    ex1, ex2, ex3, ex4 = st.columns(4)
    if ex1.button("🎁 bit.ly/win-prize"):
        url = "http://bit.ly/win-prize-now?ref=secure-login"
    if ex2.button("🐙 GitHub"):
        url = "https://github.com/pytorch/pytorch"
    if ex3.button("🔑 IP/login.php"):
        url = "http://192.168.1.44/login.php"
    if ex4.button("🏫 University portal"):
        url = "https://university.edu/portal"
    if url:
        st.info(f"المثال المختار: {url}")

# ====== التنبؤ ======
def predict_one(u: str, th: float):
    Xt = tfidf.transform([u])
    Xm = man.transform([u])
    X = hstack([Xt, Xm])
    proba = clf.predict_proba(X)[0][1] if hasattr(clf, "predict_proba") else None
    # قرار باستخدام العتبة
    if proba is not None:
        pred = 1 if proba >= th else 0
    else:
        pred = clf.predict(X)[0]
    return pred, proba

go = st.button("تحليل")
if go and url.strip():
    # تطبيع بسيط
    u = url.strip()
    # تنبؤ
    pred, prob = predict_one(u, threshold)
    label_str = "ضار" if pred == 1 else "سليم"
    show_status(label_str, prob)

    # معلومات إضافية مفيدة للمستخدم
    with st.container(border=True):
        st.markdown("**تفاصيل الرابط**")
        try:
            p = urlparse(u)
            st.write({"scheme": p.scheme, "netloc": p.netloc, "path": p.path, "query": p.query})
        except Exception:
            st.write("تعذّر تحليل الرابط.")

    # تحذير عملي في حالة السليم مع احتمال مرتفع
    if prob is not None:
        if label_str == "سليم" and prob >= (threshold - 0.05):
            st.warning("الرابط صُنّف «سليم» لكن احتمال الضار قريب من العتبة. يُستحسن الحذر.")

# ====== ملاحظات ======
st.caption("""
**تلميح:** ضع ملفين صور في نفس مجلد التطبيق:
- `malicious.png` لواجهة «ضار»
- `benign.png` لواجهة «سليم»

إن لم توجد الصور، يستخدم التطبيق رموزًا بديلة (🚨 / 🛡️).
يمكنك تغيير العتبة من الشريط للحصول على حساسية أعلى أو أقل.
""")
