import os
import re
import urllib.parse
import pickle
import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack

# ==============================
# 1) ManualFeatures class (must match the name used when saving manual.pkl)
# ==============================
class ManualFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for url in X:
            u = str(url or "")
            p = urllib.parse.urlparse(u)
            host = p.netloc or ""
            path = p.path or ""
            feats.append([
                len(u),                                   # total length
                u.count('-'),                             # '-' count
                u.count('@'),                             # '@' count
                u.count('?'),                             # '?' count
                u.count('%'),                             # '%' count
                u.count('.'),                             # '.' count
                sum(c.isdigit() for c in u),              # digits count
                1 if re.search(r'\b\d+\.\d+\.\d+\.\d+\b', host) else 0,  # IP in domain
                1 if u.lower().startswith('https') else 0,               # HTTPS flag
                len(path),                                # path length
            ])
        return np.array(feats, dtype=float)

# ==============================
# 2) Cached loaders for artifacts
# ==============================
# ⚠️ تعريف الكلاس المطلوب بنفس الاسم
class URLFeatureExtractor:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import numpy as np
        import re, urllib.parse
        feats = []
        for url in X:
            u = str(url or "")
            p = urllib.parse.urlparse(u)
            host = p.netloc or ""
            path = p.path or ""
            feats.append([
                len(u),                                   # طول الرابط
                u.count('-'),                             # عدد '-'
                u.count('@'),                             # عدد '@'
                u.count('?'),                             # عدد '?'
                u.count('%'),                             # عدد '%'
                u.count('.'),                             # عدد النقاط
                sum(c.isdigit() for c in u),              # عدد الأرقام
                1 if re.search(r'\d+\.\d+\.\d+\.\d+', host) else 0,  # وجود IP
                1 if u.lower().startswith('https') else 0,           # HTTPS
                len(path),                                # طول المسار
            ])
        return np.array(feats, dtype=float)

@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        with open("model.pkl","rb") as f:
            clf = pickle.load(f)
        with open("tfidf.pkl","rb") as f:
            tfidf = pickle.load(f)
        with open("manual.pkl","rb") as f:
            man = pickle.load(f)
        return clf, tfidf, man
    except Exception as e:
        st.exception(e)
        st.stop()

# Helper: safe probability extraction
def get_positive_prob(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
    if hasattr(model, "decision_function"):
        val = model.decision_function(X)
        if np.ndim(val) == 1:
            return float(1.0 / (1.0 + np.exp(-val[0])))
    return None

# Helper: small feature preview for a single URL
def preview_manual_features(url: str):
    u = str(url or "")
    p = urllib.parse.urlparse(u)
    host = p.netloc or ""
    path = p.path or ""
    return {
        "length": len(u),
        "dash_count": u.count('-'),
        "at_count": u.count('@'),
        "question_mark": u.count('?'),
        "percent_count": u.count('%'),
        "dot_count": u.count('.'),
        "digits_count": sum(c.isdigit() for c in u),
        "has_ip_in_domain": bool(re.search(r'\b\d+\.\d+\.\d+\.\d+\b', host)),
        "is_https": u.lower().startswith('https'),
        "path_length": len(path),
    }

# ==============================
# 3) UI
# ==============================
st.set_page_config(page_title="🔒 URL Malware Detector", page_icon="🛡️", layout="centered")
st.title("🔒 URL Malware Detector")
st.caption("تحليل الروابط لاكتشاف الروابط الضارة باستخدام TF-IDF + ميزات يدوية + نموذج مدرّب.")

with st.sidebar:
    st.markdown("## الإعدادات")
    st.markdown("هذا التطبيق يحتاج الملفات: `model.pkl`, `tfidf.pkl`, `manual.pkl`.")
    st.markdown("ضع صورتي `safe.png` و `malicious.png` اختياريًا لعرض صورة توضيحية للنتيجة.")
    st.divider()
    st.markdown("### أمثلة جاهزة")
    examples_safe = [
        "https://www.wikipedia.org/",
        "https://www.openai.com/research/",
    ]
    examples_bad = [
        "http://198.51.100.23/login/verify?acc=123",
        "http://paypal.com.security-alert.example.com/confirm%20info",
    ]
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        if st.button("مثال سليم", use_container_width=True):
            st.session_state['sample_url'] = examples_safe[0]
    with ex_col2:
        if st.button("مثال ضار", use_container_width=True):
            st.session_state['sample_url'] = examples_bad[0]

default_text = st.session_state.get('sample_url', '')
url = st.text_input("أدخل الرابط هنا:", value=default_text, placeholder="https://example.com/path?...")

analyze = st.button("تحليل 🔍", type="primary")

if analyze and url.strip():
    clf, tfidf, man = load_artifacts()

    try:
        X_tfidf = tfidf.transform([url])
    except Exception as e:
        st.error("تعذر تحويل الرابط باستخدام TF-IDF. تحقق من توافق نسخة scikit-learn/التوكنايزر.")
        st.exception(e)
        st.stop()

    try:
        X_manual = man.transform([url]) if hasattr(man, "transform") else ManualFeatures().transform([url])
    except Exception:
        X_manual = ManualFeatures().transform([url])

    try:
        X = hstack([X_tfidf, X_manual])
    except Exception as e:
        st.error("تعذر دمج الميزات. تأكد من أشكال المصفوفات وحجمها.")
        st.exception(e)
        st.stop()

    try:
        pred = int(clf.predict(X)[0])
    except Exception as e:
        st.error("تعذر إجراء التنبؤ. تحقق من توافق النموذج مع الميزات.")
        st.exception(e)
        st.stop()

    prob = get_positive_prob(clf, X)

    st.divider()
    left, right = st.columns([1,1])

    with left:
        if pred == 1:
            st.subheader("النتيجة: **ضار** ❌")
            if os.path.exists("malicious.png"):
                st.image("malicious.png", caption="تحذير: رابط ضار", use_container_width=True)
            else:
                st.warning("صورة 'malicious.png' غير موجودة. أضفها لاستخدام العرض المرئي.")
        else:
            st.subheader("النتيجة: **سليم** ✅")
            if os.path.exists("safe.png"):
                st.image("safe.png", caption="رابط سليم", use_container_width=True)
            else:
                st.info("صورة 'safe.png' غير موجودة. أضفها لاستخدام العرض المرئي.")

        if prob is not None:
            st.metric(label="درجة الثقة (احتمال الضار)", value=f"{prob:.3f}")
        else:
            st.caption("النموذج لا يدعم `predict_proba`، عُرضت النتيجة بدون درجة ثقة.")

    with right:
        st.markdown("#### معاينة بعض الميزات اليدوية")
        feats = preview_manual_features(url)
        st.dataframe({ "الميزة": list(feats.keys()), "القيمة": list(feats.values()) })

    with st.expander("تفاصيل تقنية"):
        st.write("• تم استخدام ميزات TF-IDF للنص الكامل للرابط + ميزات يدوية مثل الطول ووجود IP.\n"
                 "• إذا تغيّرت نسخ المكتبات بين التدريب والتشغيل قد تظهر أخطاء توافق. ثبّت نفس النسخ في requirements.txt.")

st.markdown("---")
st.caption("© 2025 — تطبيق توضيحي لا يُعد أداة أمنية نهائية. استخدمه كإرشاد أولي فقط.")
