# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import streamlit as st
from scipy.sparse import hstack, csr_matrix

# ===== Extracted from notebook =====
import re, ipaddress, numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.suspicious_words = ["login","verify","update","secure","account","free","win","bonus","gift","bank","paypal"]
        self.shorteners = ["bit.ly","goo.gl","tinyurl","t.co","ow.ly"]
    def is_ip(self, host):
        import ipaddress
        try: ipaddress.ip_address(host); return 1
        except: return 0
    def get_host(self, url):
        m = re.match(r"https?://([^/]+)", str(url).lower())
        return m.group(1) if m else ""
    def fit(self, X, y=None): return self
    def transform(self, X):
        feats=[]
        for u in X:
            u=str(u); host=self.get_host(u); q=u.split("?",1)[1] if "?" in u else ""; params=q.split("&") if q else []
            feats.append([
                len(u), len(host), u.count("."), u.count("-"), u.count("@"), u.count("?"),
                u.count("="), u.count("%"), u.count("&"), len(params),
                1 if u.lower().startswith("https://") else 0,
                self.is_ip(host),
                1 if any(s in u.lower() for s in self.shorteners) else 0,
                1 if any(w in u.lower() for w in self.suspicious_words) else 0
            ])
        return csr_matrix(np.array(feats, dtype=float))

tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)
# ===================================

@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open("model.pkl","rb") as f:
        clf = pickle.load(f)
    with open("tfidf.pkl","rb") as f:
        tfidf = pickle.load(f)
    with open("manual.pkl","rb") as f:
        man = pickle.load(f)
    return clf, tfidf, man

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

st.set_page_config(page_title="🔒 URL Malware Detector", page_icon="🛡️", layout="centered")
st.title("🔒 URL Malware Detector")
st.caption("تحليل الروابط لاكتشاف الروابط الضارة باستخدام TF-IDF + ميزات يدوية + نموذج مدرّب.")

# Sidebar examples
with st.sidebar:
    st.subheader("أمثلة جاهزة")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔗 مثال موقع سليم", use_container_width=True):
            st.session_state['sample_url'] = "https://www.wikipedia.org/"
    with col_b:
        if st.button("⚠️ مثال موقع ضار", use_container_width=True):
            st.session_state['sample_url'] = "http://paypal.com.security-alert.example.com/confirm%20info"
    st.caption("اضغط زرًا لملء الحقل تلقائيًا.")

# Default text uses session value if present
default_text = st.session_state.get('sample_url', 'https://www.wikipedia.org/')
url = st.text_input("أدخل الرابط هنا:", value=default_text, placeholder="https://example.com/path?...")

# ===== أمثلة جاهزة =====
with st.sidebar:
    st.subheader("أمثلة جاهزة")
    if st.button("🔗 مثال موقع سليم"):
        st.session_state['sample_url'] = "https://www.wikipedia.org/"
    if st.button("⚠️ مثال موقع ضار"):
        st.session_state['sample_url'] = "http://paypal.com.security-alert.example.com/confirm%20info"

default_text = st.session_state.get('sample_url', 'https://www.wikipedia.org/')
url = st.text_input("أدخل الرابط هنا:", value=default_text, placeholder="https://example.com/path?...")

if st.button("تحليل 🔍", type="primary") and url.strip():
    clf, tfidf, man = load_artifacts()

    # حوّل الرابط
    X_tfidf = tfidf.transform([url])
    try:
        X_manual = man.transform([url])
    except Exception as e:
        st.error("تعذر تطبيق URLFeatureExtractor.transform — تحقق من التوافق.")
        st.exception(e)
        st.stop()

    X = hstack([X_tfidf, X_manual])

    # تأكد من مطابقة عدد الميزات
    expected = getattr(clf, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        diff = expected - X.shape[1]
        st.warning(f"تصحيح تلقائي لعدد الميزات: الحالي = {X.shape[1]}, المتوقع = {expected}.")
        if diff > 0:
            pad = csr_matrix((X.shape[0], diff))
            X = hstack([X, pad])
        else:
            X = X[:, :expected]

    pred = int(clf.predict(X)[0])
    prob = get_positive_prob(clf, X)

    if pred == 1:
        st.error("النتيجة: ضار ❌")
    else:
        st.success("النتيجة: سليم ✅")

    if prob is not None:
        st.metric("درجة الثقة (احتمال الضار)", f"{prob:.3f}")
    else:
        st.caption("النموذج لا يدعم `predict_proba`.")

st.markdown("---")
st.caption("© 2025 — للاستخدام التعليمي فقط.")
