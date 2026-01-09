import streamlit as st
import numpy as np
import mne
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import plotly.express as px
import pandas as pd
from scipy import signal

@st.cache_data
def load_meg_action_demo():
    sfreq = 1000
    n_epochs = 300
    ch_names = ['MEG_0111', 'MEG_0222', 'MEG_0333']  # M1L, M1R, SMA
    info = mne.create_info(ch_names, sfreq, 'grad')
    
    duration = 1.5
    data = np.random.randn(n_epochs, len(ch_names), int(duration*sfreq))
    labels = np.random.choice([0, 1], n_epochs, p=[0.4, 0.6])
    
    for i in range(n_epochs):
        t = np.linspace(-1, 0.5, data.shape[-1])
        if labels[i] == 0:  # Mouvt GAUCHE → M1 Right
            data[i, 1, int(0.4*sfreq):] -= 15 * np.exp(-((t[int(0.4*sfreq):]+0.3)/0.15)**2)
        else:
            data[i, 0, int(0.4*sfreq):] -= 15 * np.exp(-((t[int(0.4*sfreq):]+0.3)/0.15)**2)
    
    raw = mne.EpochsArray(data, info, tmin=-1.0)
    return raw, labels

def meg_preprocessing(raw):
    raw.filter(1, 50, fir_design='firwin')
    return raw

def extract_meg_action_features(raw, labels):
    epochs = raw.crop(tmin=-0.7, tmax=0).get_data()
    
    features = []
    for i, epoch in enumerate(epochs):
        m1_left, m1_right, sma = epoch[0], epoch[1], epoch[2]
        
        roi_left = np.mean(m1_left[raw.times > -0.5])
        roi_right = np.mean(m1_right[raw.times > -0.5])
        roi_lat = np.mean(sma[raw.times > -0.5])
        
        slope_left = np.polyfit(np.arange(200), m1_left[-200:], 1)[0]
        slope_right = np.polyfit(np.arange(200), m1_right[-200:], 1)[0]
        
        f, psd_left = signal.welch(m1_left, raw.info['sfreq'], fmin=8, fmax=13)
        f, psd_right = signal.welch(m1_right, raw.info['sfreq'], fmin=8, fmax=13)
        alpha_asym = np.mean(psd_left) - np.mean(psd_right)
        
        features.append([roi_left, roi_right, roi_lat, slope_left, slope_right, alpha_asym])
    
    return np.array(features)

st.title(" MEG Action Potential Detector - Localisation Motrice")
st.markdown("**Potentiel d'Action MEG M1/SMA: Intention Gauche/Droite localisée mm-précision.**")

raw_meg, meg_labels = load_meg_action_demo()

if st.button(" Prétraitement MEG"):
    raw_meg_clean = meg_preprocessing(raw_meg.copy())
    st.success("OK!")

fig_meg, ax = plt.subplots(figsize=(12,6))
contra_left = raw_meg[meg_labels==1].average('MEG_0111')
ipsi_right = raw_meg[meg_labels==1].average('MEG_0222')
contra_left.plot(axes=[ax], show=False, color='red', linewidth=2)
ipsi_right.plot(axes=[ax], show=False, color='blue', linewidth=2)
ax.set_title("Potentiel Action: Contra (Rouge) vs Ipsi (Bleu)")
st.pyplot(fig_meg)

if st.button(" Classificateur Latéral Action"):
    features_meg = extract_meg_action_features(raw_meg_clean, meg_labels)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(f_classif, k=4)),
        ('svm', SVC(kernel='rbf', probability=True))
    ])
    
    cv_scores = cross_val_score(pipeline, features_meg, meg_labels, cv=5)
    pipeline.fit(features_meg, meg_labels)
    
    st.metric("Cross-Val Accuracy", f"{cv_scores.mean():.1%}")
    
    cm_df = pd.DataFrame(confusion_matrix(meg_labels, pipeline.predict(features_meg)),
                        index=['Gauche', 'Droite'], columns=['Pred_G', 'Pred_D'])
    st.dataframe(cm_df)

st.markdown("""
**Potentiel d'Action MEG:**
- **M1 Contra[-600ms]**: Localisation spatiale
- **SMA bilatéral**: Coordination
- **Alpha asymétrie**: Suppression ipsi
""")
