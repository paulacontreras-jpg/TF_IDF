import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

st.set_page_config(page_title="🚀 TF-IDF Espacial", page_icon="🛰️", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 20% 20%, #0f2027 0%, #203a43 45%, #0b0c10 100%);
        color: #e6f1ff;
    }
    h1, h2, h3 {
        color: #66fcf1 !important;
        text-shadow: 0 0 8px rgba(102,252,241,0.5);
    }
    .stTextArea textarea, .stTextInput input {
        background-color: #1f2833 !important;
        color: #e6f1ff !important;
        border: 1px solid #45a29e !important;
    }
    div.stButton > button {
        background-color: #45a29e;
        color: #0b0c10;
        border-radius: 8px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #66fcf1;
        color: #0b0c10;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚀 Demo de TF-IDF con Misiones y Astronautas")
st.write("""
Cada línea se trata como una **nave** (puede ser una frase, un párrafo o un texto más largo).  
⚠️ Los registros y las consultas deben estar en **inglés**, ya que el sistema está configurado para ese idioma.  
La misión aplica normalización y *stemming* para que palabras como *orbiting* y *orbit* se consideren equivalentes.
""")

# Ejemplo inicial en inglés, ahora con temática espacial
text_input = st.text_area(
    "Escribe tus reportes (uno por línea, en inglés):",
    "The rocket launches loudly.\nThe comet glows at night.\nThe rocket and the comet orbit together."
)

question = st.text_input("Escribe una pregunta (en inglés):", "Who is orbiting?")

# Inicializar stemmer para inglés
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    # Pasar a minúsculas
    text = text.lower()
    # Eliminar caracteres no alfabéticos
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Tokenizar (palabras con longitud > 1)
    tokens = [t for t in text.split() if len(t) > 1]
    # Aplicar stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems

if st.button("🛰️ Calcular TF-IDF y buscar señal"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos un reporte.")
    else:
        # Vectorizador con stemming
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None
        )
        # Ajustar con documentos
        X = vectorizer.fit_transform(documents)

        # Mostrar matriz TF-IDF
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Nave {i+1}" for i in range(len(documents))]
        )
        st.write("### 🌌 Matriz TF-IDF (rastros)")
        st.dataframe(df_tfidf.round(3))

        # Vector de la pregunta
        question_vec = vectorizer.transform([question])

        # Similitud coseno
        similarities = cosine_similarity(question_vec, X).flatten()

        # Documento más parecido
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.write("### 📡 Transmisión y respuesta")
        st.write(f"**Tu transmisión:** {question}")
        st.write(f"**Reporte más relevante (Nave {best_idx+1}):** {best_doc}")
        st.write(f"**Puntaje de similitud:** {best_score:.3f}")

        # Mostrar todas las similitudes
        sim_df = pd.DataFrame({
            "Nave": [f"Nave {i+1}" for i in range(len(documents))],
            "Mensaje": documents,
            "Similitud": similarities
        })
        st.write("### ⭐ Puntajes de similitud (ordenados)")
        st.dataframe(sim_df.sort_values("Similitud", ascending=False))

        # Mostrar coincidencias de stems
        vocab = vectorizer.get_feature_names_out()
        q_stems = tokenize_and_stem(question)
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]
        st.write("### 🔭 Stems de la transmisión presentes en el reporte elegido:", matched)
