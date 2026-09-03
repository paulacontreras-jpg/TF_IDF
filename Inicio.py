import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import SnowballStemmer
import streamlit as st

# Configuración de página con temática oscura/espacial
st.set_page_config(page_title="Radar Espacial TF-IDF", page_icon="🚀")

st.title("🛸 Radar de Transmisiones Espaciales TF-IDF")

st.write("""
Cada registro se procesa como un **bitácora estelar** (puede ser un mensaje, un reporte o una señal lejana).  
⚠️ Las bitácoras y la consulta deben registrarse en **inglés**, ya que los sensores analizan dicho idioma.  

El sistema aplica normalización y *stemming* estelar para que términos como *orbiting* y *orbit* coincidan perfectamente.
""")

# Ejemplo inicial en temática espacial
text_input = st.text_area(
    "Ingresa las bitácoras espaciales (una por línea, en inglés):",
    "The satellite orbits Earth.\nThe rocket travels to Mars.\nThe satellite and rocket send data.",
)

question = st.text_input("Ingresa la consulta de rastreo (en inglés):", "Which satellite is orbiting?")

# Inicializar stemmer para inglés
stemmer = SnowballStemmer("english")


def tokenize_and_stem(text: str):
    # Pasar a minúsculas
    text = text.lower()
    # Eliminar caracteres no alfabéticos
    text = re.sub(r"[^a-z\s]", " ", text)
    # Tokenizar (palabras con longitud > 1)
    tokens = [t for t in text.split() if len(t) > 1]
    # Aplicar stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems


if st.button("🚀 Escanear espacio y calcular TF-IDF"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(documents) < 1:
        st.warning("⚠️ Detectada ausencia de señal. Ingresa al menos una bitácora.")
    else:
        # Vectorizador con stemming
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None,
        )

        # Ajustar con documentos
        X = vectorizer.fit_transform(documents)

        # Mostrar matriz TF-IDF
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Bitácora {i+1}" for i in range(len(documents))],
        )

        st.write("### 🌌 Matriz Estelar TF-IDF (stems)")
        st.dataframe(df_tfidf.round(3))

        # Vector de la pregunta
        question_vec = vectorizer.transform([question])

        # Similitud coseno
        similarities = cosine_similarity(question_vec, X).flatten()

        # Documento más parecido
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.write("### 🛰️ Consulta e Intercepción")
        st.write(f"**Tu consulta estelar:** {question}")
        st.write(f"**Bitácora de mayor resonancia (Bitácora {best_idx+1}):** {best_doc}")
        st.write(f"**Nivel de coincidencia espectral:** {best_score:.3f}")

        # Mostrar todas las similitudes
        sim_df = pd.DataFrame({
            "Bitácora": [f"Bitácora {i+1}" for i in range(len(documents))],
            "Señal": documents,
            "Resonancia": similarities,
        })
        st.write("### 📊 Espectro de Resonancia (ordenado)")
        st.dataframe(sim_df.sort_values("Resonancia", ascending=False))

        # Mostrar coincidencias de stems
        vocab = vectorizer.get_feature_names_out()
        q_stems = tokenize_and_stem(question)
        matched = [
            s
            for s in q_stems
            if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0
        ]
        st.write(
            "### Stems de la consulta detectados en la bitácora elegida:",
            matched,
        )
