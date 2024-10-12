import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import torch


model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)


st.title("Résumé d'articles de presse avec BART")

st.write("""
    Cette application utilise le modèle **BART** (Bidirectional and Auto-Regressive Transformer) pour générer un résumé 
    d'un article de presse. Il vous suffit de copier et coller l'article, et un résumé sera généré automatiquement.
""")

article = st.text_area("Entrez l'article de presse à résumer", height=300)

if st.button("Générer le résumé"):
    if article:
        with st.spinner('Génération du résumé en cours...'):
            # Tokenization 
            inputs = tokenizer([article], max_length=1024, return_tensors="pt", truncation=True)

            # Générer le résumé
            summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

            # Décoder le résumé
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Afficher le résumé
            st.subheader("Résumé généré")
            st.write(summary)
    else:
        st.error("Veuillez entrer un article de presse.")
