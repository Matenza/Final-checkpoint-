import speech_recognition as sr
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import evaluate  # Remplacement de load_metric par evaluate
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Téléchargement des ressources nécessaires
nltk.download("stopwords")
nltk.download("punkt")

# Fonction pour obtenir du texte à partir de la commande vocale
def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Veuillez parler...")
        audio = recognizer.listen(source)

        try:
            # Reconnaissance vocale
            texte = recognizer.recognize_google(audio, language='fr-FR')
            st.write(f"Vous avez dit : {texte}")
            return texte
        except sr.UnknownValueError:
            st.error("Je n'ai pas compris ce que vous avez dit.")
            return None
        except sr.RequestError as e:
            st.error(f"Erreur de service : {e}")
            return None

# Fonction pour calculer les scores ROUGE et BLEU
def evaluate_summary(generated_summary, reference_summary):
    rouge_metric = evaluate.load("rouge")  # Remplacement de load_metric par evaluate.load
    bleu_metric = evaluate.load("bleu")    # Remplacement de load_metric par evaluate.load

    # Calcul des scores ROUGE
    rouge_score = rouge_metric.compute(predictions=[generated_summary], references=[reference_summary])
    # Calcul des scores BLEU
    bleu_score = bleu_metric.compute(predictions=[generated_summary.split()], references=[[reference_summary.split()]])

    return rouge_score, bleu_score

# Fonction pour afficher les scores
def display_scores(rouge_score, bleu_score):
    st.subheader("Évaluation des performances du résumé :")
    st.write("Scores ROUGE :", rouge_score)
    st.write("Scores BLEU :", bleu_score)

    # Visualisation des scores
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    rouge_1 = rouge_score['rouge1'].mid.fmeasure
    rouge_2 = rouge_score['rouge2'].mid.fmeasure
    rouge_l = rouge_score['rougeL'].mid.fmeasure

    scores = [rouge_1, rouge_2, rouge_l]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=metrics, y=scores)
    plt.title('Scores ROUGE')
    plt.ylim(0, 1)
    plt.ylabel('F-mesure')
    st.pyplot(plt)

# Interface utilisateur avec Streamlit
st.title("Résumé de Texte avec Commande Vocale")
st.write("""
    Cette application utilise des modèles de traitement de langage naturel pour générer un résumé d'un texte donné.
    Vous pouvez entrer le texte soit par écrit, soit par commande vocale.
    L'application évalue également la qualité du résumé généré à l'aide des métriques ROUGE et BLEU.
""")

# Option pour l'entrée vocale
choix = st.selectbox("Souhaitez-vous entrer le texte par la voix ?", ["Non", "Oui"])
if choix == "Oui":
    texte = voice_input()
else:
    texte = st.text_area("Veuillez entrer le texte à résumer :", height=300)

# Vérification si le texte est vide
if texte is not None and texte.strip() != "":
    # Initialisation des stopwords en français
    stopwords_fr = set(stopwords.words("french"))

    # Tokenisation des mots
    words = word_tokenize(texte)

    # Création d'une table de fréquence pour chaque mot non-stopword
    freqtable = dict()
    for word in words:
        word = word.lower()
        if word in stopwords_fr:
            continue
        if word in freqtable:
            freqtable[word] += 1
        else:
            freqtable[word] = 1

    # Affichage des fréquences des mots
    st.write("Fréquences des mots :", freqtable)

    # Tokenisation des phrases
    sentences = sent_tokenize(texte)

    # Calcul de la valeur de chaque phrase en fonction des fréquences des mots
    def getsentenceValue():
        sentenceValue = dict()
        for sentence in sentences:
            for word, freq in freqtable.items():
                if word in sentence.lower():
                    if sentence in sentenceValue:
                        sentenceValue[sentence] += freq
                    else:
                        sentenceValue[sentence] = freq
        return sentenceValue

    # Calcul des valeurs des phrases
    sentenceValue = getsentenceValue()
    st.write("Valeurs des phrases :", sentenceValue)

    # Calcul de la moyenne des valeurs des phrases
    def getsumValues():
        sumValues = 0
        for sentence in sentenceValue:
            sumValues += sentenceValue[sentence]
        average = int(sumValues / len(sentenceValue))
        return average

    average = getsumValues()
    st.write(f"Moyenne des valeurs des phrases : {average}")

    # Génération du résumé basé sur les valeurs
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence

    # Affichage du résumé
    st.subheader("Résumé simple basé sur la fréquence des mots :")
    st.write(summary)

    # Sélection du modèle pour le résumé
    model_choice = st.selectbox("Choisissez le modèle à utiliser pour le résumé :", ["BART", "T5"])

    # Chargement du modèle choisi
    if model_choice == "BART":
        model_name = "facebook/bart-large-cnn"
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)
    else:
        model_name = "t5-base"
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Générer le résumé avec le modèle choisi
    if st.button("Générer le résumé avec le modèle choisi"):
        with st.spinner('Génération du résumé en cours...'):
            inputs = tokenizer(texte, max_length=1024, return_tensors="pt", truncation=True)

            # Générer le résumé
            summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary_model = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Affichage du résumé généré par le modèle
            st.subheader("Résumé généré par le modèle :")
            st.write(summary_model)

            # Évaluation des performances du modèle
            reference_summary = st.text_area("Entrez le résumé de référence pour l'évaluation :", height=100)
            
            if st.button("Évaluer le résumé"):
                if reference_summary:
                    rouge_score, bleu_score = evaluate_summary(summary_model, reference_summary)
                    display_scores(rouge_score, bleu_score)
                else:
                    st.error("Veuillez entrer un résumé de référence pour l'évaluation.")
else:
    st.warning("Veuillez fournir un texte à résumer.")
