from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

import nltk
nltk.download("stopwords")
nltk.download('punkt')

texte = """La naumachie, du grec ancien ναυμαχία / naumachía, littéralement « combat naval », en latin navalia proelia) est dans le monde
romain un spectacle représentant une bataille navale, ou le bassin, ou plus largement l'édifice, dans lequel un tel spectacle se tenait.
Les moyens considérables à mettre en œuvre pour une naumachie, l'aménagement d'un plan d'eau et de places pour les spectateurs,
la mobilisation de flottes, l'engagement de nombreux combattants, en font un spectacle d'exception que seuls les empereurs pouvaient organiser.
Les connaissances actuelles sur les naumachies romaines reposent sur peu d'éléments, essentiellement littéraires, car les édifices n'ont pratiquement pas laissé de trace archéologique. 
Les premières naumachies, celles de Jules César, d'Auguste et de Claude — en tout trois seulement en un siècle — restent dans les annales.
Les suivantes, un peu plus fréquentes mais moins remarquées par les auteurs antiques, se déroulent dans des espaces plus restreints et guère navigables, dans l'arène d'amphithéâtres, inondée pour l'occasion.
Trajan, parmi ses constructions fastueuses, inaugure le dernier bassin spécialement destiné aux naumachies, identifié à la Naumachia Vaticana répertoriée dans les documents du Bas-Empire.
Les dernières naumachies datent du IIIe siècle, pour autant que les sources soient fiables. Dans les Temps modernes, quelques représentations ont repris la dénomination de naumachie, mais sans le réalisme des combats antiques."""

stopwords = set(stopwords.words("french"))
print(stopwords)

words = word_tokenize(texte)
print(words)

freqtable = dict()

freqtable = dict()
for word in words:
    word = word.lower()
    if word in stopwords:
       continue
    if word in freqtable:
        freqtable [word] += 1
    else:
        freqtable[word] = 1
print(freqtable)

sentences = sent_tokenize(texte)
print(sentences)

sentences[0]

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
    print(sentenceValue)

sentenceValeu = getsentenceValue()
print(sentenceValeu)

def getsumValues():
    sumValues = 0
    for sentence in sentenceValeu:
        sumValues += sentenceValeu[sentence]
    average = int(sumValues / len(sentenceValeu))
    return average
average = getsumValues()
print(average)

summary = ''
for sentence in sentences:
    if (sentence in sentenceValeu) and (sentenceValeu[sentence] > (1.2 * average)):
        summary +=  " "  + sentence
print(summary)

