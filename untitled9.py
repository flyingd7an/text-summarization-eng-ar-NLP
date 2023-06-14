# -*- coding: utf-8 -*-
"""Untitled9.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16SpD3NffqzraeGgYQFMMUpbj6MqedA2g
"""

!pip install nltk
!pip install sumy
!pip install pyarabic

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('arabic')

def text_summarizer(text, language):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Remove stopwords and stem the words
    if language == 'english':
        stopwords_list = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')
    elif language == 'arabic':
        stopwords_list = set(stopwords.words('arabic'))
        stemmer = SnowballStemmer('arabic')
    else:
        return "Unsupported language. Please choose either 'english' or 'arabic'."

    summarized_text = ""

    # Process each sentence
    for sentence in sentences:
        # Tokenize the sentence into words
        words = word_tokenize(sentence)

        # Remove stopwords and stem the words
        filtered_words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stopwords_list]

        # Join the words back into a sentence
        filtered_sentence = ' '.join(filtered_words)

        # Add the filtered sentence to the summarized text
        summarized_text += filtered_sentence + " "

    # Use the LexRank algorithm from sumy to generate the final summary
    parser = PlaintextParser.from_string(summarized_text, Tokenizer(language))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count=15)  # You can adjust the number of sentences in the summary

    # Join the summary sentences into a single string
    summary_text = ' '.join([str(sentence) for sentence in summary])

    return summary_text

# Example usage
english_text = "The boy went to the shop and bought a red ball from target."
arabic_text = "ذهب الولد الى المدرسة و لعب بالكرة مع أصحابه"

english_summary = text_summarizer(english_text, 'english')
arabic_summary = text_summarizer(arabic_text, 'arabic')

print("English Summary:")
print(english_summary)

print("\nArabic Summary:")
print(arabic_summary)