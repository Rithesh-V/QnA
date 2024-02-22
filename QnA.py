import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import joblib

nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)

    text = text.lower()

    words = word_tokenize(text)

    english_vocab = set(word.lower() for word in nltk.corpus.words.words())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    text = ' '.join(words)
    return text

def divide_documents(text):
    paragraphs = text.split('\n\n')
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(sent_tokenize(paragraph))
    return sentences

def generate_summaries(sentences):
    summaries = []
    for sentence in sentences:
        summary = sentence
        summaries.append(summary)
    return summaries

def readable_summary(text):
  model_name = "facebook/bart-large-cnn"
  tokenizer = BartTokenizer.from_pretrained(model_name)
  model = BartForConditionalGeneration.from_pretrained(model_name)

  input_ids = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)

# Generate summary
  summary_ids = model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)

# Decode and print the summary
  summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  print(summary)
  return summary
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased-distilled-squad")

# Save the entire code, including the model and utility functions, into a pickle file
joblib.dump({
    'qa_pipeline': qa_pipeline,
    'preprocess_text': preprocess_text,
    'divide_documents': divide_documents,
    'generate_summaries': generate_summaries,
    'readable_summary': readable_summary
}, 'QnA.pkl')
