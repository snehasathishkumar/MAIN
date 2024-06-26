import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

from compound_to_simple import compound_to_simple

def summarizer(text):
    
    def preprocess_text(text):
        # Tokenize sentences
        sentences = sent_tokenize(text)
        # Tokenize words and remove stopwords
        stop_words = set(stopwords.words('english'))
        processed_sentences = [' '.join([word for word in word_tokenize(sentence.lower()) if word.isalnum() and word not in stop_words])
                            for sentence in sentences]
        return sentences, processed_sentences


    def compute_cosine_similarity(sentences, processed_sentences):
        # Create TF-IDF vectorizer and transform sentences
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        # Compute cosine similarity matrix
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return cosine_sim_matrix


    def extractive_summarization(text, top_n=3):
        # Preprocess text
        sentences, processed_sentences = preprocess_text(text)
        # Compute cosine similarity matrix
        cosine_sim_matrix = compute_cosine_similarity(sentences, processed_sentences)
        # Rank sentences based on their average cosine similarity to other sentences
        sentence_scores = np.sum(cosine_sim_matrix, axis=1)
        # Get top N sentences
        top_sentence_indices = np.argsort(sentence_scores)[-top_n:]
        top_sentences = [sentences[index] for index in sorted(top_sentence_indices)]
        # Combine top sentences to form the summary
        summary = ' '.join(top_sentences)
        return summary


    # Example usage
    # text = compound_to_simple()

    summary = extractive_summarization(compound_to_simple(text), top_n=30)
    # print("Summary:")
    # print(summary)
    return summary