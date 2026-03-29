# shared_utils.py
import spacy

# Initialize ONCE and share across the app to save Render RAM
nlp = spacy.load("en_core_web_sm")
