import pandas as pd
import spacy
from rapidfuzz import fuzz
import os

class DataCleaner:
    def __init__(self, master_data_path):
        self.nlp = spacy.load("en_core_web_md")
        self.master_data_path = master_data_path
        # Load existing data to check for duplicates
        if os.path.exists(master_data_path):
            self.df_master = pd.read_csv(master_data_path)
        else:
            self.df_master = pd.DataFrame(columns=['Job Title', 'Industry', 'Key Skills'])

    def classify_role(self, messy_title):
        # """Uses spaCy vectors to find the most 'standard' role name."""
        existing_roles = self.df_master['Job Title'].unique().tolist()
        if not existing_roles:
            return messy_title.strip().title()

        doc1 = self.nlp(messy_title)
        best_match = messy_title
        highest_score = 0

        for role in existing_roles:
            score = doc1.similarity(self.nlp(role))
            if score > highest_score:
                highest_score = score
                best_match = role
        
        # If it's a 70% match, use the existing standard role name
        return best_match if highest_score > 0.7 else messy_title.strip().title()

    def normalize_skills(self, skills_string):
        """Clean and deduplicate skills (e.g., 'React.js, react' -> 'React')"""
        skills = [s.strip().title() for s in str(skills_string).split(',')]
        return ", ".join(sorted(list(set(skills))))

    def is_duplicate(self, new_job_title, new_skills):
        """Check if the job already exists using Fuzzy Matching."""
        for _, row in self.df_master.iterrows():
            title_sim = fuzz.ratio(new_job_title.lower(), row['Job Title'].lower())
            # If title is 95% same, check skills
            if title_sim > 95:
                return True
        return False

    def clean_and_add(self, scraped_data_list):
        """
        Input: List of dicts [{'title': '...', 'industry': '...', 'skills': '...'}]
        """
        new_entries = []
        
        for item in scraped_data_list:
            # 1. Basic Cleaning
            clean_title = item['title'].strip().title()
            clean_industry = item['industry'].strip().title()
            clean_skills = self.normalize_skills(item['skills'])

            # 2. Deduplication Check
            if not self.is_duplicate(clean_title, clean_skills):
                new_entries.append({
                    'Job Title': clean_title,
                    'Industry': clean_industry,
                    'Key Skills': clean_skills
                })
        
        if new_entries:
            new_df = pd.DataFrame(new_entries)
            # Append to master CSV
            self.df_master = pd.concat([self.df_master, new_df], ignore_index=True)
            self.df_master.to_csv(self.master_data_path, index=False)
            return f"✅ Added {len(new_entries)} new unique jobs."
        else:
            return "ℹ️ No new unique data found."