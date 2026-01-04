import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data with SSL fix
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
required_data = ['punkt', 'stopwords', 'wordnet', 'punkt_tab', 'averaged_perceptron_tagger']
for data in required_data:
    try:
        nltk.data.find(f'tokenizers/{data}')
    except LookupError:
        try:
            nltk.download(data, quiet=True)
        except:
            pass

class TextPreprocessor:
    """Text preprocessing class for cleaning and normalizing text"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        if not text:
            return ''
        
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                  if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        cleaned = self.clean_text(text)
        processed = self.tokenize_and_lemmatize(cleaned)
        return processed


class SkillExtractor:
    """Extract skills from text using a predefined skill database"""
    
    def __init__(self):
        self.skills_database = {
            'programming_languages': [
                'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 
                'swift', 'kotlin', 'go', 'rust', 'typescript', 'r', 'matlab',
                'scala', 'perl', 'shell', 'bash', 'c', 'objective-c'
            ],
            'web_technologies': [
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'nodejs', 'django',
                'flask', 'spring', 'asp.net', 'express', 'jquery', 'bootstrap',
                'tailwind', 'webpack', 'rest', 'restful', 'graphql', 'api', 'ajax',
                'json', 'xml', 'soap', 'microservices'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis',
                'cassandra', 'dynamodb', 'firebase', 'sqlite', 'mariadb',
                'elasticsearch', 'neo4j', 'couchdb', 'database'
            ],
            'cloud_devops': [
                'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins',
                'terraform', 'ansible', 'git', 'github', 'gitlab', 'ci/cd', 'cicd',
                'linux', 'unix', 'nginx', 'apache', 'devops', 'cloud'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'nlp', 'computer vision',
                'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas',
                'numpy', 'data analysis', 'statistics', 'tableau', 'power bi',
                'spark', 'hadoop', 'etl', 'data mining', 'artificial intelligence',
                'ai', 'ml', 'neural networks', 'data visualization'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'project management', 'agile', 'scrum', 'analytical', 'creative',
                'management', 'collaboration'
            ],
            'other_skills': [
                'testing', 'debugging', 'optimization', 'security', 'authentication',
                'excel', 'powerpoint', 'word', 'office', 'jira', 'confluence'
            ]
        }
        
        self.all_skills = []
        for category in self.skills_database.values():
            self.all_skills.extend(category)
    
    def extract_skills(self, text):
        """Extract skills from text"""
        if pd.isna(text) or text == '':
            return []
        
        text = str(text).lower()
        found_skills = []
        
        for skill in self.all_skills:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text):
                found_skills.append(skill)
        
        return list(set(found_skills))


def load_and_preprocess_jobs(job_path):
    """Load and preprocess job dataset"""
    jobs_df = pd.read_csv(job_path)
    
    preprocessor = TextPreprocessor()
    skill_extractor = SkillExtractor()
    
    print("Preprocessing jobs...")
    
    # Handle your actual dataset columns
    # Combine job_title and job_description
    jobs_df['job_title'] = jobs_df['job_title'].fillna('')
    jobs_df['job_description'] = jobs_df['job_description'].fillna('')
    jobs_df['organization'] = jobs_df['organization'].fillna('N/A')
    jobs_df['location'] = jobs_df['location'].fillna('N/A')
    
    # Create combined text for matching
    jobs_df['full_description'] = jobs_df['job_title'] + ' ' + jobs_df['job_description']
    jobs_df['cleaned_job'] = jobs_df['full_description'].apply(preprocessor.preprocess)
    jobs_df['required_skills'] = jobs_df['full_description'].apply(skill_extractor.extract_skills)
    
    # Rename organization to company for consistency
    jobs_df['company'] = jobs_df['organization']
    
    print(f"Loaded {len(jobs_df)} jobs")
    
    return jobs_df, preprocessor, skill_extractor