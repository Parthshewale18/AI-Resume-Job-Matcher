import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class ResumeJobMatcher:
    """Main class for matching resumes to jobs"""
    
    def __init__(self, method='transformer'):
        self.method = method
        self.tfidf_vectorizer = None
        self.transformer_model = None
        self.job_vectors = None
        
        if method in ['transformer', 'hybrid']:
            print("Loading transformer model...")
            self.transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Transformer model loaded!")
    
    def train(self, jobs_df):
        """Train the model on job descriptions"""
        print(f"Training {self.method} model on jobs...")
        
        job_texts = jobs_df['cleaned_job'].tolist()
        
        if self.method == 'tfidf':
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            self.job_vectors = self.tfidf_vectorizer.fit_transform(job_texts)
            
        elif self.method == 'transformer':
            self.job_vectors = self.transformer_model.encode(
                job_texts, 
                show_progress_bar=True,
                batch_size=32
            )
            
        elif self.method == 'hybrid':
            # TF-IDF vectors
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            tfidf_vectors = self.tfidf_vectorizer.fit_transform(job_texts)
            
            # Transformer vectors
            trans_vectors = self.transformer_model.encode(
                job_texts,
                show_progress_bar=True,
                batch_size=32
            )
            
            self.job_vectors = {
                'tfidf': tfidf_vectors,
                'transformer': trans_vectors
            }
        
        print("Training complete!")
    
    def calculate_skill_match(self, resume_skills, job_skills):
        """Calculate skill match score"""
        if not job_skills:
            return 0.0
        
        resume_skills_set = set(resume_skills)
        job_skills_set = set(job_skills)
        
        matched_skills = resume_skills_set.intersection(job_skills_set)
        
        return len(matched_skills) / len(job_skills_set) if job_skills_set else 0.0
    
    def match_resume(self, resume_text, resume_skills, jobs_df, top_k=10):
        """Find top matching jobs for a single resume"""
        
        if self.method == 'tfidf':
            # Transform resume
            resume_vector = self.tfidf_vectorizer.transform([resume_text])
            similarities = cosine_similarity(resume_vector, self.job_vectors).flatten()
            
        elif self.method == 'transformer':
            # Encode resume
            resume_vector = self.transformer_model.encode([resume_text])
            # Calculate similarity
            similarities = cosine_similarity(resume_vector, self.job_vectors).flatten()
            
        elif self.method == 'hybrid':
            # TF-IDF similarity
            resume_tfidf = self.tfidf_vectorizer.transform([resume_text])
            tfidf_sim = cosine_similarity(resume_tfidf, self.job_vectors['tfidf']).flatten()
            
            # Transformer similarity
            resume_trans = self.transformer_model.encode([resume_text])
            trans_sim = cosine_similarity(resume_trans, self.job_vectors['transformer']).flatten()
            
            # Combine (40% TF-IDF, 60% Transformer)
            similarities = 0.4 * tfidf_sim + 0.6 * trans_sim
        
        # Calculate skill scores
        skill_scores = jobs_df['required_skills'].apply(
            lambda x: self.calculate_skill_match(resume_skills, x)
        ).values
        
        # Final score (70% semantic, 30% skills)
        final_scores = 0.7 * similarities + 0.3 * skill_scores
        
        # Get top K jobs
        top_indices = np.argsort(final_scores)[-top_k:][::-1]
        
        # Create results dataframe
        results = jobs_df.iloc[top_indices].copy()
        results['match_score'] = final_scores[top_indices]
        results['semantic_score'] = similarities[top_indices]
        results['skill_score'] = skill_scores[top_indices]
        
        return results