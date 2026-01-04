import streamlit as st
import pandas as pd
import sys
import os
import PyPDF2
import docx
import joblib

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import TextPreprocessor, SkillExtractor
from utils import plot_match_scores, create_skill_comparison

# Page configuration
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def extract_text_from_txt(txt_file):
    """Extract text from TXT file"""
    try:
        return txt_file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return ""

def load_trained_model():
    """Load pre-trained model and data"""
    try:
        matcher = joblib.load("models/matcher.pkl")
        jobs_df = joblib.load("models/jobs_df.pkl")
        preprocessor = joblib.load("models/preprocessor.pkl")
        skill_extractor = joblib.load("models/skill_extractor.pkl")
        return matcher, jobs_df, preprocessor, skill_extractor, True
    except Exception as e:
        return None, None, None, None, False

# Title
st.markdown('<h1 class="main-header">📄 AI-Powered Job Matcher</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your resume and discover the perfect job matches!</p>', unsafe_allow_html=True)

# Load pre-trained model
with st.spinner("🔄 Loading AI model..."):
    matcher, jobs_df, preprocessor, skill_extractor, model_loaded = load_trained_model()

if not model_loaded:
    st.error("⚠️ Model not found! Please train the model first.")
    
    st.markdown("""
    ## 🚀 First Time Setup
    
    You need to train the model once before using the application.
    
    ### Steps:
    
    1. **Download the job dataset** from [Kaggle](https://www.kaggle.com/datasets/PromptCloudHQ/us-jobs-on-monstercom)
    
    2. **Place the file** in the `data/` folder:
       ```
       data/jobs.csv
       ```
    
    3. **Train the model** by running:
       ```bash
       python train_model.py
       ```
       
       Or with options:
       ```bash
       python train_model.py --job-csv data/jobs.csv --method transformer
       ```
    
    4. **Restart this app**:
       ```bash
       streamlit run app.py
       ```
    
    ### Model Options:
    
    | Method | Speed | Accuracy | Best For |
    |--------|-------|----------|----------|
    | `transformer` | Slow | Highest ⭐ | Best results (Recommended) |
    | `hybrid` | Medium | High | Balanced performance |
    | `tfidf` | Fast | Good | Quick results |
    
    ### Training Time:
    - TF-IDF: ~1 minute
    - Transformer: ~3-5 minutes
    - Hybrid: ~4-6 minutes
    
    **Note**: Training is done only ONCE. After that, the app loads instantly!
    """)
    st.stop()

# Model loaded successfully
st.success(f"✅ AI Model Loaded | {len(jobs_df)} jobs available")

# Sidebar with info
with st.sidebar:
    st.header("📊 System Status")
    st.metric("Jobs in Database", len(jobs_df))
    st.metric("Model Type", matcher.method.upper())
    
    st.markdown("---")
    
    st.subheader("⚙️ Settings")
    top_k = st.slider("Number of Job Matches", 5, 20, 10)
    
    st.markdown("---")
    
    st.subheader("📖 How to Use")
    st.markdown("""
    1. Upload your resume
    2. Review extracted skills
    3. Click "Find Matching Jobs"
    4. Explore results
    5. Download recommendations
    """)
    
    st.markdown("---")
    
    if st.button("🔄 Retrain Model"):
        st.info("Run: `python train_model.py`")

# Main Content
st.markdown("---")

# Resume Upload Section
st.header("📤 Upload Your Resume")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_resume = st.file_uploader(
        "Choose your resume file (PDF, DOCX, or TXT)",
        type=['pdf', 'docx', 'txt'],
        help="Upload your resume to find matching jobs"
    )

if uploaded_resume is not None:
    # Extract text based on file type
    file_extension = uploaded_resume.name.split('.')[-1].lower()
    
    with st.spinner("📖 Reading your resume..."):
        if file_extension == 'pdf':
            resume_text = extract_text_from_pdf(uploaded_resume)
        elif file_extension == 'docx':
            resume_text = extract_text_from_docx(uploaded_resume)
        elif file_extension == 'txt':
            resume_text = extract_text_from_txt(uploaded_resume)
        else:
            st.error("Unsupported file format")
            resume_text = ""
    
    if resume_text and len(resume_text.strip()) > 50:
        
        with col2:
            st.metric("📝 Words", len(resume_text.split()))
            st.metric("📏 Characters", len(resume_text))
        
        # Process resume
        cleaned_resume = preprocessor.preprocess(resume_text)
        extracted_skills = skill_extractor.extract_skills(resume_text)
        
        st.success("✅ Resume uploaded successfully!")
        
        # Display Resume Analysis
        st.markdown("---")
        st.header("🔍 Resume Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("💼 Detected Skills")
            
            if extracted_skills:
                # Categorize skills
                skill_db = skill_extractor.skills_database
                categorized = {}
                
                for skill in extracted_skills:
                    for category, skills_list in skill_db.items():
                        if skill in skills_list:
                            if category not in categorized:
                                categorized[category] = []
                            categorized[category].append(skill)
                            break
                
                for category, skills in categorized.items():
                    category_name = category.replace('_', ' ').title()
                    with st.expander(f"**{category_name}** ({len(skills)} skills)", expanded=True):
                        st.write(", ".join(skills))
                
                st.info(f"**Total Skills Found:** {len(extracted_skills)}")
            else:
                st.warning("⚠️ No specific technical skills detected. Consider adding more details to your resume.")
        
        with col2:
            st.subheader("📄 Resume Preview")
            preview_text = resume_text[:500] + "..." if len(resume_text) > 500 else resume_text
            st.text_area("First 500 characters", preview_text, height=300, disabled=True)
        
        # Find Jobs Button
        st.markdown("---")
        
        if st.button("🎯 Find Matching Jobs", type="primary", use_container_width=True):
            
            with st.spinner("🤖 AI is analyzing and finding best matches..."):
                # Get matches
                results = matcher.match_resume(
                    cleaned_resume,
                    extracted_skills,
                    jobs_df,
                    top_k
                )
                
                st.session_state.results = results
                st.session_state.resume_skills = extracted_skills
            
            st.success(f"✅ Found {len(results)} matching jobs!")
        
        # Display Results
        if 'results' in st.session_state:
            results = st.session_state.results
            resume_skills = st.session_state.resume_skills
            
            st.markdown("---")
            st.header(f"🎯 Top {len(results)} Job Matches")
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                best_match = results.iloc[0]['match_score']
                st.metric("🏆 Best Match", f"{best_match:.1%}")
            
            with col2:
                avg_match = results['match_score'].mean()
                st.metric("📊 Average Match", f"{avg_match:.1%}")
            
            with col3:
                high_matches = len(results[results['match_score'] >= 0.7])
                st.metric("⭐ High Matches", high_matches)
            
            with col4:
                st.metric("📋 Total Matches", len(results))
            
            # Chart
            st.plotly_chart(plot_match_scores(results), use_container_width=True, key="main_scores_chart")
            
            st.markdown("---")
            
            # Display each job
            for idx in range(len(results)):
                row = results.iloc[idx]
                match_score = row['match_score']
                
                # Color coding
                if match_score >= 0.8:
                    emoji = "🟢"
                    label = "Excellent Match"
                elif match_score >= 0.6:
                    emoji = "🟡"
                    label = "Good Match"
                else:
                    emoji = "🟠"
                    label = "Moderate Match"
                
                with st.expander(
                    f"{emoji} **{row['job_title']}** at **{row.get('company', 'N/A')}** • "
                    f"{label} ({match_score:.1%})",
                    expanded=(idx == 0)
                ):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**🏢 Company:** {row.get('company', 'N/A')}")
                        st.markdown(f"**📍 Location:** {row.get('location', 'N/A')}")
                        
                        st.markdown("**📊 Score Breakdown:**")
                        st.progress(float(row['match_score']), text=f"Overall: {row['match_score']:.1%}")
                        st.progress(float(row['semantic_score']), text=f"Content Match: {row['semantic_score']:.1%}")
                        st.progress(float(row['skill_score']), text=f"Skills Match: {row['skill_score']:.1%}")
                        
                        if 'job_description' in row and pd.notna(row['job_description']):
                            st.markdown("**📝 Job Description:**")
                            desc = str(row['job_description'])[:400]
                            st.write(desc + "..." if len(str(row['job_description'])) > 400 else desc)
                    
                    with col2:
                        job_idx = results.index[idx]
                        job_skills = jobs_df.loc[job_idx, 'required_skills']
                        
                        matched = set(resume_skills).intersection(set(job_skills))
                        missing = set(job_skills) - set(resume_skills)
                        
                        st.markdown("**✅ Your Matching Skills:**")
                        if matched:
                            st.success(", ".join(list(matched)[:8]))
                        else:
                            st.info("Skills match based on context")
                        
                        st.markdown("**🎯 Skills to Develop:**")
                        if missing:
                            st.warning(", ".join(list(missing)[:8]))
                        else:
                            st.success("You have all key skills!")
                        
                        # Chart
                        fig = create_skill_comparison(resume_skills, job_skills)
                        st.plotly_chart(fig, use_container_width=True, key=f"skill_chart_{idx}")
            
            # Export
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                csv = results[['job_title', 'company', 'location', 'match_score', 
                              'semantic_score', 'skill_score']].to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "job_matches.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    elif resume_text:
        st.error("❌ Resume text is too short. Please upload a complete resume.")

else:
    # Instructions
    st.info("👆 **Upload your resume to get started**")
    
    st.markdown("""
    ## 💡 Tips for Best Results
    
    ### Resume Content Tips:
    - ✅ Include specific technical skills (e.g., Python, React, AWS)
    - ✅ Mention frameworks and tools you've used
    - ✅ Add relevant projects and achievements
    - ✅ Use industry-standard terminology
    - ✅ Keep your resume detailed and up-to-date
    
    ### Supported Skills (100+):
    
    | Category | Examples |
    |----------|----------|
    | **Programming** | Python, Java, JavaScript, C++, Go, Rust |
    | **Web** | React, Angular, Node.js, Django, Flask |
    | **Database** | MySQL, MongoDB, PostgreSQL, Redis |
    | **Cloud** | AWS, Azure, GCP, Docker, Kubernetes |
    | **Data Science** | ML, TensorFlow, Pandas, Scikit-learn |
    | **Soft Skills** | Leadership, Communication, Agile |
    
    ### Match Score Guide:
    
    - 🟢 **80%+** - Excellent Match (Highly Recommended)
    - 🟡 **60-80%** - Good Match (Worth Applying)
    - 🟠 **Below 60%** - Moderate Match (Consider Upskilling)
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "Built with ❤️ using Streamlit, Transformers & NLP | "
    "© 2024 AI Resume Matcher"
    "</div>",
    unsafe_allow_html=True
)