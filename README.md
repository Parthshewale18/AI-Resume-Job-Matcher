# AI-Resume-Job-Matcher
# 📄 AI-Powered Resume Job Matcher

An intelligent system that matches individual resumes to job postings using AI and NLP.

## 🎯 Features

- ✅ Upload single resume (PDF, DOCX, TXT)
- ✅ AI-powered semantic matching
- ✅ Automatic skill extraction (100+ skills)
- ✅ Skill gap analysis
- ✅ Visual match scores
- ✅ Export results to CSV
- ✅ Three matching algorithms (TF-IDF, Transformer, Hybrid)

## 📁 Project Structure

```
resume_job_matcher/
│
├── data/                   # Data directory
│   └── jobs.csv           # Job dataset (download from Kaggle)
│
├── models/                 # Trained models (auto-created)
│   ├── matcher.pkl        # Trained matcher model
│   ├── jobs_df.pkl        # Processed jobs dataframe
│   ├── preprocessor.pkl   # Text preprocessor
│   └── skill_extractor.pkl # Skill extractor
│
├── src/                    # Source code
│   ├── preprocessing.py   # Text processing & skill extraction
│   ├── model.py          # AI matching models
│   └── utils.py          # Visualization utilities
│
├── app.py                 # Main Streamlit application
├── train_model.py        # One-time training script
├── setup.py              # Setup script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Installation & Setup

### Step 1: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### Step 2: Run Setup Script

```bash
# Download NLTK data and create directories
python setup.py
```

### Step 3: Download Job Dataset

Download the job dataset from Kaggle:
- URL: https://www.kaggle.com/datasets/PromptCloudHQ/us-jobs-on-monstercom
- Save it as `data/jobs.csv`

### Step 4: Train Model (ONE-TIME ONLY)

```bash
# Train and save the model (takes 3-5 minutes)
python train_model.py

# Or with custom options:
python train_model.py --job-csv data/jobs.csv --method transformer
```

**Model Options:**
- `transformer` - Best accuracy (recommended) - 3-5 mins
- `hybrid` - Balanced performance - 4-6 mins  
- `tfidf` - Fastest - 1 min

### Step 5: Start Application

```bash
# Now the app loads instantly!
streamlit run app.py
```

**Note:** Training is done ONLY ONCE. After that, the model loads in seconds!

## 📖 How to Use

### For First Time Setup:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python setup.py
   ```

2. **Download job dataset** from [Kaggle](https://www.kaggle.com/datasets/PromptCloudHQ/us-jobs-on-monstercom)

3. **Train the model** (ONE TIME):
   ```bash
   python train_model.py
   ```
   This will:
   - Load and process all jobs
   - Train the AI model
   - Save everything to `models/` folder
   - Takes 3-5 minutes

4. **Start the app**:
   ```bash
   streamlit run app.py
   ```
   The model loads instantly from saved files!

### For Daily Use (After Training):

1. **Run the app**:
   ```bash
   streamlit run app.py
   ```
   (Model loads in seconds!)

2. **Upload resume** (PDF/DOCX/TXT)

3. **Review extracted skills**

4. **Click "Find Matching Jobs"**

5. **Explore and download results**

### Matching Algorithms:

| Algorithm | Speed | Accuracy | Best For |
|-----------|-------|----------|----------|
| **Transformer** | Slow | Highest | Best results |
| **Hybrid** | Medium | High | Balanced |
| **TF-IDF** | Fast | Good | Large datasets |

## 💡 Tips for Best Results

### Resume Writing Tips:

1. **Be Specific**: Include exact skill names (e.g., "Python", "React", "AWS")
2. **Use Keywords**: Match job description terminology
3. **Add Details**: Mention frameworks, tools, and technologies
4. **Include Projects**: Real-world experience matters
5. **Certifications**: Add relevant certifications

### Skills the System Detects:

- **Programming**: Python, Java, JavaScript, C++, etc.
- **Web Tech**: React, Angular, Node.js, Django, etc.
- **Databases**: MySQL, MongoDB, PostgreSQL, etc.
- **Cloud**: AWS, Azure, GCP, Docker, Kubernetes
- **Data Science**: ML, AI, TensorFlow, Pandas, etc.
- **Soft Skills**: Leadership, Communication, Agile, etc.

## 🎨 Screenshots

### Main Interface
Upload your resume and get instant job matches.

### Match Results
- Color-coded match quality (🟢 Excellent, 🟡 Good, 🟠 Moderate)
- Detailed score breakdown
- Skill comparison charts

### Skill Analysis
- Automatic skill detection
- Categorized by domain
- Gap analysis

## 🔧 Troubleshooting

### NLTK Error
**Problem**: `LookupError: Resource punkt not found`

**Solution**:
```python
python setup.py
```

### PDF Reading Error
**Problem**: Cannot read PDF file

**Solution**: 
- Ensure PDF is not password-protected
- Try converting to DOCX or TXT

### Low Match Scores
**Problem**: All matches below 50%

**Solution**:
- Add more technical details to resume
- Include specific skill names
- Use industry-standard terminology

### Model Loading Slow
**Problem**: Training takes too long

**Solution**:
- Use TF-IDF method instead of Transformer
- Reduce job dataset size
- Use a computer with more RAM

## 📊 System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space
- **Internet**: Required for initial model download

## 🔐 Privacy & Security

- All processing is done locally
- No data is sent to external servers
- Uploaded resumes are not stored permanently
- Job database remains on your machine

## 🚧 Future Enhancements

- [ ] Multiple resume upload
- [ ] Resume builder
- [ ] Email notifications
- [ ] Job application tracking
- [ ] Salary estimation
- [ ] Interview prep suggestions
- [ ] LinkedIn integration
- [ ] Cover letter generator

## 📝 Dataset Information

### Job Dataset Columns Required:
- `job_title`: Job title
- `company`: Company name
- `location`: Job location
- `job_description`: Full description

### Resume Format:
Any text-based resume in PDF, DOCX, or TXT format.

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Kaggle for job datasets
- Hugging Face for Sentence Transformers
- Streamlit team for the framework
- NLTK and spaCy communities

## 📧 Support

If you encounter issues:
1. Check this README
2. Run `python setup.py` again
3. Ensure all files are in correct folders
4. Check Python version (3.8+)

---

**Built with ❤️ for job seekers worldwide**

🌟 Star this project if you find it helpful!
