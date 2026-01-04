"""
Setup script for AI Resume Matcher
Run this before starting the application
"""

import os
import nltk
import ssl

print("=" * 60)
print("AI Resume Matcher - Setup Script")
print("=" * 60)

# Fix SSL certificate issue
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Create necessary directories
print("\n1. Creating directories...")
os.makedirs("data", exist_ok=True)
os.makedirs("src", exist_ok=True)
print("   ✓ Directories created")

# Download NLTK data
print("\n2. Downloading NLTK data...")
required_data = ['punkt', 'stopwords', 'wordnet', 'punkt_tab', 'averaged_perceptron_tagger']

for data in required_data:
    try:
        nltk.download(data, quiet=False)
        print(f"   ✓ Downloaded {data}")
    except Exception as e:
        print(f"   ✗ Failed to download {data}: {e}")

print("\n" + "=" * 60)
print("Setup Complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Make sure all files are in the correct folders:")
print("   - preprocessing.py, model.py, utils.py → src/")
print("   - app.py → root folder")
print("   - requirements.txt → root folder")
print("\n2. Run: streamlit run app.py")
print("=" * 60)