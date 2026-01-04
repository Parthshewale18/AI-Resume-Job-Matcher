"""
One-time model training script
Run this once to train and save the model
"""

import sys
import os
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import load_and_preprocess_jobs
from model import ResumeJobMatcher

def train_and_save_model(job_csv_path, model_method='transformer'):
    """
    Train the model and save everything needed for inference
    
    Args:
        job_csv_path: Path to jobs.csv file
        model_method: 'transformer', 'hybrid', or 'tfidf'
    """
    
    print("=" * 60)
    print("AI Resume Matcher - Model Training")
    print("=" * 60)
    
    # Step 1: Load and preprocess jobs
    print(f"\n[1/4] Loading job dataset from: {job_csv_path}")
    jobs_df, preprocessor, skill_extractor = load_and_preprocess_jobs(job_csv_path)
    print(f"✓ Loaded {len(jobs_df)} jobs")
    
    # Step 2: Train model
    print(f"\n[2/4] Training {model_method} model...")
    print("This may take 2-5 minutes depending on your system...")
    matcher = ResumeJobMatcher(method=model_method)
    matcher.train(jobs_df)
    print("✓ Model trained successfully!")
    
    # Step 3: Save everything
    print("\n[3/4] Saving model and data...")
    
    os.makedirs("models", exist_ok=True)
    
    # Save the trained matcher
    joblib.dump(matcher, "models/matcher.pkl")
    print("✓ Saved matcher model")
    
    # Save the jobs dataframe
    joblib.dump(jobs_df, "models/jobs_df.pkl")
    print("✓ Saved jobs dataframe")
    
    # Save preprocessor and skill extractor
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    joblib.dump(skill_extractor, "models/skill_extractor.pkl")
    print("✓ Saved preprocessor and skill extractor")
    
    # Step 4: Verify
    print("\n[4/4] Verifying saved files...")
    
    files_to_check = [
        "models/matcher.pkl",
        "models/jobs_df.pkl",
        "models/preprocessor.pkl",
        "models/skill_extractor.pkl"
    ]
    
    all_exist = all(os.path.exists(f) for f in files_to_check)
    
    if all_exist:
        print("✓ All files saved successfully!")
        
        # Show file sizes
        print("\nModel files:")
        for file in files_to_check:
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"  - {file}: {size_mb:.2f} MB")
        
        print("\n" + "=" * 60)
        print("SUCCESS! Model training complete.")
        print("=" * 60)
        print("\nYou can now run: streamlit run app.py")
        print("The app will load the pre-trained model instantly!")
        print("=" * 60)
    else:
        print("✗ Error: Some files were not saved properly")
        return False
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and save the resume matching model")
    parser.add_argument(
        "--job-csv",
        type=str,
        default="data/jobs.csv",
        help="Path to jobs CSV file (default: data/jobs.csv)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=['transformer', 'hybrid', 'tfidf'],
        default='transformer',
        help="Matching method (default: transformer)"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.job_csv):
        print(f"Error: Job CSV file not found at: {args.job_csv}")
        print("\nPlease:")
        print("1. Download jobs.csv from Kaggle")
        print("2. Place it in the data/ folder")
        print("3. Run this script again")
        sys.exit(1)
    
    # Train and save
    success = train_and_save_model(args.job_csv, args.method)
    
    if not success:
        sys.exit(1)