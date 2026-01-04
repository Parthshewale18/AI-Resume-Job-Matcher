import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter


def visualize_category_distribution(resumes_df):
    """
    Visualize distribution of resume categories
    
    Args:
        resumes_df (DataFrame): Resume dataframe
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    category_counts = resumes_df['Category'].value_counts()
    
    sns.barplot(x=category_counts.values, y=category_counts.index, ax=ax)
    ax.set_xlabel('Count')
    ax.set_ylabel('Category')
    ax.set_title('Distribution of Resume Categories')
    
    return fig


def visualize_top_skills(resumes_df, top_n=20):
    """
    Visualize top skills across all resumes
    
    Args:
        resumes_df (DataFrame): Resume dataframe
        top_n (int): Number of top skills to show
        
    Returns:
        matplotlib figure
    """
    all_skills = []
    for skills in resumes_df['skills']:
        all_skills.extend(skills)
    
    skill_counts = Counter(all_skills)
    top_skills = dict(skill_counts.most_common(top_n))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    skills = list(top_skills.keys())
    counts = list(top_skills.values())
    
    sns.barplot(x=counts, y=skills, ax=ax)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Skill')
    ax.set_title(f'Top {top_n} Skills in Resumes')
    
    return fig


def create_skill_wordcloud(resumes_df):
    """
    Create word cloud of skills
    
    Args:
        resumes_df (DataFrame): Resume dataframe
        
    Returns:
        matplotlib figure
    """
    all_skills = []
    for skills in resumes_df['skills']:
        all_skills.extend(skills)
    
    skills_text = ' '.join(all_skills)
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis'
    ).generate(skills_text)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Skills Word Cloud', fontsize=20)
    
    return fig


def plot_match_scores(results_df):
    """
    Plot match scores for job recommendations
    
    Args:
        results_df (DataFrame): Results from matching
        
    Returns:
        plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=results_df.index,
        y=results_df['match_score'],
        name='Overall Match',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        x=results_df.index,
        y=results_df['semantic_score'],
        name='Semantic Match',
        marker_color='lightcoral'
    ))
    
    fig.add_trace(go.Bar(
        x=results_df.index,
        y=results_df['skill_score'],
        name='Skill Match',
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title='Match Scores Breakdown',
        xaxis_title='Job Rank',
        yaxis_title='Score',
        barmode='group',
        height=500
    )
    
    return fig


def create_skill_comparison(resume_skills, job_skills):
    """
    Create visualization comparing resume and job skills
    
    Args:
        resume_skills (list): Skills from resume
        job_skills (list): Required skills from job
        
    Returns:
        plotly figure
    """
    resume_set = set(resume_skills)
    job_set = set(job_skills)
    
    matched = list(resume_set.intersection(job_set))
    missing = list(job_set - resume_set)
    extra = list(resume_set - job_set)
    
    fig = go.Figure()
    
    categories = ['Matched Skills', 'Missing Skills', 'Additional Skills']
    values = [len(matched), len(missing), len(extra)]
    colors = ['green', 'red', 'blue']
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=values,
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Skill Comparison',
        yaxis_title='Number of Skills',
        height=400
    )
    
    return fig


def calculate_metrics(resumes_df, jobs_df, matcher):
    """
    Calculate and display matching metrics
    
    Args:
        resumes_df (DataFrame): Resume dataframe
        jobs_df (DataFrame): Job dataframe
        matcher: Trained matcher model
        
    Returns:
        dict: Metrics dictionary
    """
    metrics = {
        'total_resumes': len(resumes_df),
        'total_jobs': len(jobs_df),
        'avg_resume_skills': resumes_df['skills'].apply(len).mean(),
        'avg_job_skills': jobs_df['required_skills'].apply(len).mean(),
        'unique_categories': resumes_df['Category'].nunique()
    }
    
    return metrics


def format_job_display(job_row):
    """
    Format job information for display
    
    Args:
        job_row (Series): Job data row
        
    Returns:
        str: Formatted job information
    """
    output = f"""
    **Job Title:** {job_row.get('job_title', 'N/A')}
    **Company:** {job_row.get('company', 'N/A')}
    **Location:** {job_row.get('location', 'N/A')}
    **Match Score:** {job_row.get('match_score', 0):.2%}
    **Semantic Score:** {job_row.get('semantic_score', 0):.2%}
    **Skill Score:** {job_row.get('skill_score', 0):.2%}
    """
    return output


def format_resume_display(resume_row):
    """
    Format resume information for display
    
    Args:
        resume_row (Series): Resume data row
        
    Returns:
        str: Formatted resume information
    """
    output = f"""
    **Category:** {resume_row.get('Category', 'N/A')}
    **Match Score:** {resume_row.get('match_score', 0):.2%}
    **Semantic Score:** {resume_row.get('semantic_score', 0):.2%}
    **Skill Score:** {resume_row.get('skill_score', 0):.2%}
    **Skills:** {', '.join(resume_row.get('skills', [])[:10])}
    """
    return output


def export_results_to_csv(results_df, filename):
    """
    Export matching results to CSV
    
    Args:
        results_df (DataFrame): Results dataframe
        filename (str): Output filename
    """
    results_df.to_csv(filename, index=False)
    print(f"Results exported to {filename}")


def get_category_recommendations(category, resumes_df, jobs_df, matcher, top_k=5):
    """
    Get job recommendations for a specific category
    
    Args:
        category (str): Resume category
        resumes_df (DataFrame): Resume dataframe
        jobs_df (DataFrame): Job dataframe
        matcher: Trained matcher
        top_k (int): Number of recommendations
        
    Returns:
        DataFrame: Top jobs for the category
    """
    category_resumes = resumes_df[resumes_df['Category'] == category]
    
    if len(category_resumes) == 0:
        return pd.DataFrame()
    
    # Get recommendations for first resume in category
    sample_idx = category_resumes.index[0]
    recommendations = matcher.match_resume_to_jobs(sample_idx, resumes_df, jobs_df, top_k)
    
    return recommendations