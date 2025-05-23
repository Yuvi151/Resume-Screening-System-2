from flask import Flask, request, render_template, jsonify
from PyPDF2 import PdfReader
from datetime import datetime
from dateutil import parser as date_parser
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
import pickle

app = Flask(__name__)
resume_database = []


# Load models===========================================================================================================
rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))

# Resume ranker=========================================================================================================
class ResumeRanker:
    def __init__(self):
        self.skill_weights = {
            'technical_skills': 0.3,
            'education': 0.2,
            'experience_indicators': 0.2,
            'job_match': 0.1,
            'projects':0.2
            # 'contact_completeness': 0.05,
            # 'soft_skills': 0.15,
        }
        
        # Define skill categories
        self.technical_skills = [
            'Python', 'Java', 'JavaScript', 'C++', 'SQL', 'HTML', 'CSS', 'React', 'Angular', 
            'Node.js', 'MongoDB', 'Machine Learning', 'Deep Learning', 'Data Analysis',
            'TensorFlow', 'PyTorch', 'Git', 'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP'
        ]
        
        self.experience_keywords = [
            'years', 'experience', 'worked', 'developed', 'managed', 'led', 'created', 
            'implemented', 'designed', 'built', 'achieved', 'improved', 'optimized'
        ]
        
        self.education_levels = {
            'PhD': 5, 'Doctorate': 5, 'Ph.D': 5,
            'Master': 4, 'Masters': 4, 'M.S': 4, 'M.A': 4, 'MBA': 4,
            'Bachelor': 3, 'Bachelors': 3, 'B.S': 3, 'B.A': 3, 'B.Tech': 3,
            'Associate': 2, 'Diploma': 2,
            'Certificate': 1
        }

    def calculate_technical_skills_score(self, skills):
        """Calculate score based on technical skills"""
        if not skills:
            return 0
        
        technical_found = [skill for skill in skills if skill in self.technical_skills]
        # Normalize score (max 100)
        return min(len(technical_found) * 5, 100)
    
    def calculate_education_score(self, education):
        """Calculate score based on education level"""
        if not education:
            return 0
        
        max_level = 0
        for edu in education:
            for level, score in self.education_levels.items():
                if level.lower() in edu.lower():
                    max_level = max(max_level, score)
        
        return (max_level / 5) * 100  # Normalize to 100
    
    def score_experience(self,exp_list):
        score = 0
        for exp in exp_list:
            try:
                start = date_parser.parse(exp["start_date"])
                end = datetime.now() if "Present" in exp["end_date"] else date_parser.parse(exp["end_date"])
                years = (end - start).days / 365

                role = exp["title"].lower()
                if "intern" in role:
                    multiplier = 1.5
                elif any(x in role for x in ["lead", "manager", "head"]):
                    multiplier = 2.0
                elif any(x in role for x in ["senior"]):
                    multiplier = 1.7
                else:
                    multiplier = 1

                score += years * multiplier
            except:
                continue
        return min(score * 10, 100)  # Cap to 100

    # below function not preferred as it is a simple keywords and pattern matching method
    '''def calculate_experience_score(self, resume_text):  
        """Calculate experience score based on keywords and patterns"""
        experience_count = 0
        
        # Look for experience keywords
        for keyword in self.experience_keywords:
            experience_count += len(re.findall(r'\b' + keyword + r'\b', resume_text, re.IGNORECASE))
        
        # Look for year patterns (e.g., "3 years", "2+ years")
        year_pattern = r'\b\d+\+?\s*years?\b'
        year_matches = re.findall(year_pattern, resume_text, re.IGNORECASE)
        
        # Extract numerical values from year matches
        total_years = 0
        for match in year_matches:
            numbers = re.findall(r'\d+', match)
            if numbers:
                total_years += int(numbers[0])
        
        # Combine keyword frequency and years mentioned
        experience_score = min((experience_count * 2) + (total_years * 5), 100)
        return experience_score'''

    def score_projects(self,projects):
        if not projects:
            return 0
        
        # Load SpaCy model (this contains word vectors)
        nlp = spacy.load('en_core_web_md')
    
        high_quality_examples = [
            "Built an end-to-end ML pipeline using AWS Sagemaker and deployed using FastAPI.",
            "Developed a scalable web application with React, Node.js, and MongoDB used by 10k+ users.",
            "Created an AI chatbot using NLP and transformers to automate customer support."
        ]
    
        # Convert examples to SpaCy docs with embeddings
        example_docs = [nlp(example) for example in high_quality_examples]
    
        total_score = 0
        for project in projects:
            project_doc = nlp(project)
        
            # Calculate similarities with each example
            similarities = [project_doc.similarity(example_doc) for example_doc in example_docs]
            max_similarity = max(similarities)
        
            # Scale to 100
            project_score = 100 * max_similarity
            total_score += project_score
    
        avg_score = total_score / len(projects)
        return min(round(avg_score, 2), 10)
    
    def calculate_job_match_score(self, resume_text, job_description=""):
        """Calculate how well resume matches a job description"""
        if not job_description:
            return 50  # Default neutral score if no job description provided
        
        # Use TF-IDF to calculate similarity
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
        try:
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity * 100
        except:
            return 50  # Return neutral score if calculation fails
    
    def rank_resume(self, resume_data, job_description=""):
        """Calculate overall ranking score for a resume"""
        scores = {}
        
        # Calculate individual component scores
        scores['technical_skills'] = self.calculate_technical_skills_score(resume_data.get('skills', []))
        scores['projects'] = self.score_projects(resume_data.get('projects', []))
        scores['education'] = self.calculate_education_score(resume_data.get('education', []))
        scores['experience_indicators'] = self.score_experience(resume_data.get('text', ''))
        scores['job_match'] = self.calculate_job_match_score(resume_data.get('text', ''), job_description)
        
        # Calculate weighted total score
        total_score = sum(scores[component] * self.skill_weights[component] for component in scores)
        
        return {
            'total_score': round(total_score, 2),
            'component_scores': scores,
            'grade': self.get_grade(total_score)
        }
    
    def get_grade(self, score):
        """Convert numerical score to letter grade"""
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        else:
            return 'D'
    
    def rank_multiple_resumes(self, resumes, job_description=""):
        """Rank multiple resumes and return sorted list"""
        ranked_resumes = []
        
        for resume in resumes:
            ranking = self.rank_resume(resume, job_description)
            resume_with_ranking = resume.copy()
            resume_with_ranking['ranking'] = ranking
            ranked_resumes.append(resume_with_ranking)
        
        # Sort by total score (descending)
        ranked_resumes.sort(key=lambda x: x['ranking']['total_score'], reverse=True)
        
        # Add rank position
        for i, resume in enumerate(ranked_resumes):
            resume['rank_position'] = i + 1
        
        return ranked_resumes

# Initialize ranker
ranker = ResumeRanker()

# Clean resume==========================================================================================================
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Prediction and Category Name
def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
    return predicted_category

# Prediction and Category Name
def job_recommendation(resume_text):
    resume_text= cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    recommended_job = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    return recommended_job

def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text




# resume parsing
import re

def extract_contact_number_from_resume(text):
    contact_number = None

    # Use regex pattern to find a potential contact number
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    if match:
        contact_number = match.group()

    return contact_number
def extract_email_from_resume(text):
    email = None

    # Use regex pattern to find a potential email address
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        email = match.group()

    return email

def extract_skills_from_resume(text):
    # List of predefined skills
    skills_list = [
        'Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management', 'Deep Learning', 'SQL',
        'Tableau',
        'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'React', 'Angular', 'Node.js', 'MongoDB', 'Express.js', 'Git',
        'Research', 'Statistics', 'Quantitative Analysis', 'Qualitative Analysis', 'SPSS', 'R', 'Data Visualization',
        'Matplotlib',
        'Seaborn', 'Plotly', 'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'NLTK', 'Text Mining',
        'Natural Language Processing', 'Computer Vision', 'Image Processing', 'OCR', 'Speech Recognition',
        'Recommendation Systems',
        'Collaborative Filtering', 'Content-Based Filtering', 'Reinforcement Learning', 'Neural Networks',
        'Convolutional Neural Networks',
        'Recurrent Neural Networks', 'Generative Adversarial Networks', 'XGBoost', 'Random Forest', 'Decision Trees',
        'Support Vector Machines',
        'Linear Regression', 'Logistic Regression', 'K-Means Clustering', 'Hierarchical Clustering', 'DBSCAN',
        'Association Rule Learning',
        'Apache Hadoop', 'Apache Spark', 'MapReduce', 'Hive', 'HBase', 'Apache Kafka', 'Data Warehousing', 'ETL',
        'Big Data Analytics',
        'Cloud Computing', 'Amazon Web Services (AWS)', 'Microsoft Azure', 'Google Cloud Platform (GCP)', 'Docker',
        'Kubernetes', 'Linux',
        'Shell Scripting', 'Cybersecurity', 'Network Security', 'Penetration Testing', 'Firewalls', 'Encryption',
        'Malware Analysis',
        'Digital Forensics', 'CI/CD', 'DevOps', 'Agile Methodology', 'Scrum', 'Kanban', 'Continuous Integration',
        'Continuous Deployment',
        'Software Development', 'Web Development', 'Mobile Development', 'Backend Development', 'Frontend Development',
        'Full-Stack Development',
        'UI/UX Design', 'Responsive Design', 'Wireframing', 'Prototyping', 'User Testing', 'Adobe Creative Suite',
        'Photoshop', 'Illustrator',
        'InDesign', 'Figma', 'Sketch', 'Zeplin', 'InVision', 'Product Management', 'Market Research',
        'Customer Development', 'Lean Startup',
        'Business Development', 'Sales', 'Marketing', 'Content Marketing', 'Social Media Marketing', 'Email Marketing',
        'SEO', 'SEM', 'PPC',
        'Google Analytics', 'Facebook Ads', 'LinkedIn Ads', 'Lead Generation', 'Customer Relationship Management (CRM)',
        'Salesforce',
        'HubSpot', 'Zendesk', 'Intercom', 'Customer Support', 'Technical Support', 'Troubleshooting',
        'Ticketing Systems', 'ServiceNow',
        'ITIL', 'Quality Assurance', 'Manual Testing', 'Automated Testing', 'Selenium', 'JUnit', 'Load Testing',
        'Performance Testing',
        'Regression Testing', 'Black Box Testing', 'White Box Testing', 'API Testing', 'Mobile Testing',
        'Usability Testing', 'Accessibility Testing',
        'Cross-Browser Testing', 'Agile Testing', 'User Acceptance Testing', 'Software Documentation',
        'Technical Writing', 'Copywriting',
        'Editing', 'Proofreading', 'Content Management Systems (CMS)', 'WordPress', 'Joomla', 'Drupal', 'Magento',
        'Shopify', 'E-commerce',
        'Payment Gateways', 'Inventory Management', 'Supply Chain Management', 'Logistics', 'Procurement',
        'ERP Systems', 'SAP', 'Oracle',
        'Microsoft Dynamics', 'Tableau', 'Power BI', 'QlikView', 'Looker', 'Data Warehousing', 'ETL',
        'Data Engineering', 'Data Governance',
        'Data Quality', 'Master Data Management', 'Predictive Analytics', 'Prescriptive Analytics',
        'Descriptive Analytics', 'Business Intelligence',
        'Dashboarding', 'Reporting', 'Data Mining', 'Web Scraping', 'API Integration', 'RESTful APIs', 'GraphQL',
        'SOAP', 'Microservices',
        'Serverless Architecture', 'Lambda Functions', 'Event-Driven Architecture', 'Message Queues', 'GraphQL',
        'Socket.io', 'WebSockets'
                     'Ruby', 'Ruby on Rails', 'PHP', 'Symfony', 'Laravel', 'CakePHP', 'Zend Framework', 'ASP.NET', 'C#',
        'VB.NET', 'ASP.NET MVC', 'Entity Framework',
        'Spring', 'Hibernate', 'Struts', 'Kotlin', 'Swift', 'Objective-C', 'iOS Development', 'Android Development',
        'Flutter', 'React Native', 'Ionic',
        'Mobile UI/UX Design', 'Material Design', 'SwiftUI', 'RxJava', 'RxSwift', 'Django', 'Flask', 'FastAPI',
        'Falcon', 'Tornado', 'WebSockets',
        'GraphQL', 'RESTful Web Services', 'SOAP', 'Microservices Architecture', 'Serverless Computing', 'AWS Lambda',
        'Google Cloud Functions',
        'Azure Functions', 'Server Administration', 'System Administration', 'Network Administration',
        'Database Administration', 'MySQL', 'PostgreSQL',
        'SQLite', 'Microsoft SQL Server', 'Oracle Database', 'NoSQL', 'MongoDB', 'Cassandra', 'Redis', 'Elasticsearch',
        'Firebase', 'Google Analytics',
        'Google Tag Manager', 'Adobe Analytics', 'Marketing Automation', 'Customer Data Platforms', 'Segment',
        'Salesforce Marketing Cloud', 'HubSpot CRM',
        'Zapier', 'IFTTT', 'Workflow Automation', 'Robotic Process Automation (RPA)', 'UI Automation',
        'Natural Language Generation (NLG)',
        'Virtual Reality (VR)', 'Augmented Reality (AR)', 'Mixed Reality (MR)', 'Unity', 'Unreal Engine', '3D Modeling',
        'Animation', 'Motion Graphics',
        'Game Design', 'Game Development', 'Level Design', 'Unity3D', 'Unreal Engine 4', 'Blender', 'Maya',
        'Adobe After Effects', 'Adobe Premiere Pro',
        'Final Cut Pro', 'Video Editing', 'Audio Editing', 'Sound Design', 'Music Production', 'Digital Marketing',
        'Content Strategy', 'Conversion Rate Optimization (CRO)',
        'A/B Testing', 'Customer Experience (CX)', 'User Experience (UX)', 'User Interface (UI)', 'Persona Development',
        'User Journey Mapping', 'Information Architecture (IA)',
        'Wireframing', 'Prototyping', 'Usability Testing', 'Accessibility Compliance', 'Internationalization (I18n)',
        'Localization (L10n)', 'Voice User Interface (VUI)',
        'Chatbots', 'Natural Language Understanding (NLU)', 'Speech Synthesis', 'Emotion Detection',
        'Sentiment Analysis', 'Image Recognition', 'Object Detection',
        'Facial Recognition', 'Gesture Recognition', 'Document Recognition', 'Fraud Detection',
        'Cyber Threat Intelligence', 'Security Information and Event Management (SIEM)',
        'Vulnerability Assessment', 'Incident Response', 'Forensic Analysis', 'Security Operations Center (SOC)',
        'Identity and Access Management (IAM)', 'Single Sign-On (SSO)',
        'Multi-Factor Authentication (MFA)', 'Blockchain', 'Cryptocurrency', 'Decentralized Finance (DeFi)',
        'Smart Contracts', 'Web3', 'Non-Fungible Tokens (NFTs)']


    skills = []

    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)

    return skills


def extract_education_from_resume(text):
    education = []

    # List of education keywords to match against
    education_keywords = [
        'Computer Science', 'Information Technology', 'Software Engineering', 'Electrical Engineering', 'Mechanical Engineering', 'Civil Engineering',
        'Chemical Engineering', 'Biomedical Engineering', 'Aerospace Engineering', 'Nuclear Engineering', 'Industrial Engineering', 'Systems Engineering',
        'Environmental Engineering', 'Petroleum Engineering', 'Geological Engineering', 'Marine Engineering', 'Robotics Engineering', 'Biotechnology',
        'Biochemistry', 'Microbiology', 'Genetics', 'Molecular Biology', 'Bioinformatics', 'Neuroscience', 'Biophysics', 'Biostatistics', 'Pharmacology',
        'Physiology', 'Anatomy', 'Pathology', 'Immunology', 'Epidemiology', 'Public Health', 'Health Administration', 'Nursing', 'Medicine', 'Dentistry',
        'Pharmacy', 'Veterinary Medicine', 'Medical Technology', 'Radiography', 'Physical Therapy', 'Occupational Therapy', 'Speech Therapy', 'Nutrition',
        'Sports Science', 'Kinesiology', 'Exercise Physiology', 'Sports Medicine', 'Rehabilitation Science', 'Psychology', 'Counseling', 'Social Work',
        'Sociology', 'Anthropology', 'Criminal Justice', 'Political Science', 'International Relations', 'Economics', 'Finance', 'Accounting', 'Business Administration',
        'Management', 'Marketing', 'Entrepreneurship', 'Hospitality Management', 'Tourism Management', 'Supply Chain Management', 'Logistics Management',
        'Operations Management', 'Human Resource Management', 'Organizational Behavior', 'Project Management', 'Quality Management', 'Risk Management',
        'Strategic Management', 'Public Administration', 'Urban Planning', 'Architecture', 'Interior Design', 'Landscape Architecture', 'Fine Arts',
        'Visual Arts', 'Graphic Design', 'Fashion Design', 'Industrial Design', 'Product Design', 'Animation', 'Film Studies', 'Media Studies',
        'Communication Studies', 'Journalism', 'Broadcasting', 'Creative Writing', 'English Literature', 'Linguistics', 'Translation Studies',
        'Foreign Languages', 'Modern Languages', 'Classical Studies', 'History', 'Archaeology', 'Philosophy', 'Theology', 'Religious Studies',
        'Ethics', 'Education', 'Early Childhood Education', 'Elementary Education', 'Secondary Education', 'Special Education', 'Higher Education',
        'Adult Education', 'Distance Education', 'Online Education', 'Instructional Design', 'Curriculum Development'
        'Library Science', 'Information Science', 'Computer Engineering', 'Software Development', 'Cybersecurity', 'Information Security',
        'Network Engineering', 'Data Science', 'Data Analytics', 'Business Analytics', 'Operations Research', 'Decision Sciences',
        'Human-Computer Interaction', 'User Experience Design', 'User Interface Design', 'Digital Marketing', 'Content Strategy',
        'Brand Management', 'Public Relations', 'Corporate Communications', 'Media Production', 'Digital Media', 'Web Development',
        'Mobile App Development', 'Game Development', 'Virtual Reality', 'Augmented Reality', 'Blockchain Technology', 'Cryptocurrency',
        'Digital Forensics', 'Forensic Science', 'Criminalistics', 'Crime Scene Investigation', 'Emergency Management', 'Fire Science',
        'Environmental Science', 'Climate Science', 'Meteorology', 'Geography', 'Geomatics', 'Remote Sensing', 'Geoinformatics',
        'Cartography', 'GIS (Geographic Information Systems)', 'Environmental Management', 'Sustainability Studies', 'Renewable Energy',
        'Green Technology', 'Ecology', 'Conservation Biology', 'Wildlife Biology', 'Zoology']

    for keyword in education_keywords:
        pattern = r"(?i)\b{}\b".format(re.escape(keyword))
        match = re.search(pattern, text)
        if match:
            education.append(match.group())

    return education


def extract_projects_from_resume(text):
    project_keywords = ['projects', 'project experience', 'personal projects']
    section_delimiters = ['experience', 'education', 'skills', 'certifications', 'summary', 'profile']

    # Normalize text
    lower_text = text.lower()

    # Search for a project section
    for keyword in project_keywords:
        if keyword in lower_text:
            start_idx = lower_text.index(keyword)
            end_idx = len(text)

            # Try to find where the next section begins
            for delimiter in section_delimiters:
                delimiter_idx = lower_text.find(delimiter, start_idx + len(keyword))
                if delimiter_idx != -1:
                    end_idx = min(end_idx, delimiter_idx)

            # Extract and clean the section
            project_section = text[start_idx:end_idx].strip()
            lines = project_section.split('\n')
            project_lines = [line.strip() for line in lines if len(line.strip()) > 20]

            return project_lines[:5]

    return []


def extract_name_from_resume(text):
    name = None

    # Use regex pattern to find a potential name
    pattern = r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)"
    match = re.search(pattern, text)
    if match:
        name = match.group()

    return name




# routes===============================================

@app.route('/')
def resume():
    # Provide a simple UI to upload a resume
    return render_template("resume.html")

@app.route('/pred', methods=['POST'])
def pred():
    # Process the PDF or TXT file and make prediction
    if 'resume' in request.files:
        file = request.files['resume']
        filename = file.filename
        if filename.endswith('.pdf'):
            text = pdf_to_text(file)
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            return render_template('resume.html', message="Invalid file format. Please upload a PDF or TXT file.")

        predicted_category = predict_category(text)
        recommended_job = job_recommendation(text)
        phone = extract_contact_number_from_resume(text)
        email = extract_email_from_resume(text)

        extracted_skills = extract_skills_from_resume(text)
        extracted_education = extract_education_from_resume(text)
        name = extract_name_from_resume(text)
        projects = extract_projects_from_resume(text)
        resume_entry = {
            'filename': filename,
            'name': name,
            'text': text,
            'skills': extracted_skills,
            'education': extracted_education,
            'projects': projects
        }
        resume_database.append(resume_entry)

        return render_template('resume.html', predicted_category=predicted_category,recommended_job=recommended_job,
                               phone=phone,name=name,email=email,extracted_skills=extracted_skills,extracted_education=extracted_education)
    else:
        return render_template("resume.html", message="No resume file uploaded.")
    
@app.route('/bulk_rank', methods=['POST'])
def bulk_rank():
    """Rank all uploaded resumes"""
    if not resume_database:
        return jsonify({'error': 'No resumes available for ranking'})
    
    job_description = request.form.get('job_description', '')
    ranked_resumes = ranker.rank_multiple_resumes(resume_database, job_description)
    
    # Prepare response data
    result = []
    for resume in ranked_resumes:
        result.append({
            'filename': resume['filename'],
            'name': resume['name'],
            'rank_position': resume['rank_position'],
            'total_score': resume['ranking']['total_score'],
            'grade': resume['ranking']['grade'],
            'component_scores': resume['ranking']['component_scores']
        })
    
    return jsonify({'ranked_resumes': result})

@app.route('/clear_database', methods=['POST'])
def clear_database():
    """Clear the resume database"""
    global resume_database
    resume_database = []
    return jsonify({'message': 'Database cleared successfully'})

@app.route('/ranking_dashboard')
def ranking_dashboard():
    """Display ranking dashboard"""
    return render_template('ranking_dashboard.html', resumes=resume_database)

if __name__ == '__main__':
    app.run(debug=True)