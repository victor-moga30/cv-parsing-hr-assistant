# CV Parsing and HR Assistant

## Team

- Matei Amalia Andreea
- Moga Victor Gabriel
- Marin Lorena Anamaria

---

## Problem Addressed

In modern recruitment processes, HR departments need to quickly analyze a large number of CVs for each available position. Manual analysis is time-consuming, can be inconsistent, and may introduce bias into the selection process.

The goal of this project is to develop an intelligent AI Agent system that automatically analyzes CVs and job descriptions, extracts relevant skills, calculates a matching score, and provides explainable recommendations for recruiters.

The application does not make automated hiring decisions. It provides decision support for the human user.

---

## What Problem Does the Project Solve?

The project addresses the problem of initial CV screening for a given job description.

The application receives multiple CVs, extracts relevant information from them, and compares it with the job requirements. At the end, candidates are ranked according to their matching score, while the recruiter receives explanations, matched skills, missing skills, recommendations, and interview questions.

---

## Input

The application receives:

- one or more CVs in PDF, TXT, or image format;
- a job description entered by the user;
- optionally, already processed data from the `data/` folder.

---

## Output

The application generates:

- final matching score for each CV;
- candidate ranking;
- top 3 candidates;
- matched skills;
- missing skills;
- semantic score;
- TF-IDF score;
- skill coverage score;
- HR recommendation;
- personalized interview questions;
- red flags;
- dashboard with charts and statistics;
- model evaluation metrics.

---

## Target Users

The application is intended for recruiters, HR specialists, or technical teams that want an explainable first-stage CV screening process.

---

## Type of AI Used

The project uses a combination of NLP methods, machine learning, and Sentence-BERT models.

Main components:

1. Classical NLP:
   - text cleaning;
   - skill extraction;
   - TF-IDF;
   - cosine similarity.

2. Sentence-BERT:
   - pretrained `all-MiniLM-L6-v2` model;
   - model fine-tuned on CV–job description pairs.

3. AI Agent:
   - the agent orchestrates several internal tools;
   - each tool has a clearly defined responsibility;
   - the agent follows a complete analysis pipeline and produces an explainable final result.

---

## AI Agent Architecture

The agent workflow is:

```text
User input
  |
  |-- CVs in PDF/TXT/image format
  |-- Job description
  v
HRAssistantAgent
  |
  |-- DocumentParserTool
  |      extracts text from PDF/TXT/images
  |
  |-- ApplicantInfoTool
  |      extracts name, email, phone number, LinkedIn, GitHub
  |
  |-- TextCleaningTool
  |      normalizes and cleans the text
  |
  |-- SkillExtractionTool
  |      extracts skills from the CV and job description
  |
  |-- MatchingTool
  |      calculates semantic score, TF-IDF score, skill score,
  |      and final score
  |
  |-- RecommendationTool
  |      generates recommendations, questions, and red flags
  v
Final output:
ranking, scores, explanations, and recommendations
```

---

## Internal Tools Used by the Agent

The agent uses the following internal tools. These tools are called in sequence by the `HRAssistantAgent` class.

---

### 1. DocumentParserTool

This tool extracts text from files uploaded by the user.

It supports:

- digital PDFs;
- TXT files;
- images;
- scanned PDFs, if Tesseract OCR is installed.

For PDFs, `pypdf` is used first. If text extraction is insufficient, PyMuPDF is used. OCR is used for scanned files or images.

This tool plays an important role because, in practice, most CVs are submitted in PDF format.

---

### 2. ApplicantInfoTool

This tool extracts basic information about the candidate:

- name;
- email;
- phone number;
- LinkedIn;
- GitHub;
- length of the extracted text.

The application does not use sensitive information such as age, gender, ethnicity, photo, or other data that could introduce bias.

The purpose of this tool is to provide recruiters with useful contact information without allowing sensitive personal information to influence the matching score.

---

### 3. TextCleaningTool

This tool cleans and normalizes the text extracted from the CV and job description.

Examples of preprocessing:

- removing unnecessary spaces;
- text normalization;
- repairing variants such as `C + +`;
- transforming the text into a format that is easier to process;
- preparing the text for skill extraction and semantic matching.

This step is necessary because text extracted from PDFs may contain irregular spacing, broken lines, or inconsistent characters.

---

### 4. SkillExtractionTool

This tool identifies technical skills and soft skills from the CV and job description.

Examples of detected skills:

- Python;
- Java;
- C++;
- C#;
- SQL;
- MySQL;
- PostgreSQL;
- Git;
- GitHub;
- Linux;
- Bash;
- Machine Learning;
- Deep Learning;
- Artificial Intelligence;
- NLP;
- TensorFlow;
- PyTorch;
- Pandas;
- NumPy;
- Scikit-learn;
- React;
- Angular;
- Spring;
- Docker;
- Kubernetes;
- Excel;
- Power BI;
- Communication;
- Teamwork;
- Problem Solving.

Skills are extracted using a dictionary of skills and aliases.

Example:

```text
"py" -> Python
"sklearn" -> Scikit-learn
"postgre sql" -> PostgreSQL
"powerbi" -> Power BI
"object oriented programming" -> OOP
```

This tool helps produce an explainable score because the recruiter can see exactly which skills were found and which skills are missing.

---

### 5. MatchingTool

This tool calculates the matching scores between a CV and a job description.

Several scores are calculated:

- `semantic_score` - semantic similarity using Sentence-BERT;
- `tfidf_score` - classical text similarity using TF-IDF;
- `skill_score` - percentage of job-required skills found in the CV;
- `baseline_score` - classical hybrid score;
- `final_score` - final score used for ranking.

The general formula is:

```text
final_score =
    semantic_weight * semantic_score
  + baseline_weight * baseline_score
  + skill_weight * skill_score
```

This combination makes the score more stable and explainable.

The semantic score is useful when the CV and job description use different wording but express similar meanings.

The TF-IDF score provides a classical text similarity comparison.

The skill score improves explainability because it shows exactly which requirements are satisfied.

---

### 6. RecommendationTool

This tool generates:

- HR recommendation;
- interview questions;
- red flags;
- final verdict.

Examples of verdicts:

- Strong match;
- Good match;
- Medium match;
- Weak match.

Examples of recommendations:

- the candidate is suitable for a technical interview;
- the candidate can be kept as a backup option;
- the candidate is missing too many required skills;
- the profile should be further evaluated through practical questions.

Examples of generated questions:

- Describe a concrete project where you used Python.
- Do you have experience with Docker or a similar technology?
- What was the most complex technical project you worked on?

Examples of red flags:

- very little text extracted from the CV;
- low coverage of required skills;
- low semantic similarity;
- many missing skills.

---

## Agent Step Planning

The agent executes the following steps:

```text
1. extract_applicant_info
2. clean_cv_text
3. clean_job_text
4. extract_cv_skills
5. extract_job_skills
6. compute_matching_scores
7. generate_recommendation
8. generate_interview_questions
9. detect_red_flags
10. return_explainable_result
```

This workflow shows that the agent does not simply return a single score, but instead coordinates several software tools to produce a complete and explainable result.

---

## Solution Architecture

```text
CV PDF/TXT/Image + Job Description
        |
        v
Document parsing
        |
        v
Applicant information extraction
        |
        v
Text cleaning
        |
        v
Skill extraction
        |
        v
Semantic matching + TF-IDF + skill coverage
        |
        v
Final score
        |
        v
Ranking + recommendations + dashboard + metrics
```

---

## Project Structure

```text
.
|-- app.py
|-- main.py
|-- README.md
|-- requirements.txt
|-- data/
|   |-- processed_resumes.csv
|   |-- processed_jobs.csv
|   |-- training_pairs.csv
|-- models/
|   |-- fine_tuned_sbert/
|-- reports/
|   |-- model_metrics.csv
|   |-- classification_report_*.txt
|   |-- confusion_matrix_*.csv
|   |-- optimization_results.csv
|-- src/
|   |-- agent.py
|   |-- baseline.py
|   |-- create_training_pairs.py
|   |-- document_parser.py
|   |-- evaluate.py
|   |-- final_matcher.py
|   |-- optimize_matcher.py
|   |-- preprocessing.py
|   |-- semantic_matcher.py
|   |-- skill_extraction.py
|   |-- tools.py
|   |-- train_finetune.py
```

---

## Running the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For OCR on scanned PDFs or images, Tesseract OCR must be installed separately on the system.

OCR is not required for digital PDFs.

---

### 2. Run the Dashboard

```bash
streamlit run app.py
```

---

### 3. Run the CLI Demo

```bash
python main.py
```

---

### 4. Regenerate Metrics

```bash
python -m src.evaluate
```

---

### 5. Train the Fine-Tuned Model

```bash
python -m src.train_finetune
```

---

## Dataset

The project uses processed data stored in the `data/` folder.

Important files:

- `processed_resumes.csv`;
- `processed_jobs.csv`;
- `training_pairs.csv`.

CV–job pairs are labeled according to the coverage of skills required by the job. The label ranges from 0 to 1.

This label represents how well a CV matches a job description.

---

## Evaluated Models

Three approaches were evaluated.

---

### 1. baseline_tfidf_skill

This is the baseline model.

It uses:

- TF-IDF similarity;
- skill coverage;
- hybrid scoring.

This model is explainable, but it may miss semantic matches when concepts in the CV and job description are expressed using different wording.

---

### 2. pretrained_sbert

This model uses a pretrained Sentence-BERT model.

Its advantage is that it can compare texts semantically rather than relying only on exact word matching.

For example, it can understand that two different expressions may have similar meanings.

---

### 3. fine_tuned_sbert

This is the final improved model.

The Sentence-BERT model was adapted using CV–job description pairs so that it could better learn what constitutes a match between a candidate and a job description.

This model is used for the final semantic score.

---

## Evaluation Metrics

The following metrics were used for evaluation:

- MAE;
- MSE;
- RMSE;
- Pearson correlation;
- Spearman correlation;
- Accuracy;
- Precision;
- Recall;
- F1-score;
- confusion matrix;
- classification report.

Accuracy alone is not sufficient because the data may be imbalanced. Therefore, Precision, Recall, and F1-score are also used.

---

## Interpretation of Results

The baseline model may achieve high accuracy if most pairs are negative, while still obtaining a poor F1-score for the positive class.

The fine-tuned SBERT model is more relevant because it learns similarity between CVs and job descriptions rather than relying only on exact term matching.

Therefore, the project evaluation is not based solely on accuracy, but also on metrics that are more relevant for the matching problem, such as:

- F1-score;
- Precision;
- Recall;
- RMSE;
- Pearson correlation;
- Spearman correlation.

---

## Dashboard

The Streamlit dashboard provides:

- multiple CV uploads;
- job description input;
- automatic analysis;
- candidate ranking;
- top 3 candidates;
- recommendations;
- interview questions;
- red flags;
- score distribution;
- comparison of matched and missing skills;
- comparison between semantic score and skill score;
- comparison between model metrics.

---

## Bias Reduction

The application attempts to reduce bias by:

- evaluating candidates based on skills and professional text;
- ignoring sensitive information;
- using the same standardized process for all candidates;
- displaying explanations for the score;
- keeping the final decision in the hands of the human user.

The system should not be used as an automated candidate rejection or acceptance mechanism.

The application is not intended to replace recruiters, but rather to help them make faster and better-informed decisions.

---

## Recommendations and Interview Questions

For each candidate, the system generates recommendations and questions.

Examples:

- if a skill appears both in the CV and in the job description, a practical question about that skill is generated;
- if an important skill is missing, a clarification question is generated;
- if the score is low, the system indicates that the profile is not sufficiently well matched to the job.

These recommendations are explainable and can be reviewed by the recruiter.

---

## Red Flags

The application can detect red flags such as:

- very little text extracted from the CV;
- low coverage of required skills;
- low semantic similarity;
- many missing skills.

These red flags are not final decisions, but signals for the recruiter.

---

## Implemented Improvements

The following improvements were implemented in the project:

- text extraction from PDFs;
- support for TXT files and images;
- OCR fallback for scanned files;
- candidate information extraction;
- expanded skill list;
- semantic model using Sentence-BERT;
- fine-tuning of the SBERT model;
- evaluation using multiple metrics;
- ranking generation for multiple candidates;
- interactive dashboard;
- explainable recommendations;
- interview questions;
- red flags;
- separation of logic into internal tools used by the agent;
- CLI demo through `main.py`;
- saving results to `reports/hr_dashboard_results.csv`.

---

## Limitations

- Scanned PDFs require Tesseract OCR to be installed separately.
- Skill extraction depends on the skills dictionary.
- Some CVs may use formatting that is difficult to parse.
- The model may inherit bias from the training data.
- The final score is intended as decision support, not as an automated hiring decision.
- Generated recommendations are explainable rules, not legal verdicts or final decisions.
- If the job description is very short or unclear, the score may be less relevant.
- If the CV does not explicitly mention certain skills, the system may consider them missing.
- The labels used for evaluation are automatically generated based on the coverage of skills required by the job. For this reason, the metrics should be interpreted as a technical evaluation of the prototype rather than validation against real human recruitment decisions. The application is a decision-support tool and should not be used for automated hiring decisions.

---

## Impacted SDGs

### SDG 8 - Decent Work and Economic Growth

The project can improve the efficiency of recruitment and help companies identify suitable candidates more quickly.

By reducing the amount of time spent on manual filtering, HR teams can invest more time in interviews, qualitative evaluations, and real discussions with applicants.

---

### SDG 10 - Reduced Inequalities

By standardizing evaluation and avoiding sensitive personal information, the project may contribute to reducing bias during the initial candidate selection process.

The application compares CVs with job requirements rather than with candidates' personal characteristics.

---

## Conclusion

The project implements a functional prototype of a CV Parsing and HR Assistant.

The system extracts text from CVs, analyzes skills, compares CVs with job descriptions, calculates explainable scores, and provides useful recommendations for recruiters.

The project includes both AI/ML components and an interactive Streamlit module, as well as a minimal CLI demo.

The application is designed as an explainable HR decision-support tool, not as an automated hiring system.
