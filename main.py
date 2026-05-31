from src.agent import HRAssistantAgent


def print_result(result):
    applicant = result.get("applicant_info", {})

    print("\n" + "=" * 70)
    print("CV Parsing and HR Assistant - CLI demo")
    print("=" * 70)

    print("\nApplicant info:")
    print(f"Name: {applicant.get('name', 'Unknown')}")
    print(f"Email: {applicant.get('email', 'Unknown')}")
    print(f"Phone: {applicant.get('phone', 'Unknown')}")
    print(f"LinkedIn: {applicant.get('linkedin', 'Unknown')}")
    print(f"GitHub: {applicant.get('github', 'Unknown')}")

    print("\nScores:")
    print(f"Final score: {result['final_score']}%")
    print(f"Semantic score: {result['semantic_score']}%")
    print(f"Baseline score: {result['baseline_score']}%")
    print(f"TF-IDF score: {result['tfidf_score']}%")
    print(f"Skill score: {result['skill_score']}%")
    print(f"Verdict: {result['verdict']}")

    print("\nMatched skills:")
    print(", ".join(result["matched_skills"]) if result["matched_skills"] else "None")

    print("\nMissing skills:")
    print(", ".join(result["missing_skills"]) if result["missing_skills"] else "None")

    print("\nRecommendation:")
    print(result["recommendation"])

    print("\nInterview questions:")
    for index, question in enumerate(result["interview_questions"], start=1):
        print(f"{index}. {question}")

    print("\nRed flags:")
    for flag in result["red_flags"]:
        print(f"- {flag}")

    print("\nAgent steps:")
    print(" -> ".join(result["agent_steps"]))

    print("\n" + "=" * 70)


def main():
    cv_text = """
    Victor Moga
    Email: victor@example.com
    GitHub: github.com/victormoga

    Computer Science student with experience in Python, SQL, Git, machine learning,
    pandas, scikit-learn and data analysis. I worked on projects involving NLP,
    classification, data preprocessing and dashboard development.
    """

    job_text = """
    We are looking for a Machine Learning Intern with Python, SQL, Git,
    machine learning, NLP, pandas, scikit-learn and good communication skills.
    The candidate should be able to analyze data and explain model results.
    """

    agent = HRAssistantAgent()
    result = agent.run(cv_text=cv_text, job_text=job_text)

    print_result(result)


if __name__ == "__main__":
    main()