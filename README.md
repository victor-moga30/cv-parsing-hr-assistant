# CV Parsing and HR Assistant

## Echipa

- Matei Amalia Andreea
- Moga Victor Gabriel
- Marin Lorena Anamaria

---

## Problema abordata

In procesele moderne de recrutare, departamentele de HR trebuie sa analizeze rapid un numar mare de CV-uri pentru fiecare pozitie disponibila. Analiza manuala este consumatoare de timp, poate fi neuniforma si poate introduce bias in procesul de selectie.

Scopul proiectului este dezvoltarea unui sistem inteligent de tip AI Agent care analizeaza automat CV-uri si descrieri de job, extrage competente relevante, calculeaza un scor de potrivire si ofera recomandari explicabile pentru recrutori.

Aplicatia nu ia decizii automate de angajare. Ea ofera suport decizional pentru utilizatorul uman.

---

## Ce problema rezolva proiectul

Proiectul rezolva problema filtrarii initiale a CV-urilor pentru o anumita descriere de job.

Aplicatia primeste mai multe CV-uri, extrage informatiile importante din ele si le compara cu cerintele postului. La final, candidatii sunt ordonati dupa scorul de potrivire, iar recrutorul primeste explicatii, skill-uri potrivite, skill-uri lipsa, recomandari si intrebari pentru interviu.

---

## Input

Aplicatia primeste:

- unul sau mai multe CV-uri in format PDF, TXT sau imagine;
- o descriere de job introdusa de utilizator;
- optional, date deja procesate din folderul `data/`.

---

## Output

Aplicatia genereaza:

- scor final de matching pentru fiecare CV;
- ranking al candidatilor;
- top 3 candidati;
- skill-uri potrivite;
- skill-uri lipsa;
- scor semantic;
- scor TF-IDF;
- scor bazat pe acoperirea skill-urilor;
- recomandare HR;
- intrebari personalizate pentru interviu;
- red flags;
- dashboard cu grafice si statistici;
- metrici de evaluare a modelelor.

---

## Utilizator tinta

Aplicatia este destinata recrutorilor, specialistilor HR sau echipelor tehnice care doresc o prima filtrare explicabila a CV-urilor.

---

## Tipul de AI folosit

Proiectul foloseste o combinatie de metode NLP, machine learning si modele de tip Sentence-BERT.

Componente principale:

1. NLP clasic:
   - curatare text;
   - extragere de skill-uri;
   - TF-IDF;
   - cosine similarity.

2. Sentence-BERT:
   - model pretrained `all-MiniLM-L6-v2`;
   - model fine-tuned pe perechi CV - job description.

3. AI Agent:
   - agentul orchestreaza mai multe tool-uri interne;
   - fiecare tool are o responsabilitate clara;
   - agentul urmeaza un flux complet de analiza si produce rezultatul final explicabil.

---

## Arhitectura AI Agent

Fluxul agentului este:

```text
User input
  |
  |-- CV-uri PDF/TXT/imagini
  |-- Job description
  v
HRAssistantAgent
  |
  |-- DocumentParserTool
  |      extrage text din PDF/TXT/imagini
  |
  |-- ApplicantInfoTool
  |      extrage nume, email, telefon, LinkedIn, GitHub
  |
  |-- TextCleaningTool
  |      normalizeaza si curata textul
  |
  |-- SkillExtractionTool
  |      extrage competente din CV si job description
  |
  |-- MatchingTool
  |      calculeaza semantic score, TF-IDF score, skill score si final score
  |
  |-- RecommendationTool
  |      genereaza recomandari, intrebari si red flags
  v
Output final:
ranking, scoruri, explicatii si recomandari
```

---

## Tool-uri interne folosite de agent

Agentul foloseste urmatoarele tool-uri interne. Aceste tool-uri sunt apelate in ordine de clasa `HRAssistantAgent`.

---

### 1. DocumentParserTool

Acest tool extrage textul din fisierele incarcate de utilizator.

Suporta:

- PDF-uri digitale;
- fisiere TXT;
- imagini;
- PDF-uri scanate, daca este instalat Tesseract OCR.

Pentru PDF-uri se foloseste initial `pypdf`. Daca extragerea nu este suficienta, se foloseste PyMuPDF. Pentru fisiere scanate sau imagini se foloseste OCR.

Rolul acestui tool este important deoarece, in practica, majoritatea CV-urilor sunt trimise in format PDF.

---

### 2. ApplicantInfoTool

Acest tool extrage informatii de baza despre candidat:

- nume;
- email;
- telefon;
- LinkedIn;
- GitHub;
- lungimea textului extras.

Aplicatia nu foloseste informatii sensibile precum varsta, gen, etnie, poza sau alte date care pot introduce bias.

Scopul acestui tool este sa ofere recrutorului date utile de contact, fara sa influenteze scorul de matching prin informatii personale sensibile.

---

### 3. TextCleaningTool

Acest tool curata si normalizeaza textul extras din CV si job description.

Exemple de preprocesare:

- eliminare spatii inutile;
- normalizare text;
- reparare variante precum `C + +`;
- transformare intr-o forma mai usor de procesat;
- pregatirea textului pentru extragerea de skill-uri si matching semantic.

Acest pas este necesar deoarece textul extras din PDF-uri poate contine spatii, randuri rupte sau caractere neuniforme.

---

### 4. SkillExtractionTool

Acest tool identifica competentele tehnice si soft skills din CV si job description.

Exemple de skill-uri detectate:

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

Skill-urile sunt extrase folosind un dictionar de competente si alias-uri.

Exemplu:

```text
"py" -> Python
"sklearn" -> Scikit-learn
"postgre sql" -> PostgreSQL
"powerbi" -> Power BI
"object oriented programming" -> OOP
```

Acest tool ajuta la obtinerea unui scor explicabil, deoarece recrutorul poate vedea exact ce skill-uri au fost gasite si ce skill-uri lipsesc.

---

### 5. MatchingTool

Acest tool calculeaza scorurile de potrivire dintre CV si job description.

Sunt calculate mai multe scoruri:

- `semantic_score` - similaritate semantica folosind Sentence-BERT;
- `tfidf_score` - similaritate clasica intre texte folosind TF-IDF;
- `skill_score` - procentul de skill-uri cerute de job care apar in CV;
- `baseline_score` - scor hibrid clasic;
- `final_score` - scor final folosit pentru ranking.

Formula generala este:

```text
final_score =
    semantic_weight * semantic_score
  + baseline_weight * baseline_score
  + skill_weight * skill_score
```

Aceasta combinatie face scorul mai stabil si mai explicabil.

Scorul semantic ajuta atunci cand CV-ul si job description-ul folosesc formulari diferite, dar au sens apropiat.

Scorul TF-IDF ajuta la compararea clasica a textelor.

Scorul pe skill-uri ajuta la explicabilitate, deoarece se poate vedea concret ce cerinte sunt indeplinite.

---

### 6. RecommendationTool

Acest tool genereaza:

- recomandare HR;
- intrebari de interviu;
- red flags;
- verdict final.

Exemple de verdict:

- Strong match;
- Good match;
- Medium match;
- Weak match.

Exemple de recomandari:

- candidatul este potrivit pentru interviu tehnic;
- candidatul poate fi pastrat ca backup;
- candidatul are prea multe skill-uri lipsa;
- profilul trebuie verificat prin intrebari practice.

Exemple de intrebari generate:

- Describe a concrete project where you used Python.
- Do you have experience with Docker or a similar technology?
- What was the most complex technical project you worked on?

Exemple de red flags:

- text extras foarte scurt din CV;
- acoperire redusa a skill-urilor cerute;
- similaritate semantica redusa;
- multe skill-uri lipsa.

---

## Planificarea pasilor agentului

Agentul executa urmatorii pasi:

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

Acest flux arata ca agentul nu intoarce doar un scor simplu, ci coordoneaza mai multe instrumente software pentru a produce un rezultat final complet si explicabil.

---

## Schema solutiei

```text
CV PDF/TXT/Imagine + Job Description
        |
        v
Document parsing
        |
        v
Applicant info extraction
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
Ranking + recomandari + dashboard + metrici
```

---

## Structura proiectului

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

## Rulare proiect

### 1. Instalare dependinte

```bash
pip install -r requirements.txt
```

Pentru OCR pe PDF-uri scanate sau imagini trebuie instalat separat Tesseract OCR in sistem.

Pentru PDF-uri digitale, OCR-ul nu este obligatoriu.

---

### 2. Rulare dashboard

```bash
streamlit run app.py
```

---

### 3. Rulare demo CLI

```bash
python main.py
```

---

### 4. Regenerare metrici

```bash
python -m src.evaluate
```

---

### 5. Antrenare model fine-tuned

```bash
python -m src.train_finetune
```

---

## Dataset

Proiectul foloseste date procesate in folderul `data/`.

Fisiere importante:

- `processed_resumes.csv`;
- `processed_jobs.csv`;
- `training_pairs.csv`.

Perechile CV - job sunt etichetate prin acoperirea skill-urilor cerute de job. Label-ul este intre 0 si 1.

Aceasta eticheta reprezinta cat de bine se potriveste un CV cu o descriere de job.

---

## Modele evaluate

Au fost evaluate trei variante.

---

### 1. baseline_tfidf_skill

Acesta este modelul de baza.

Foloseste:

- similaritate TF-IDF;
- acoperirea skill-urilor;
- scor hibrid.

Acest model este explicabil, dar poate rata potriviri semantice daca termenii din CV si job description sunt exprimati diferit.

---

### 2. pretrained_sbert

Acest model foloseste Sentence-BERT pretrained.

Avantajul sau este ca poate compara texte semantic, nu doar prin potrivire exacta de cuvinte.

De exemplu, poate intelege ca doua formulari diferite pot avea sens apropiat.

---

### 3. fine_tuned_sbert

Acesta este modelul final imbunatatit.

Modelul Sentence-BERT a fost adaptat pe perechi CV - job description, astfel incat sa invete mai bine ce inseamna potrivire intre un candidat si o descriere de job.

Acest model este folosit pentru scorul semantic final.

---

## Metrici folosite

Pentru evaluare s-au folosit:

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

Accuracy nu este suficienta, deoarece datele pot fi dezechilibrate. De aceea se folosesc si Precision, Recall si F1-score.

---

## Interpretarea rezultatelor

Baseline-ul poate avea accuracy mare daca majoritatea perechilor sunt negative, dar poate avea F1-score slab pentru clasa pozitiva.

Modelul fine-tuned SBERT este mai relevant deoarece invata similaritatea dintre CV si job description, nu doar potrivirea exacta a termenilor.

Astfel, evaluarea proiectului nu se bazeaza doar pe accuracy, ci si pe metrici mai relevante pentru problema de matching, cum ar fi:

- F1-score;
- Precision;
- Recall;
- RMSE;
- Pearson correlation;
- Spearman correlation.

---

## Dashboard

Dashboard-ul Streamlit ofera:

- incarcare multipla de CV-uri;
- introducere job description;
- analiza automata;
- ranking candidati;
- top 3 candidati;
- recomandari;
- intrebari de interviu;
- red flags;
- distributia scorurilor;
- comparatie skill-uri potrivite/lipsa;
- comparatie intre scor semantic si skill score;
- comparatie intre metricile modelelor.

---

## Reducerea biasului

Aplicatia incearca sa reduca biasul prin:

- evaluarea candidatilor pe baza competentelor si textului profesional;
- ignorarea informatiilor sensibile;
- folosirea aceluiasi proces standardizat pentru toti candidatii;
- afisarea explicatiilor pentru scor;
- pastrarea deciziei finale la utilizatorul uman.

Sistemul nu trebuie folosit ca mecanism automat de respingere sau acceptare a candidatilor.

Aplicatia nu trebuie sa inlocuiasca recrutorul, ci sa il ajute sa ia decizii mai rapide si mai bine argumentate.

---

## Recomandari si intrebari de interviu

Pentru fiecare candidat, sistemul genereaza recomandari si intrebari.

Exemple:

- daca un skill apare si in CV si in job description, se genereaza o intrebare practica despre acel skill;
- daca un skill important lipseste, se genereaza o intrebare de clarificare;
- daca scorul este mic, sistemul semnaleaza ca profilul nu este suficient de potrivit pentru job.

Aceste recomandari sunt explicabile si pot fi verificate de recrutor.

---

## Red flags

Aplicatia poate detecta red flags precum:

- text extras foarte scurt din CV;
- acoperire redusa a skill-urilor cerute;
- similaritate semantica redusa;
- multe skill-uri lipsa.

Aceste red flags nu sunt decizii finale, ci semnale pentru recrutor.

---

## Imbunatatiri realizate

In proiect au fost realizate urmatoarele imbunatatiri:

- extragere text din PDF-uri;
- suport pentru fisiere TXT si imagini;
- fallback OCR pentru fisiere scanate;
- extragere informatii despre candidat;
- extinderea listei de skill-uri;
- model semantic folosind Sentence-BERT;
- fine-tuning pentru modelul SBERT;
- evaluare cu metrici multiple;
- generare ranking pentru mai multi candidati;
- dashboard interactiv;
- recomandari explicabile;
- intrebari de interviu;
- red flags;
- separarea logicii in tool-uri interne folosite de agent;
- demo CLI prin `main.py`;
- salvarea rezultatelor in `reports/hr_dashboard_results.csv`.

---

## Limitari

- PDF-urile scanate necesita Tesseract OCR instalat separat.
- Extragerea skill-urilor depinde de dictionarul de competente.
- Unele CV-uri pot avea formatare greu de citit.
- Modelul poate mosteni bias din datele de antrenare.
- Scorul final este un suport decizional, nu o decizie automata de angajare.
- Recomandarile generate sunt reguli explicabile, nu verdict juridic sau decizie finala.
- Daca descrierea jobului este foarte scurta sau neclara, scorul poate fi mai putin relevant.
- Daca CV-ul nu contine explicit anumite skill-uri, sistemul poate considera ca acestea lipsesc.
- Etichetele folosite pentru evaluare sunt generate automat pe baza acoperirii skill-urilor cerute de job. Din acest motiv, metricile trebuie interpretate ca evaluare tehnica a prototipului, nu ca validare pe decizii reale de recrutare umana. Aplicatia este un instrument de suport decizional si nu trebuie folosita pentru decizii automate de angajare.

---

## SDG-uri impactate

### SDG 8 - Decent Work and Economic Growth

Proiectul poate eficientiza recrutarea si poate ajuta companiile sa identifice mai rapid candidati potriviti.

Prin reducerea timpului petrecut pentru filtrarea manuala, echipele HR pot investi mai mult timp in interviuri, evaluari calitative si discutii reale cu aplicantii.

---

### SDG 10 - Reduced Inequalities

Prin standardizarea evaluarii si evitarea informatiilor personale sensibile, proiectul poate contribui la reducerea biasului in selectia initiala a candidatilor.

Aplicatia compara CV-urile cu cerintele jobului, nu cu aspecte personale ale candidatului.

---

## Concluzie

Proiectul implementeaza un prototip functional de CV Parsing and HR Assistant.

Sistemul extrage text din CV-uri, analizeaza competentele, compara CV-urile cu descrierea jobului, calculeaza scoruri explicabile si ofera recomandari utile pentru recrutori.

Proiectul include atat partea de AI/ML, cat si un modul de interactiune prin Streamlit si un demo CLI minimal.

Aplicatia este gandita ca suport decizional explicabil pentru HR, nu ca sistem automat de angajare.