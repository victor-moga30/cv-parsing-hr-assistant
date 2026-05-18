# CV Parsing and HR Assistant

## Echipa
- Matei Amalia Andreea  
- Moga Victor Gabriel  
- Marin Lorena Anamaria  

---

## Problema abordata

In procesele moderne de recrutare, departamentele de HR trebuie sa analizeze un volum mare de CV-uri pentru fiecare pozitie disponibila. Acest proces este consumator de timp, ineficient si predispus la erori si bias.

Scopul acestui proiect este dezvoltarea unui sistem inteligent bazat pe AI, capabil sa analizeze automat CV-uri si descrieri de joburi, oferind suport decizional in procesul de selectie a candidatilor.

---

## Input

- CV-uri (format PDF sau text)
- Job Description (text)

---

## Output

- Scor de potrivire (%)
- Skill-uri identificate
- Skill-uri lipsa
- Recomandari pentru recrutori

---

## Utilizator tinta

- Recruiteri / specialisti HR

---

## Analiza datelor de intrare

Datele utilizate sunt:
- texte nestructurate (CV-uri)
- descrieri de joburi

### Probleme identificate:
- formate diferite (PDF, text)
- informatii nestructurate
- dificultate in extragerea automata a competentelor

### Informatii extrase:
- competente (skills)
- experienta profesionala
- educatie

---

## Schema solutiei

### Schema principala (flow sistem)

CV + Job Description  
        ↓  
Procesare text  
        ↓  
Extragere skill-uri  
        ↓  
Comparare skill-uri  
        ↓  
Scor de potrivire  
        ↓  
Recomandari HR  

![schema](img.png)

---

### Schema tip AI Agent (conceptual)

User input (CV + Job)  
        ↓  
AI Agent  
        ↓  
┌────────────────────┐  
│ Tool 1: Text Parse │  
│ Tool 2: Skill Extract │  
│ Tool 3: Matching │  
└────────────────────┘  
        ↓  
Output (scor + recomandari)

---

### Schema pentru etapa 1 (date)

Dataset CV-uri  
        ↓  
Curatare text  
        ↓  
Extragere skill-uri  
        ↓  
Analiza statistica  
        ↓  
Concluzii  

---

## SDG-uri relevante

- SDG 8 – Decent Work and Economic Growth  
  contribuie la eficientizarea procesului de recrutare  

- SDG 10 – Reduced Inequalities  
  reduce biasul in selectia candidatilor  

---

## Demo minimal

Pentru demonstrarea functionalitatii de baza, a fost implementat un exemplu simplu in fisierul `main.py`.

Acesta:
1. primeste un CV si o descriere de job  
2. extrage competentele relevante  
3. compara skill-urile  
4. calculeaza un scor de potrivire  
5. afiseaza rezultate si recomandari  

---

## Dataset

A fost utilizat un dataset de CV-uri si job descriptions provenit din surse publice.

Datele contin:
- `resume_text`
- `job_description`
- `skills_found`

Observatii:
- unele informatii personale lipsesc (anonimizare)
- datele sunt nestructurate
- nu toate CV-urile contin skill-uri detectabile

---

## AI Approach

Metode utilizate in etapa curenta:
- NLP (procesare text)
- keyword matching pentru extragerea skill-urilor
- analiza statistica a datelor

Extensii propuse (etapa urmatoare):
- embeddings (Sentence-BERT)
- semantic matching intre CV si job description
- scor de similaritate folosind cosine similarity

---

## Bias Reduction

Sistemul evalueaza candidatii exclusiv pe baza competentelor si experientei, fara utilizarea informatiilor personale (gen, varsta etc.), contribuind la reducerea biasului in procesul de recrutare.

---

## Concluzie – Etapa 1

In aceasta etapa:
- a fost definita problema
- au fost analizate datele de intrare
- s-au extras informatii relevante din CV-uri
- s-a realizat o analiza statistica a datelor
- s-a pregatit baza pentru dezvoltarea modelului AI

Etapa urmatoare va consta in:
- dezvoltarea modelului de scor de potrivire
- utilizarea embeddings si similarity
- imbunatatirea acuratetii sistemului