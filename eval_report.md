# RAG Pipeline Evaluation Report

Generated: 2026-08-26T09:01:12.894158+00:00
Total questions evaluated: 10

## Overall Scores

| Metric | Score |
|---|---|
| Correctness (1-5, judge) | 5.0 |
| Faithfulness (1-5, judge) | 5.0 |
| Retrieval — keyword overlap (0-1) | 0.93 |
| Retrieval — semantic similarity (0-1) | 0.39 |

## By Category

| Category | Count | Correctness | Faithfulness | Retrieval (keyword) | Retrieval (semantic) |
|---|---|---|---|---|---|
| academic_policy | 1 | None | None | 1.0 | 0.45 |
| helb | 3 | 5.0 | 5.0 | 1.0 | 0.33 |
| exams_grading | 3 | 5.0 | 5.0 | 1.0 | 0.53 |
| attendance | 1 | 5.0 | 5.0 | 1.0 | 0.47 |
| conduct | 1 | None | None | 1.0 | 0.36 |
| out_of_scope | 1 | None | None | 0.33 | 0.07 |

## Flagged for Manual Review (8)

### credit_waiver_01: Can I get a credit waiver

- **Correctness:** None/5
- **Faithfulness:** None/5
- **Retrieval (keyword):** 1.0
- **Retrieval (semantic):** 0.445
- **Judge reasoning:** None
- **Generated answer:** Yes, the University of Embu allows credit waiver/transfers. If you wish to apply for a credit waiver or transfer, you are required to fill in an application form, which is available at the Office of the Registrar (ARE). Once you have completed the form, you should submit it for processing. Please note that this process is separate from fee waivers, which are offered to specific groups such as university staff, their dependents, employees of organizations with memoranda of understanding with the University, and other groups as determined by management.
- **Reference answer:** Yes, the University allows Credit waiver/transfers. Students who wish to apply for credit waiver/transfers are required to fill-in an application form which is available at the Office of Registrar (ARE) and submit the same for processing.

### helb_01: Can I benefit from HELB even if my brother is currently using HELB

- **Correctness:** 5/5
- **Faithfulness:** 5/5
- **Retrieval (keyword):** 1.0
- **Retrieval (semantic):** 0.222
- **Judge reasoning:** The generated answer correctly matches the reference answer and is directly supported by retrieved context FAQ 18, which states each student's application is assessed individually even if a sibling is already benefiting.
- **Generated answer:** Yes, you can still benefit from HELB funding even if your sibling is currently benefiting from the same program. According to the official HELB FAQs, each student makes their own independent application, which is assessed individually. This means that your eligibility for funding is determined based on your own circumstances and financial needs, not on whether a sibling is already receiving funding. Therefore, you are encouraged to apply on your own behalf, and your application will be considered separately.
- **Reference answer:** Yes, each student makes their own application which is assessed individually.

### helb_02: Do I need guarantors to get HELB

- **Correctness:** None/5
- **Faithfulness:** None/5
- **Retrieval (keyword):** 1.0
- **Retrieval (semantic):** 0.327
- **Judge reasoning:** None
- **Generated answer:** No, you do not need a guarantor to be eligible for the HELB (Higher Education Loans Board) loan. According to the information provided, for the Government Student Scholarship, **no guarantors are required**. However, please note that this applies specifically to the scholarship application. For the loan application, the steps do mention updating guarantor details, which suggests that guarantors may be needed for the loan component of the funding. It's always best to check the specific requirements during the application process on the HEF Student Portal at www.hef.co.ke.
- **Reference answer:** No guarantors are required for scholarship applications.

### helb_03: What is HELB

- **Correctness:** None/5
- **Faithfulness:** None/5
- **Retrieval (keyword):** 1.0
- **Retrieval (semantic):** 0.45
- **Judge reasoning:** None
- **Generated answer:** HELB stands for the Higher Education Loans Board. It is a Kenyan government agency that provides financial assistance in the form of loans and scholarships to students pursuing higher education in public universities and TVET colleges. The primary purpose of HELB is to help students cover the costs of tuition, upkeep, and other educational expenses, making higher education more accessible to Kenyan students.

Key features of HELB include:
- **Loan Types**: There are two main types of undergraduate loan applications: First Time Application (for new applicants) and Second & Subsequent Applications (for continuing students, based on the program duration).
- **Loan Award Range**: The loan amount ranges from a minimum of KShs. 40,000 to a maximum of KShs. 60,000. For government-sponsored students, a portion is allocated for tuition (KShs. 4,000 per semester, totaling KShs. 8,000 per year) and the rest is sent to the student's bank account as upkeep, while self-sponsored students receive the full amount as tuition fees paid directly to the university.
- **Application Process**: Students apply online through the HEF (Higher Education Financing) portal, where they must create an account, fill in their personal, academic, and family details, and submit the application. Required documents include a valid email, mobile phone number, KCPE and KCSE index numbers, a passport-size photo, admission letter, birth certificate or National ID, and parent/guardian details.
- **Repayment**: Loan repayment starts within one year of completing studies, with a 4% interest rate per annum, and is repayable over a maximum of 120 months. Voluntary payments are allowed to reduce the loan balance.
- **Scholarships**: Needy government-sponsored students can receive a bursary of between KShs. 4,000 and KShs. 8,000 per year, paid directly to the university.

HELB is managed under the Universities Fund and is crucial in enabling many Kenyans to access and complete higher education. Students can apply annually for funding, and additional support can be sought through appeals if their financial situation changes.
- **Reference answer:** The Higher Education Government Student Scholarship is a financial assistance programme provided by the Kenyan government to students studying in public universities and TVET colleges.

### cat_01: When is CAT1 done

- **Correctness:** 5/5
- **Faithfulness:** 5/5
- **Retrieval (keyword):** 1.0
- **Retrieval (semantic):** 0.338
- **Judge reasoning:** The generated answer directly matches the reference answer and is explicitly supported by the retrieved context, which states CAT 1 is done between the 6th and 7th week of the semester.
- **Generated answer:** CAT1 is done between the 6th and 7th week of the semester.
- **Reference answer:** CAT 1 is done between the 6th and 7th week of a Semester

### attendance_01: What percentage of lectures am I supposed to attend

- **Correctness:** 5/5
- **Faithfulness:** 5/5
- **Retrieval (keyword):** 1.0
- **Retrieval (semantic):** 0.468
- **Judge reasoning:** The generated answer gives the required 70% attendance and adds accurate consequences drawn directly from the retrieved context, matching the reference answer without unsupported claims.
- **Generated answer:** You are required to attend at least **70%** of the total lectures for each unit in the semester. If you attend less than 70%, you will **not be allowed to sit for the end-of-semester examinations** for that unit, or you may automatically be deemed to have failed the unit(s). Additionally, if you are absent from **30% or more** of the classes in a semester—whether with or without permission—you become ineligible to sit for the end-of-semester examinations.
- **Reference answer:** A student should attend at least 70% of the total lectures for each unit in the Semester

### dress_code_01: What clothes am I supposed to wear

- **Correctness:** None/5
- **Faithfulness:** None/5
- **Retrieval (keyword):** 1.0
- **Retrieval (semantic):** 0.358
- **Judge reasoning:** None
- **Generated answer:** Based on the University's guidelines, you are expected to wear acceptable and decent attire that promotes safety, professionalism, and a good image of the University. Specifically, your clothing should be characterized by:

- Minimal skin/body exposure.
- Non-transparency.
- Well-fitting clothes.
- No revealing of inner wear.
- Non-dramatic jewelry and makeup.
- Well-kept hairdos.
- Closed shoes.
- Prescribed safety attire in designated areas, such as laboratories or farms.
- Minimal or no dramatic writings or drawings on your attire.

These guidelines apply while you are on campus and are intended to ensure a respectful and professional environment for all students.
- **Reference answer:** In an effort to promote safety, professionalism and good image of the University, students are encouraged to wear acceptable and decent attire. While on campus, students should wear attire characterized by minimal skin/body exposure, non-transparency, well fitting, no revealing of inner wear, non-dramatic jewelry and makeup, well-kept hairdos, closed shoes, prescribed safety attire in desired places e.g. in laboratories/farms, minimal or no dramatic writings or drawings.

### out_of_scope_01: What time does the university library close on weekends?

- **Correctness:** None/5
- **Faithfulness:** None/5
- **Retrieval (keyword):** 0.333
- **Retrieval (semantic):** 0.074
- **Judge reasoning:** None
- **Generated answer:** Based on the provided information, the operating hours for the University of Embu Health Unit are specified, but there are no details provided regarding the library's closing time on weekends.

However, the library's website is mentioned as a resource: "Kindly visit the Library Website (https://library.embuni.ac.ke/) for detailed guide on LCC" — though this is specifically about the Library of Congress Classification system, not operating hours.

For accurate weekend closing times, I recommend checking the official library website (https://library.embuni.ac.ke/) or contacting the library directly. The library is open to users, but specific hours are not listed in the provided context.
- **Reference answer:** [OUT-OF-SCOPE CHECK: correct behavior is for the assistant to say it doesn't have this information, not to guess from unrelated retrieved content.]
