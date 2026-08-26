"""
Real evaluation dataset for the Vixxon AI RAG pipeline, covering
academic policy topics (HELB, grading, retakes, attendance, dress code,
credit waivers).

`category` tags were inferred from question content to enable per-topic
breakdown in the report — verify these groupings make sense for how you
want the report organized.

No `expected_keywords` field — retrieval scoring is fully automatic,
derived from each reference answer's own significant terms (see
retrieval_metrics.py), so no manual tagging was needed here.
"""

EVAL_CASES = [
    {
        "id": "credit_waiver_01",
        "category": "academic_policy",
        "question": "Can I get a credit waiver",
        "reference_answer": (
            "Yes, the University allows Credit waiver/transfers. Students who wish "
            "to apply for credit waiver/transfers are required to fill-in an "
            "application form which is available at the Office of Registrar (ARE) "
            "and submit the same for processing."
        ),
    },
    {
        "id": "helb_01",
        "category": "helb",
        "question": "Can I benefit from HELB even if my brother is currently using HELB",
        "reference_answer": (
            "Yes, each student makes their own application which is assessed "
            "individually."
        ),
    },
    {
        "id": "helb_02",
        "category": "helb",
        "question": "Do I need guarantors to get HELB",
        "reference_answer": "No guarantors are required for scholarship applications.",
    },
    {
        "id": "helb_03",
        "category": "helb",
        "question": "What is HELB",
        "reference_answer": (
            "The Higher Education Government Student Scholarship is a financial "
            "assistance programme provided by the Kenyan government to students "
            "studying in public universities and TVET colleges."
        ),
    },
    {
        "id": "grading_01",
        "category": "exams_grading",
        "question": "How are exams graded",
        "reference_answer": (
            "UoEm uses the Mark Grade System. Course Mark is a measure of "
            "performance of a student for a Semester. A course mark ranges from "
            "0 to 100%. Each course/unit is assessed based on the Continuous "
            "Assessment Test and End of Semester Examinations. Distribution of "
            "marks is subject to the respective School's rules and regulations. "
            "Grades: 70% and above = A; 60% to below 70% = B; 50% to below 60% = C; "
            "40% to below 50% = D; below 40% = F."
        ),
    },
    {
        "id": "retakes_01",
        "category": "exams_grading",
        "question": "How many times am I allowed to have retakes",
        "reference_answer": (
            "A candidate may be allowed to retake a Unit TWICE after which he/she "
            "shall be required to repeat the year. A candidate shall be allowed to "
            "repeat a year of study ONLY ONCE and shall be required to pay the "
            "statutory fees for each Semester repeated in addition to the "
            "applicable fee for each retake unit registered for. A candidate "
            "repeating a particular year of study will be required to take only "
            "the failed units and shall not be allowed to register for additional "
            "units."
        ),
    },
    {
        "id": "cat_01",
        "category": "exams_grading",
        "question": "When is CAT1 done",
        "reference_answer": "CAT 1 is done between the 6th and 7th week of a Semester",
    },
    {
        "id": "attendance_01",
        "category": "attendance",
        "question": "What percentage of lectures am I supposed to attend",
        "reference_answer": (
            "A student should attend at least 70% of the total lectures for each "
            "unit in the Semester"
        ),
    },
    {
        "id": "dress_code_01",
        "category": "conduct",
        "question": "What clothes am I supposed to wear",
        "reference_answer": (
            "In an effort to promote safety, professionalism and good image of the "
            "University, students are encouraged to wear acceptable and decent "
            "attire. While on campus, students should wear attire characterized by "
            "minimal skin/body exposure, non-transparency, well fitting, no "
            "revealing of inner wear, non-dramatic jewelry and makeup, well-kept "
            "hairdos, closed shoes, prescribed safety attire in desired places "
            "e.g. in laboratories/farms, minimal or no dramatic writings or "
            "drawings."
        ),
    },

    # Out-of-scope control — deliberately not in your original 9. Kept
    # from the earlier draft because it tests something none of your
    # real questions do: does the pipeline correctly decline when the
    # knowledge base has nothing relevant, rather than confidently
    # answering from an unrelated retrieved chunk? Given a real official
    # is evaluating this, a system that never admits "I don't know" is
    # a bigger red flag than one that occasionally does.
    {
        "id": "out_of_scope_01",
        "category": "out_of_scope",
        "question": "What time does the university library close on weekends?",
        "reference_answer": (
            "[OUT-OF-SCOPE CHECK: correct behavior is for the assistant to say "
            "it doesn't have this information, not to guess from unrelated "
            "retrieved content.]"
        ),
        "is_out_of_scope_check": True,
    },
]
