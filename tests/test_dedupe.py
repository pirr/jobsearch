from jobsearch.vacancies.dedupe import vacancy_id


def test_same_role_across_boards_collapses():
    # doctolib and parloa really do publish to Greenhouse and Ashby at once.
    a = vacancy_id("Acme", "Senior Backend Engineer", "Berlin", "Build services.")
    b = vacancy_id("ACME ", "senior backend engineer", "berlin", "Build services.")
    assert a == b


def test_gender_suffix_variants_collapse():
    a = vacancy_id("Acme", "Backend Engineer (m/f/d)", "Berlin", "x" * 40)
    b = vacancy_id("Acme", "Backend Engineer", "Berlin", "x" * 40)
    assert a == b


def test_different_roles_stay_distinct():
    a = vacancy_id("Acme", "Backend Engineer", "Berlin", "Python work")
    b = vacancy_id("Acme", "Frontend Engineer", "Berlin", "React work")
    assert a != b


def test_late_description_edits_do_not_fork_identity():
    # Identity uses the first 500 chars, so an edited paragraph further down
    # must not resurrect a vacancy the candidate already dismissed.
    base = "y" * 600
    assert vacancy_id("A", "T", "L", base) == vacancy_id("A", "T", "L", base + " appended note")
