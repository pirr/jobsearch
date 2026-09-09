from conftest import make_vacancy


def seed_active(store, profile, n=3):
    """n vacancies, scores descending (90, 80, 70, ...), all active/rated."""
    h = profile.hash()
    ids = []
    for i in range(n):
        v = make_vacancy(source_id=str(i), title=f"Role {i}", company=f"Company {i}")
        store.upsert_vacancies([v])
        store.save_rating(v.id, h, "f", score=90 - i * 10)
        ids.append(v.id)
    return ids


async def test_app_renders_active_vacancies_score_desc_on_mount(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import DataTable

    ids = seed_active(store, profile)
    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#table", DataTable)
        assert table.row_count == 3
        assert app._loaded_ids == set(ids)


async def test_engine_respects_view_and_sort_state_directly(store, profile):
    # Exercises the data engine at its own interface (state attrs +
    # load_first_page) -- the actual Input/Select widgets that drive these
    # attributes interactively are wired up in later tasks.
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import DataTable

    h = profile.hash()
    a = make_vacancy(source_id="1", company="Zeta")
    b = make_vacancy(source_id="2", company="Alpha")
    store.upsert_vacancies([a, b])
    # Above the default JOBSEARCH_THRESHOLD (60) -- load_config()'s default
    # is what JobsearchApp() uses here since no config override is passed.
    store.save_rating(a.id, h, "f", score=70)
    store.save_rating(b.id, h, "f", score=70)

    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.sort_by, app.sort_dir = "company", "asc"
        app.load_first_page()
        await pilot.pause()
        table = app.query_one("#table", DataTable)
        assert app._loaded_ids == {b.id, a.id}
        assert table.row_count == 2


async def test_no_profile_screen_shown_when_profile_path_missing(tmp_path, store):
    from jobsearch.config import Config
    from jobsearch.tui.app import JobsearchApp
    from jobsearch.tui.screens import NoProfileScreen

    config = Config(profile_path=tmp_path / "missing-profile.json", db_path=tmp_path / "db")
    app = JobsearchApp(config=config, store=store)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, NoProfileScreen)


async def test_no_profile_screen_shown_when_profile_has_zero_ratings(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from jobsearch.tui.screens import NoProfileScreen

    # profile exists (passed directly) but nothing has been rated for it yet
    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, NoProfileScreen)


async def test_list_shown_normally_when_profile_has_ratings(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from jobsearch.tui.screens import NoProfileScreen

    v = make_vacancy()
    store.upsert_vacancies([v])
    store.save_rating(v.id, profile.hash(), "f", score=80)

    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not isinstance(app.screen, NoProfileScreen)
