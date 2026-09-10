from conftest import make_vacancy


def seed_active(store, profile, n=3):
    """n vacancies, scores descending (99, 98, 97, ...), all active/rated.
    Scores stay well above the default JOBSEARCH_THRESHOLD (60) even for a
    larger n, so tests using JobsearchApp() with no config override (and
    thus load_config()'s real default threshold) don't lose rows to it."""
    h = profile.hash()
    ids = []
    for i in range(n):
        v = make_vacancy(source_id=str(i), title=f"Role {i}", company=f"Company {i}")
        store.upsert_vacancies([v])
        store.save_rating(v.id, h, "f", score=99 - i)
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


async def test_table_has_keyboard_focus_right_after_mount(store, profile):
    # Real bug report: outside the headless test driver, nothing ends up
    # with actual Textual focus at startup -- the cursor row renders in its
    # blurred/gray style (DataTable's :focus CSS never applies) and hotkeys
    # need a mouse click first. Relying on Textual's implicit AUTO_FOCUS
    # heuristic isn't reliable across environments, so on_mount now grabs
    # focus explicitly.
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import DataTable

    seed_active(store, profile, n=1)
    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#table", DataTable)
        assert table.has_focus


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


async def test_search_input_gets_a_usable_width_not_just_its_border(store, profile):
    # Real bug: the trailing "remote only" Static label has no width rule,
    # so it greedily claims the filter bar's remaining space and the
    # search Input's `1fr` collapses to its intrinsic minimum (border +
    # padding only, zero columns of actual content) -- nothing typed, or
    # even the placeholder, can ever render in that.
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import Input

    seed_active(store, profile, n=1)
    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        search = app.query_one("#search", Input)
        assert search.region.width >= 30


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


async def test_mark_seen_updates_status_in_row_cache_and_keeps_row_in_active_view(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import DataTable

    ids = seed_active(store, profile)
    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        table = app.query_one("#table", DataTable)
        assert table.row_count == 3  # "seen" still belongs in the active view
        assert app._row_cache[ids[0]]["status"] == "seen"
        assert store.get_vacancy(ids[0]) is not None


async def test_mark_dismissed_removes_row_from_active_view(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import DataTable

    ids = seed_active(store, profile)
    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        table = app.query_one("#table", DataTable)
        assert table.row_count == 2
        assert ids[0] not in app._loaded_ids
        assert store.get_vacancy(ids[0]) is not None  # hidden from this view, not deleted


async def test_mark_dismissed_while_viewing_dismissed_keeps_the_row(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import DataTable

    ids = seed_active(store, profile, n=1)
    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view = "dismissed"
        app.load_first_page()
        await pilot.pause()
        # nothing dismissed yet
        assert app.query_one("#table", DataTable).row_count == 0

        store.set_status(ids[0], "dismissed")
        app.load_first_page()
        await pilot.pause()
        await pilot.press("d")  # re-marking dismissed while already viewing dismissed
        await pilot.pause()
        assert app.query_one("#table", DataTable).row_count == 1


async def test_scrolling_near_the_bottom_loads_the_next_page(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import DataTable

    ids = seed_active(store, profile, n=8)  # more than one small page
    app = JobsearchApp(store=store, candidate=profile)
    app.page_size = 3
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#table", DataTable)
        # page_size (3) is smaller than the 5-row lookahead, so the initial
        # auto-highlight of row 0 already triggers one prefetch -- not all
        # 8 rows are loaded yet, but more than just the first page is.
        loaded_after_mount = table.row_count
        assert 3 <= loaded_after_mount < 8

        for _ in range(6):  # walk the cursor down until every row is loaded
            await pilot.press("down")
        await pilot.pause()

        assert table.row_count == 8
        assert app._loaded_ids == set(ids)


async def test_load_more_never_repeats_or_skips_rows_across_batches(store, profile):
    from jobsearch.tui.app import JobsearchApp

    ids = seed_active(store, profile, n=8)
    app = JobsearchApp(store=store, candidate=profile)
    app.page_size = 3
    async with app.run_test() as pilot:
        await pilot.pause()
        while len(app._loaded_ids) < len(ids):
            app.load_more()
            await pilot.pause()
        assert app._loaded_ids == set(ids)


async def test_typing_in_search_filters_and_resets_pagination(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import DataTable, Input

    h = profile.hash()
    a = make_vacancy(source_id="1", title="Backend Engineer")
    b = make_vacancy(source_id="2", title="Frontend Engineer")
    store.upsert_vacancies([a, b])
    store.save_rating(a.id, h, "f", score=80)
    store.save_rating(b.id, h, "f", score=80)

    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#table", DataTable).row_count == 2

        search = app.query_one("#search", Input)
        search.focus()
        for ch in "backend":
            await pilot.press(ch)
        await pilot.pause()

        table = app.query_one("#table", DataTable)
        assert table.row_count == 1
        assert app._loaded_ids == {a.id}


async def test_slash_focuses_the_search_input(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import Input

    seed_active(store, profile, n=1)
    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("slash")
        await pilot.pause()
        assert app.query_one("#search", Input).has_focus


async def test_changing_sort_select_resorts_and_resets_pagination(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import DataTable, Select

    h = profile.hash()
    a = make_vacancy(source_id="1", company="Zeta")
    b = make_vacancy(source_id="2", company="Alpha")
    store.upsert_vacancies([a, b])
    store.save_rating(a.id, h, "f", score=70)
    store.save_rating(b.id, h, "f", score=90)

    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#sort", Select).value = "company"
        await pilot.pause()
        assert app.sort_by == "company"
        table = app.query_one("#table", DataTable)
        assert table.row_count == 2


async def test_sort_direction_button_flips_and_reloads(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import Button

    seed_active(store, profile, n=2)
    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.sort_dir == "desc"
        await pilot.click("#sort-dir")
        await pilot.pause()
        assert app.sort_dir == "asc"


async def test_remote_only_switch_filters_and_resets_pagination(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import DataTable, Switch

    h = profile.hash()
    remote = make_vacancy(source_id="1", title="Remote Role", is_remote=True)
    onsite = make_vacancy(source_id="2", title="Onsite Role", is_remote=False)
    store.upsert_vacancies([remote, onsite])
    store.save_rating(remote.id, h, "f", score=80)
    store.save_rating(onsite.id, h, "f", score=80)

    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#table", DataTable).row_count == 2

        app.query_one("#remote-only", Switch).value = True
        await pilot.pause()

        table = app.query_one("#table", DataTable)
        assert table.row_count == 1
        assert app._loaded_ids == {remote.id}


async def test_cycle_view_moves_through_active_dismissed_closed_duplicates(store, profile):
    from jobsearch.tui.app import JobsearchApp

    app = JobsearchApp(store=store, candidate=profile)
    seed_active(store, profile, n=1)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.view == "active"
        await pilot.press("c")
        await pilot.pause()
        assert app.view == "dismissed"
        await pilot.press("c")
        await pilot.pause()
        assert app.view == "closed"
        await pilot.press("c")
        await pilot.pause()
        assert app.view == "duplicates"
        await pilot.press("c")
        await pilot.pause()
        assert app.view == "active"


async def test_refresh_reloads_the_current_view_with_a_fresh_cursor(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import DataTable

    ids = seed_active(store, profile, n=8)
    app = JobsearchApp(store=store, candidate=profile)
    app.page_size = 2
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#table", DataTable)

        for _ in range(6):  # walk the cursor down to load most/all of it
            await pilot.press("down")
        await pilot.pause()
        scrolled_count = table.row_count
        assert scrolled_count > 2

        await pilot.press("r")
        await pilot.pause()

        # Fewer rows than the scrolled-to state, and exactly a fresh prefix
        # of the sorted list -- proof this re-queried from a fresh cursor
        # rather than just continuing pagination from where it left off.
        after_refresh = table.row_count
        assert after_refresh < scrolled_count
        assert app._loaded_ids == set(ids[:after_refresh])


async def test_duplicates_view_shows_marked_pairs_with_their_own_columns(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import DataTable

    dup = make_vacancy(source_id="1", title="Dup Title")
    canonical = make_vacancy(source_id="2", title="Canonical Title")
    store.upsert_vacancies([dup, canonical])
    store.save_rating(canonical.id, profile.hash(), "f", score=80)
    store.mark_duplicate(dup.id, canonical.id)

    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view = "duplicates"
        app.load_first_page()
        await pilot.pause()
        table = app.query_one("#table", DataTable)
        assert table.row_count == 1
        assert dup.id in app._loaded_ids


async def test_enter_opens_detail_screen_with_matched_and_missing_skills(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from jobsearch.tui.screens import DetailScreen

    h = profile.hash()
    v = make_vacancy(title="Backend Role")
    store.upsert_vacancies([v])
    store.save_rating(
        v.id, h, "f", score=85, verdict="strong",
        matched_skills=[{"skill": "Python", "evidence_cv": "5 years Python",
                          "evidence_vacancy": "Python required"}],
        missing_skills=["Kubernetes"],
    )

    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, DetailScreen)
        from textual.widgets import Static
        body = app.screen.query_one("#detail-body", Static).render()
        assert "Python" in str(body)
        assert "Kubernetes" in str(body)


async def test_escape_closes_detail_screen_without_reloading_the_list(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from jobsearch.tui.screens import DetailScreen
    from textual.widgets import DataTable

    seed_active(store, profile, n=2)
    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        loaded_before = set(app._loaded_ids)
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, DetailScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, DetailScreen)
        assert app._loaded_ids == loaded_before
        assert app.query_one("#table", DataTable).row_count == 2


async def test_interview_accepted_rejected_bindings_set_status(store, profile):
    from jobsearch.tui.app import JobsearchApp

    ids = seed_active(store, profile, n=1)
    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        for key, status in (("i", "interview"), ("o", "accepted"), ("x", "rejected")):
            await pilot.press(key)
            await pilot.pause()
            assert app._row_cache[ids[0]]["status"] == status
            row = store.conn.execute(
                "SELECT status FROM vacancy_status WHERE vacancy_id=?", (ids[0],)
            ).fetchone()
            assert row["status"] == status


async def test_edit_reason_saves_via_prompt_editor(store, profile, monkeypatch):
    from jobsearch.tui.app import JobsearchApp

    ids = seed_active(store, profile, n=1)
    store.set_status(ids[0], "rejected")
    monkeypatch.setattr(JobsearchApp, "_prompt_editor", lambda self, current: "salary too low")

    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._selected_id = ids[0]
        await pilot.press("n")
        await pilot.pause()

        row = store.conn.execute(
            "SELECT reason FROM vacancy_status WHERE vacancy_id=?", (ids[0],)
        ).fetchone()
        assert row["reason"] == "salary too low"


async def test_edit_reason_notifies_on_error_when_no_status_set_yet(store, profile, monkeypatch):
    from jobsearch.tui.app import JobsearchApp

    ids = seed_active(store, profile, n=1)
    monkeypatch.setattr(JobsearchApp, "_prompt_editor", lambda self, current: "too early")

    notifications = []
    app = JobsearchApp(store=store, candidate=profile)
    app.notify = lambda message, **kw: notifications.append((message, kw.get("severity")))
    async with app.run_test() as pilot:
        await pilot.pause()
        app._selected_id = ids[0]
        await pilot.press("n")
        await pilot.pause()

        assert notifications and notifications[0][1] == "error"


async def test_status_and_reason_keys_stay_visible_in_an_80_column_footer(store, profile):
    # A real bug: adding the interview/accepted/rejected/reason bindings
    # pushed the footer past 80 columns (a common terminal width, and
    # Docker's default when no TTY size is negotiated), silently scrolling
    # "Reason" and others out of view -- present in BINDINGS, invisible in
    # practice. FooterKey is private API, but it's the only way to check
    # what's actually rendered on screen rather than just declared.
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets._footer import FooterKey

    seed_active(store, profile, n=1)
    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        visible = {
            key.key for key in app.screen.query(FooterKey)
            if key.region.x >= 0 and key.region.x + key.region.width <= 80
        }
        assert {"s", "d", "a", "i", "o", "x", "n"} <= visible


async def test_status_action_works_from_detail_screen(store, profile):
    from jobsearch.tui.app import JobsearchApp

    ids = seed_active(store, profile, n=1)
    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        assert store.get_vacancy(ids[0])["id"] == ids[0]
        row = store.conn.execute(
            "SELECT status FROM vacancy_status WHERE vacancy_id=?", (ids[0],)
        ).fetchone()
        assert row["status"] == "dismissed"


async def test_interview_binding_works_from_detail_screen(store, profile):
    from jobsearch.tui.app import JobsearchApp

    ids = seed_active(store, profile, n=1)
    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("o")
        await pilot.pause()
        row = store.conn.execute(
            "SELECT status FROM vacancy_status WHERE vacancy_id=?", (ids[0],)
        ).fetchone()
        assert row["status"] == "accepted"


async def test_status_and_reason_keys_stay_visible_in_detail_screen_footer(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets._footer import FooterKey

    seed_active(store, profile, n=1)
    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        visible = {
            key.key for key in app.screen.query(FooterKey)
            if key.region.x >= 0 and key.region.x + key.region.width <= 80
        }
        assert {"s", "d", "a", "i", "o", "x", "n"} <= visible


async def test_m_opens_duplicate_picker_modal(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from jobsearch.tui.duplicate_modal import DuplicatePickerModal

    seed_active(store, profile, n=2)
    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("m")
        await pilot.pause()
        assert isinstance(app.screen, DuplicatePickerModal)


async def test_duplicate_picker_search_finds_any_vacancy_not_just_loaded_ones(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from jobsearch.tui.duplicate_modal import DuplicatePickerModal
    from textual.widgets import Input, ListView

    seed_active(store, profile, n=1)
    # a vacancy that's unrated (never loaded into the main list) but should
    # still be findable as a canonical target
    unrated = make_vacancy(source_id="unrated", title="Unrated Canonical Role")
    store.upsert_vacancies([unrated])

    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("m")
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, DuplicatePickerModal)
        search = modal.query_one("#dup-search", Input)
        search.focus()
        for ch in "unrated":
            await pilot.press(ch)
        await pilot.pause()
        results = modal.query_one("#dup-results", ListView)
        assert len(results.children) == 1


async def test_confirming_duplicate_pick_hides_the_row_and_writes_the_mark(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import DataTable, Input, ListView, Button

    h = profile.hash()
    dup = make_vacancy(source_id="1", title="Dup Role")
    canonical = make_vacancy(source_id="2", title="Canonical Role")
    store.upsert_vacancies([dup, canonical])
    store.save_rating(dup.id, h, "f", score=80)
    store.save_rating(canonical.id, h, "f", score=75)

    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#table", DataTable)
        table.move_cursor(row=0)  # dup sorts first: score 80 > canonical's 75
        await pilot.pause()
        assert app._selected_id == dup.id

        await pilot.press("m")
        await pilot.pause()
        modal = app.screen
        search = modal.query_one("#dup-search", Input)
        search.focus()
        for ch in "canonical":
            await pilot.press(ch)
        await pilot.pause()
        results = modal.query_one("#dup-results", ListView)
        results.focus()
        results.index = 0
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        confirm_button = modal.query_one("#dup-confirm-btn", Button)
        assert not confirm_button.disabled
        await pilot.click("#dup-confirm-btn")
        await pilot.pause()

        assert dup.id not in app._loaded_ids
        row = store.conn.execute(
            "SELECT canonical_id FROM vacancy_duplicates WHERE vacancy_id=?", (dup.id,)
        ).fetchone()
        assert row["canonical_id"] == canonical.id


async def test_u_unmarks_duplicate_in_duplicates_view(store, profile):
    from jobsearch.tui.app import JobsearchApp
    from textual.widgets import DataTable

    dup = make_vacancy(source_id="1", title="Dup Role")
    canonical = make_vacancy(source_id="2", title="Canonical Role")
    store.upsert_vacancies([dup, canonical])
    store.save_rating(dup.id, profile.hash(), "f", score=80)
    store.mark_duplicate(dup.id, canonical.id)

    app = JobsearchApp(store=store, candidate=profile)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view = "duplicates"
        app.load_first_page()
        await pilot.pause()
        table = app.query_one("#table", DataTable)
        assert table.row_count == 1
        app._selected_id = dup.id

        await pilot.press("u")
        await pilot.pause()

        assert table.row_count == 0
        row = store.conn.execute(
            "SELECT * FROM vacancy_duplicates WHERE vacancy_id=?", (dup.id,)
        ).fetchone()
        assert row is None
        assert dup.id in [r["id"] for r in store.ranked_page(profile.hash(), 0)]


async def test_edit_description_updates_store_and_rerates_in_background(store, profile, monkeypatch):
    from jobsearch.tui.app import JobsearchApp
    from jobsearch.matching.llm.fake import FakeLLM

    v = make_vacancy(description="short snippet")
    store.upsert_vacancies([v])
    store.save_rating(v.id, profile.hash(), "fake", score=80)

    monkeypatch.setattr(
        JobsearchApp, "_prompt_editor", lambda self, current: "full real description"
    )
    llm = FakeLLM(responses=[{
        "score": 77, "hard_blockers": [], "matched_skills": [],
        "missing_skills": [], "summary_fit": "", "summary_gap": "",
    }])

    app = JobsearchApp(store=store, candidate=profile, llm=llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._selected_id = v.id
        await pilot.press("e")
        await pilot.pause(0.2)  # background thread worker needs a beat

        assert store.get_vacancy(v.id)["description"] == "full real description"
        assert app._row_cache[v.id]["score"] == 77


async def test_edit_description_shows_llm_hint_notification_on_failure(store, profile, monkeypatch):
    from jobsearch.tui.app import JobsearchApp
    from jobsearch.matching.llm.base import LLMError

    v = make_vacancy()
    store.upsert_vacancies([v])
    store.save_rating(v.id, profile.hash(), "fake", score=80)

    monkeypatch.setattr(JobsearchApp, "_prompt_editor", lambda self, current: "new text")

    class FailingLLM:
        name = "fake"

        def complete_json(self, *a, **kw):
            raise LLMError("boom")

        def embed(self, *a, **kw):
            return []

    notifications = []
    app = JobsearchApp(store=store, candidate=profile, llm=FailingLLM())
    app.notify = lambda message, **kw: notifications.append((message, kw.get("severity")))
    async with app.run_test() as pilot:
        await pilot.pause()
        app._selected_id = v.id
        await pilot.press("e")
        await pilot.pause(0.2)

        assert any(sev == "error" for _, sev in notifications)
        assert store.get_vacancy(v.id)["description"] == "new text"  # description saved even if re-rate fails
