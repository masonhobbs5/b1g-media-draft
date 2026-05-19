"""
scripts/run_dashboard.py
────────────────────────
Streamlit dashboard for 2026 Big Ten simulation outputs.

Reads exported artifact files (no live recomputation) and displays:
  - Expected team records (Artifact 1)
  - Predicted viewership by game (Artifact 2)
  - Broadcaster draft assignments (Artifact 3)
  - Weekly rankings with top games per week

Usage:
    streamlit run scripts/run_dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings

OUTPUT_DIR = settings.OUTPUT_DIR


# ── Data loading (cached) ────────────────────────────────────────────────────


@st.cache_data
def load_viewership() -> pd.DataFrame:
    with open(OUTPUT_DIR / "expected_viewership.json") as f:
        data = json.load(f)
    return pd.DataFrame(data)


@st.cache_data
def load_draft() -> pd.DataFrame:
    with open(OUTPUT_DIR / "draft_assignments.json") as f:
        data = json.load(f)
    return pd.DataFrame(data)


@st.cache_data
def load_weekly_viewers() -> dict:
    with open(OUTPUT_DIR / "draft_weekly_viewers.json") as f:
        return json.load(f)


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="2026 B1G Simulation",
    page_icon="🏈",
    layout="wide",
)

st.title("🏈 2026 Big Ten Football Simulation")
st.caption("Results from Monte Carlo simulation of the full 2026 B1G schedule")

# ── Sidebar ──────────────────────────────────────────────────────────────────

page = st.sidebar.radio(
    "Navigate",
    ["Viewership Predictions", "Broadcaster Draft", "Week Draft Board", "Draft Facilitator"],
)

# ── Page: Viewership Predictions ─────────────────────────────────────────────

if page == "Viewership Predictions":
    st.header("Expected Game Viewership")
    st.markdown("Ridge regression predictions with 90% confidence intervals.")

    df = load_viewership()

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        conf_only = st.checkbox("Conference games only", value=False)
    with col2:
        min_viewers = st.slider(
            "Minimum predicted viewers (M)", 0.0, 10.0, 0.0, 0.5
        )

    filtered = df.copy()
    if conf_only:
        filtered = filtered[filtered["is_conference_game"]]
    filtered = filtered[filtered["predicted_viewers_millions"] >= min_viewers]

    # Display table
    display_cols = [
        "week", "home_team", "away_team", "predicted_viewers_millions",
        "lower_bound_millions", "upper_bound_millions",
        "is_conference_game", "is_rivalry",
    ]
    show_df = filtered[display_cols].copy()
    show_df.columns = [
        "Week", "Home", "Away", "Pred (M)", "Lower (M)", "Upper (M)",
        "Conf", "Rivalry",
    ]
    show_df = show_df.sort_values("Pred (M)", ascending=False)

    st.dataframe(show_df, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(show_df)} of {len(df)} games")

    # Top 10 chart
    st.subheader("Top 10 Games by Expected Viewership")
    top10 = df.nlargest(10, "predicted_viewers_millions").copy()
    top10["matchup"] = top10["away_team"] + " @ " + top10["home_team"]
    chart_data = top10.set_index("matchup")[["predicted_viewers_millions"]]
    chart_data.columns = ["Viewers (M)"]
    st.bar_chart(chart_data.sort_values("Viewers (M)"), horizontal=True)

# ── Page: Broadcaster Draft ──────────────────────────────────────────────────

elif page == "Broadcaster Draft":
    st.header("📺 Broadcaster Draft Assignments")
    st.markdown("Week-by-week game selections — navigate between weeks to see each network's picks.")

    from config.constants import TEAM_META

    df = load_draft()
    weekly = load_weekly_viewers()

    # ── Network display (styled badges — external SVG logos are unreliable) ──
    NETWORK_COLORS = {
        "FOX": ("#003366", "#ffffff"),
        "CBS": ("#1a1a6e", "#ffffff"),
        "NBC": ("#000000", "#ffffff"),
    }

    def network_badge(net: str) -> str:
        bg, fg = NETWORK_COLORS.get(net, ("#888", "#fff"))
        return (
            f"<div style='display:inline-block; background:{bg}; color:{fg}; "
            f"font-weight:bold; font-size:1.1em; padding:6px 16px; "
            f"border-radius:6px; letter-spacing:2px;'>{net}</div>"
        )

    # ── ESPN CDN logo lookup (B1G teams from TEAM_META + non-conference opponents) ──
    NON_B1G_ESPN_IDS: dict[str, int] = {
        "Akron": 2006,
        "Ball State": 2050,
        "Boise State": 68,
        "Bowling Green": 189,
        "Buffalo": 2084,
        "Colorado": 38,
        "Duke": 150,
        "Eastern Illinois": 2173,
        "Eastern Michigan": 2199,
        "Eastern Washington": 331,
        "Fresno State": 278,
        "Howard": 2305,
        "Indiana State": 318,
        "Iowa State": 66,
        "Kent State": 2309,
        "Louisiana": 309,
        "Marshall": 276,
        "Massachusetts": 113,
        "Mississippi State": 344,
        "Nevada": 2440,
        "North Dakota": 2446,
        "North Texas": 249,
        "Northern Illinois": 2459,
        "Northern Iowa": 2460,
        "Ohio": 195,
        "Oklahoma": 201,
        "Portland State": 2490,
        "San Diego State": 21,
        "San José State": 23,
        "South Dakota State": 2571,
        "Southern Illinois": 79,
        "Toledo": 2649,
        "UAB": 2629,
        "UTEP": 2638,
        "Utah State": 328,
        "Virginia Tech": 259,
        "Wake Forest": 154,
        "Washington State": 265,
        "Western Illinois": 2710,
        "Western Kentucky": 98,
        "Western Michigan": 2711,
    }

    def team_logo_url(team_name: str) -> str:
        espn_id = TEAM_META.get(team_name, {}).get("espn_id")
        if not espn_id:
            espn_id = NON_B1G_ESPN_IDS.get(team_name)
        if espn_id:
            return f"https://a.espncdn.com/i/teamlogos/ncaa/500/{espn_id}.png"
        return ""

    # ── Load predicted schedule from simulation output ──
    all_weeks = sorted(df["week"].unique())
    nd_weeks = set(settings.NBC_NOTRE_DAME_WEEKS)

    # Build lookup from predicted_schedule (deterministic predictions from simulator)
    predicted_schedule = weekly.get("predicted_schedule", [])
    pred_by_week: dict[int, dict] = {w["week"]: w for w in predicted_schedule}

    # Build Notre Dame game lookup from viewership data (ND games not in draft pool)
    vw_df_draft = load_viewership()
    nd_game_by_week: dict[int, dict] = {}
    for _, row in vw_df_draft.iterrows():
        if "Notre Dame" in (row["home_team"], row["away_team"]):
            nd_game_by_week[int(row["week"])] = row.to_dict()

    def get_week_assignments(week_num: int) -> list[dict]:
        """Get the network game assignments for a week from predicted schedule."""
        pred = pred_by_week.get(week_num, {})
        games = pred.get("games", [])

        assignments = []
        for pick_pos, game in enumerate(games, 1):
            assignments.append({
                "pick_order": pick_pos,
                "network": game["network"],
                "game_id": game.get("game_id", ""),
                "home_team": game.get("home_team", "Unknown"),
                "away_team": game.get("away_team", "Unknown"),
                "viewers": game.get("predicted_viewers", 0.0),
                "probability": game.get("probability", 0.0),
                "is_nd": False,
            })

        # In ND weeks, append NBC's pre-committed Notre Dame game
        if week_num in nd_weeks and week_num in nd_game_by_week:
            nd = nd_game_by_week[week_num]
            assignments.append({
                "pick_order": len(assignments) + 1,
                "network": "NBC",
                "game_id": nd.get("game_id", ""),
                "home_team": nd["home_team"],
                "away_team": nd["away_team"],
                "viewers": nd["predicted_viewers_millions"],
                "probability": 1.0,
                "is_nd": True,
            })

        return assignments

    # ── Season totals scoreboard ──
    totals = weekly.get("predicted_season_totals", weekly.get("season_totals", {}))
    score_cols = st.columns(3)
    for i, net in enumerate(["FOX", "CBS", "NBC"]):
        with score_cols[i]:
            st.markdown(network_badge(net), unsafe_allow_html=True)
            st.metric("", f"{totals.get(net, 0):.1f}M viewers")

    st.divider()

    # ── Week navigation ──
    if "draft_week_idx" not in st.session_state:
        st.session_state.draft_week_idx = 0

    nav_cols = st.columns([1, 6, 1])
    with nav_cols[0]:
        if st.button("◀ Prev", disabled=st.session_state.draft_week_idx == 0):
            st.session_state.draft_week_idx -= 1
            st.rerun()
    with nav_cols[2]:
        if st.button("Next ▶", disabled=st.session_state.draft_week_idx >= len(all_weeks) - 1):
            st.session_state.draft_week_idx += 1
            st.rerun()

    current_week = all_weeks[st.session_state.draft_week_idx]

    # ── Week header ──
    pred = pred_by_week.get(current_week, {})
    pick_num = pred.get("draft_pick", "—")
    drafting_net = pred.get("drafted_by", "—")
    is_nd = current_week in nd_weeks

    with nav_cols[1]:
        nd_badge = " 🍀 Notre Dame Week" if is_nd else ""
        st.markdown(
            f"### Week {current_week}{nd_badge}\n"
            f"**Drafted #{pick_num}** by **{drafting_net}**"
        )

    st.divider()

    # ── Game cards ──
    assignments = get_week_assignments(current_week)

    if not assignments:
        st.info("No games assigned this week.")
    else:
        for game in assignments:
            pick_label = {1: "1st Pick", 2: "2nd Pick", 3: "3rd Pick"}.get(
                game["pick_order"], f"Pick #{game['pick_order']}"
            )
            if game.get("is_nd"):
                pick_label = "Pre-committed"
            net = game["network"]

            # Game card
            with st.container():
                card_cols = st.columns([1, 1, 3, 1, 1, 2])

                # Away team logo
                with card_cols[0]:
                    away_logo = team_logo_url(game["away_team"])
                    if away_logo:
                        st.markdown(
                            f"<div style='background:white; border-radius:6px; "
                            f"padding:4px; display:inline-block;'>"
                            f"<img src='{away_logo}' width='50'></div>",
                            unsafe_allow_html=True,
                        )
                    st.caption(game["away_team"])

                # "at" separator
                with card_cols[1]:
                    st.markdown("<div style='text-align:center; padding-top:15px; font-size:1.2em;'><b>@</b></div>", unsafe_allow_html=True)

                # Home team logo
                with card_cols[2]:
                    home_logo = team_logo_url(game["home_team"])
                    if home_logo:
                        st.markdown(
                            f"<div style='background:white; border-radius:6px; "
                            f"padding:4px; display:inline-block;'>"
                            f"<img src='{home_logo}' width='50'></div>",
                            unsafe_allow_html=True,
                        )
                    st.caption(game["home_team"])

                # Network badge
                with card_cols[3]:
                    st.markdown(
                        "<div style='padding-top:10px;'>" + network_badge(net) + "</div>",
                        unsafe_allow_html=True,
                    )

                # Viewership
                with card_cols[4]:
                    st.metric("Viewers", f"{game['viewers']:.2f}M")

                # Pick order badge
                with card_cols[5]:
                    conf = game.get("probability", 0)
                    conf_str = f"<br><small>{conf:.0%} conf.</small>" if conf and not game.get("is_nd") else ""
                    st.markdown(
                        f"<div style='background-color:#f0f2f6; border-radius:8px; "
                        f"padding:8px 12px; text-align:center; margin-top:8px;'>"
                        f"<b>{pick_label}</b>{conf_str}</div>",
                        unsafe_allow_html=True,
                    )

    # ── Week mini-nav (dots) ──
    st.markdown(
        "<div style='text-align:center; margin-top:-10px;'>"
        + "".join(
            f"<span style='font-size:1.2em; margin:0 3px; "
            f"{'color:#1f77b4; font-weight:bold' if i == st.session_state.draft_week_idx else 'color:#ccc'}"
            f"'>●</span>"
            for i in range(len(all_weeks))
        )
        + "</div>",
        unsafe_allow_html=True,
    )

# ── Page: Week Draft Board ───────────────────────────────────────────────────

elif page == "Week Draft Board":
    st.header("Week Draft Board")
    st.markdown(
        "Weeks ranked by **draft value** — the gap in predicted viewership between "
        "the top two eligible games. Higher gap = more dominant #1 game = more "
        "attractive week to draft."
    )

    from config.constants import BIG_TEN_TEAMS as _B1G_TEAMS_WDB

    vw_df = load_viewership()

    # Exclude ineligible games:
    # 1. Notre Dame games (pre-committed to NBC)
    # 2. Games at non-B1G venues (e.g. Ohio State @ Texas)
    pool = vw_df[
        ~vw_df["home_team"].str.contains("Notre Dame", na=False)
        & ~vw_df["away_team"].str.contains("Notre Dame", na=False)
        & vw_df["home_team"].isin(_B1G_TEAMS_WDB)
    ].copy()

    nd_weeks = set(settings.NBC_NOTRE_DAME_WEEKS)

    # Build per-week summary
    week_rows = []
    for week, grp in pool.groupby("week"):
        top3 = grp.nlargest(3, "predicted_viewers_millions").reset_index(drop=True)
        v1 = top3.loc[0, "predicted_viewers_millions"] if len(top3) >= 1 else 0.0
        v2 = top3.loc[1, "predicted_viewers_millions"] if len(top3) >= 2 else 0.0
        draft_value = v1 - v2
        week_rows.append(
            {
                "week": int(week),
                "draft_value": draft_value,
                "top1_viewers": v1,
                "top2_viewers": v2,
                "nd_week": int(week) in nd_weeks,
                "top3": top3,
            }
        )

    week_rows.sort(key=lambda r: r["draft_value"], reverse=True)

    # ── Summary table ──
    st.subheader("All Weeks Ranked")
    summary_rows = []
    for rank, r in enumerate(week_rows, 1):
        top3 = r["top3"]
        games = []
        for _, g in top3.iterrows():
            games.append(f"{g['away_team']} @ {g['home_team']} ({g['predicted_viewers_millions']:.2f}M)")
        summary_rows.append(
            {
                "Rank": rank,
                "Week": r["week"],
                "Draft Value (M)": f"{r['draft_value']:.2f}",
                "#1 Game": games[0] if len(games) > 0 else "—",
                "#2 Game": games[1] if len(games) > 1 else "—",
                "#3 Game": games[2] if len(games) > 2 else "—",
                "ND Week": "★" if r["nd_week"] else "",
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.caption("★ = NBC pre-committed Notre Dame week (only 2 games drafted that week)")

    st.divider()

    # ── Draft value bar chart ──
    st.subheader("Draft Value by Week")
    chart_df = (
        pd.DataFrame({"Week": [r["week"] for r in week_rows],
                      "Draft Value (M)": [r["draft_value"] for r in week_rows]})
        .set_index("Week")
        .sort_index()
    )
    st.bar_chart(chart_df)

    st.divider()

    # ── Detailed week cards ──
    st.subheader("Week Details")
    for rank, r in enumerate(week_rows, 1):
        nd_label = " ★ Notre Dame Week" if r["nd_week"] else ""
        with st.expander(
            f"#{rank}  Week {r['week']}{nd_label}  —  "
            f"Draft Value: {r['draft_value']:.2f}M  "
            f"(top game: {r['top1_viewers']:.2f}M)"
        ):
            top3 = r["top3"]
            for i, (_, game) in enumerate(top3.iterrows(), 1):
                cols = st.columns([4, 2, 2])
                with cols[0]:
                    rivalry_badge = " 🏆" if game.get("is_rivalry") else ""
                    cfp_badge = " ⭐" if game.get("is_cfp_rematch") else ""
                    st.markdown(
                        f"**#{i}. {game['away_team']} @ {game['home_team']}**"
                        f"{rivalry_badge}{cfp_badge}"
                    )
                    st.caption(game.get("game_date", ""))
                with cols[1]:
                    st.metric(
                        "Projected Viewers",
                        f"{game['predicted_viewers_millions']:.2f}M",
                        help=(
                            f"90% CI: [{game['lower_bound_millions']:.1f}M"
                            f" – {game['upper_bound_millions']:.1f}M]"
                        ),
                    )
                with cols[2]:
                    if i == 1 and len(top3) > 1:
                        gap = (
                            game["predicted_viewers_millions"]
                            - top3.loc[top3.index[1], "predicted_viewers_millions"]
                        )
                        st.metric("Gap to #2", f"{gap:.2f}M")
                if i < len(top3):
                    st.divider()

# ── Page: Draft Facilitator ──────────────────────────────────────────────────

elif page == "Draft Facilitator":
    st.header("🏈 Network Draft Facilitator")
    st.markdown(
        "Facilitate the two-phase network draft interactively. "
        "**Phase 1**: Networks select their weeks per the draft order. "
        "**Phase 2**: From Week 1 onward, networks select games in the order "
        "they earned."
    )

    from config.constants import PACIFIC_TZ_TEAMS, BIG_TEN_TEAMS as _B1G_TEAMS_DF

    # ── Load data ──
    vw_df = load_viewership()
    # Exclude ineligible games:
    # 1. Notre Dame games (pre-committed to NBC)
    # 2. Games at non-B1G venues (e.g. Ohio State @ Texas)
    pool = vw_df[
        ~vw_df["home_team"].str.contains("Notre Dame", na=False)
        & ~vw_df["away_team"].str.contains("Notre Dame", na=False)
        & vw_df["home_team"].isin(_B1G_TEAMS_DF)
    ].copy()

    draft_order = settings.DRAFT_ORDER
    nd_weeks = set(settings.NBC_NOTRE_DAME_WEEKS)
    all_weeks = sorted(pool["week"].unique())

    # ── Session state initialization ──
    if "week_picks" not in st.session_state:
        st.session_state.week_picks = {}  # pick_index -> week_number
    if "game_picks" not in st.session_state:
        st.session_state.game_picks = {}  # (week, network) -> game_id
    if "phase" not in st.session_state:
        st.session_state.phase = 1

    week_picks: dict[int, int] = st.session_state.week_picks
    game_picks: dict[tuple[int, str], str] = st.session_state.game_picks

    # Reset button
    if st.sidebar.button("🔄 Reset Draft"):
        st.session_state.week_picks = {}
        st.session_state.game_picks = {}
        st.session_state.phase = 1
        st.rerun()

    # ── Helper: compute week draft value ──
    def get_week_values() -> dict[int, dict]:
        """Compute draft value and top games for each available week."""
        available_weeks = [w for w in all_weeks if w not in week_picks.values()]
        week_info = {}
        for w in all_weeks:
            wg = pool[pool["week"] == w].nlargest(5, "predicted_viewers_millions")
            top_games = []
            for _, g in wg.iterrows():
                top_games.append({
                    "game_id": g["game_id"],
                    "matchup": f"{g['away_team']} @ {g['home_team']}",
                    "viewers": g["predicted_viewers_millions"],
                    "home_team": g["home_team"],
                })
            v1 = top_games[0]["viewers"] if len(top_games) >= 1 else 0.0
            v2 = top_games[1]["viewers"] if len(top_games) >= 2 else 0.0
            week_info[w] = {
                "draft_value": v1 - v2,
                "top_game_viewers": v1,
                "top_games": top_games,
                "available": w in available_weeks,
                "nd_week": w in nd_weeks,
            }
        return week_info

    # ── Helper: get pick order for a week ──
    def get_game_pick_order(week: int) -> list[str]:
        """Return the network order for game selection within a week."""
        drafting_net = None
        for idx, picked_week in week_picks.items():
            if picked_week == week:
                drafting_net = draft_order[idx]
                break

        is_nd_week = week in nd_weeks
        if is_nd_week:
            # Only 2 games drafted, NBC excluded
            if drafting_net == "FOX":
                return ["FOX", "CBS"]
            elif drafting_net == "CBS":
                return ["CBS", "FOX"]
            else:
                return ["FOX", "CBS"]  # fallback
        else:
            if drafting_net == "FOX":
                return ["FOX", "CBS", "NBC"]
            elif drafting_net == "CBS":
                return ["CBS", "FOX", "NBC"]
            elif drafting_net == "NBC":
                return ["NBC", "FOX", "CBS"]
            else:
                return ["FOX", "CBS", "NBC"]

    # ── Scoreboard ──
    st.divider()
    score_cols = st.columns(3)
    network_totals = {"FOX": 0.0, "CBS": 0.0, "NBC": 0.0}
    for (w, net), gid in game_picks.items():
        row = pool[pool["game_id"] == gid]
        if not row.empty:
            network_totals[net] += row.iloc[0]["predicted_viewers_millions"]

    score_cols[0].metric("📺 FOX Total", f"{network_totals['FOX']:.2f}M")
    score_cols[1].metric("📺 CBS Total", f"{network_totals['CBS']:.2f}M")
    score_cols[2].metric("📺 NBC Total", f"{network_totals['NBC']:.2f}M")
    st.divider()

    # ── PHASE 1: Week Draft ──
    current_pick = len(week_picks)
    all_weeks_drafted = current_pick >= len(draft_order)

    if not all_weeks_drafted:
        st.session_state.phase = 1
    elif st.session_state.phase == 1:
        st.session_state.phase = 2

    if st.session_state.phase == 1:
        st.subheader("Phase 1: Week Selection")
        picking_network = draft_order[current_pick]
        st.info(
            f"**Pick #{current_pick + 1} of {len(draft_order)}** — "
            f"**{picking_network}** is on the clock"
        )

        # Show draft board — weeks ranked by value
        week_info = get_week_values()
        ranked = sorted(
            [(w, info) for w, info in week_info.items() if info["available"]],
            key=lambda x: x[1]["draft_value"],
            reverse=True,
        )

        st.markdown("#### Available Weeks (ranked by draft value)")
        # Table view
        board_rows = []
        for rank, (w, info) in enumerate(ranked, 1):
            nd_mark = "★" if info["nd_week"] else ""
            top_game = info["top_games"][0] if info["top_games"] else None
            fox_flag = (
                "⚠️ Top game ineligible"
                if picking_network == "FOX"
                and top_game
                and top_game["home_team"] in PACIFIC_TZ_TEAMS
                else ""
            )
            nbc_flag = (
                "⚠️ NBC pre-committed (ND)"
                if picking_network == "NBC" and info["nd_week"]
                else ""
            )
            top3_str = " | ".join(
                f"{g['matchup']} ({g['viewers']:.2f}M)"
                for g in info["top_games"][:3]
            )
            board_rows.append({
                "Rank": rank,
                "Week": w,
                "Draft Value": f"{info['draft_value']:.2f}M",
                "Top Game": f"{info['top_game_viewers']:.2f}M",
                "Top 3 Games": top3_str,
                "ND": nd_mark,
                "FOX Alert": fox_flag,
                "NBC Alert": nbc_flag,
            })
        st.dataframe(pd.DataFrame(board_rows), use_container_width=True, hide_index=True)

        # Selection
        available_weeks = [w for w, info in ranked]
        selected_week = st.selectbox(
            f"Select week for {picking_network}",
            available_weeks,
            format_func=lambda w: (
                f"Week {w} — Top: {week_info[w]['top_games'][0]['matchup']} "
                f"({week_info[w]['top_game_viewers']:.2f}M) "
                f"[Value: {week_info[w]['draft_value']:.2f}M]"
                + (" ★ ND" if week_info[w]["nd_week"] else "")
            ),
        )

        # Warn FOX if the top game of the selected week is on the West Coast
        if picking_network == "FOX" and selected_week in week_info:
            top = week_info[selected_week]["top_games"]
            if top and top[0]["home_team"] in PACIFIC_TZ_TEAMS:
                st.warning(
                    f"⚠️ The top game in Week {selected_week} — "
                    f"**{top[0]['matchup']}** ({top[0]['viewers']:.2f}M) — "
                    f"is a West Coast home game and **ineligible for FOX**. "
                    f"FOX's best available game will be #{2 if len(top) > 1 else 1} "
                    f"on the board."
                )

        # Warn NBC if the selected week is a Notre Dame pre-committed week
        if picking_network == "NBC" and week_info.get(selected_week, {}).get("nd_week"):
            st.warning(
                f"⚠️ Week {selected_week} is a **Notre Dame pre-committed week**. "
                f"NBC's primetime slot is already committed to the Notre Dame game — "
                f"NBC will **not pick a B1G game** in this week."
            )

        if st.button(f"✅ Confirm: {picking_network} selects Week {selected_week}"):
            st.session_state.week_picks[current_pick] = selected_week
            st.rerun()

        # Show picks so far
        if week_picks:
            st.markdown("#### Picks Made")
            pick_rows = []
            for idx in sorted(week_picks.keys()):
                w = week_picks[idx]
                net = draft_order[idx]
                top = week_info[w]["top_games"][0] if week_info[w]["top_games"] else None
                pick_rows.append({
                    "Pick #": idx + 1,
                    "Network": net,
                    "Week": w,
                    "Top Game": top["matchup"] if top else "—",
                    "Top Viewers": f"{top['viewers']:.2f}M" if top else "—",
                })
            st.dataframe(pd.DataFrame(pick_rows), use_container_width=True, hide_index=True)

    # ── PHASE 2: Game Selection ──
    elif st.session_state.phase == 2:
        st.subheader("Phase 2: Game Selection")
        st.markdown(
            "Select games week by week. Within each week, networks pick in the "
            "order determined by who drafted that week."
        )

        # Determine which (week, network) still need picks
        weeks_in_order = sorted(week_picks.values())
        pending_picks = []
        for w in weeks_in_order:
            pick_order = get_game_pick_order(w)
            for net in pick_order:
                if (w, net) not in game_picks:
                    pending_picks.append((w, net))

        if not pending_picks:
            st.success("🎉 Draft complete! All games selected.")
            st.balloons()
        else:
            current_week, current_net = pending_picks[0]
            drafting_net_for_week = None
            for idx, picked_week in week_picks.items():
                if picked_week == current_week:
                    drafting_net_for_week = draft_order[idx]
                    break

            is_nd_week = current_week in nd_weeks
            st.info(
                f"**Week {current_week}** — **{current_net}** selecting "
                f"({'🍀 Notre Dame week — NBC excluded' if is_nd_week else ''})"
            )

            # Already picked games this week
            picked_this_week = {
                gid for (w, _), gid in game_picks.items() if w == current_week
            }

            # Available games for this network
            week_games = pool[pool["week"] == current_week].copy()
            available = week_games[~week_games["game_id"].isin(picked_this_week)]

            # Warn FOX if the top game is on the west coast
            if current_net == "FOX" and not available.empty:
                top_game = available.sort_values(
                    "predicted_viewers_millions", ascending=False
                ).iloc[0]
                if top_game["home_team"] in PACIFIC_TZ_TEAMS:
                    st.warning(
                        f"⚠️ The top game this week — "
                        f"**{top_game['away_team']} @ {top_game['home_team']}** "
                        f"({top_game['predicted_viewers_millions']:.2f}M) — "
                        f"is a West Coast home game and **ineligible for FOX** "
                        f"(noon kickoff restriction)."
                    )

            # Apply FOX eligibility filter
            if current_net == "FOX":
                available = available[
                    ~available["home_team"].isin(PACIFIC_TZ_TEAMS)
                ]

            available = available.sort_values(
                "predicted_viewers_millions", ascending=False
            )

            if available.empty:
                st.warning(f"No eligible games left for {current_net} in Week {current_week}.")
                if st.button("⏭️ Skip this pick"):
                    st.session_state.game_picks[(current_week, current_net)] = "__skipped__"
                    st.rerun()
            else:
                # Show available games
                st.markdown(f"**Available games for {current_net}:**")
                game_options = []
                for _, g in available.iterrows():
                    game_options.append(g["game_id"])

                display_df = available[
                    ["home_team", "away_team", "predicted_viewers_millions",
                     "is_conference_game", "is_rivalry"]
                ].copy()
                display_df.columns = ["Home", "Away", "Viewers (M)", "Conf", "Rivalry"]
                display_df["Viewers (M)"] = display_df["Viewers (M)"].apply(lambda x: f"{x:.2f}")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                selected_game = st.selectbox(
                    f"Select game for {current_net}",
                    game_options,
                    format_func=lambda gid: (
                        f"{available[available['game_id'] == gid].iloc[0]['away_team']} "
                        f"@ {available[available['game_id'] == gid].iloc[0]['home_team']} "
                        f"({available[available['game_id'] == gid].iloc[0]['predicted_viewers_millions']:.2f}M)"
                    ),
                )

                if st.button(
                    f"✅ Confirm: {current_net} selects "
                    f"{available[available['game_id'] == selected_game].iloc[0]['away_team']} @ "
                    f"{available[available['game_id'] == selected_game].iloc[0]['home_team']}"
                ):
                    st.session_state.game_picks[(current_week, current_net)] = selected_game
                    st.rerun()

        # Show all game picks so far
        if game_picks:
            st.divider()
            st.markdown("#### Game Selections")
            pick_rows = []
            for (w, net), gid in sorted(game_picks.items(), key=lambda x: (x[0][0], x[0][1])):
                if gid == "__skipped__":
                    pick_rows.append({"Week": w, "Network": net, "Matchup": "— skipped —", "Viewers (M)": "—"})
                else:
                    row = pool[pool["game_id"] == gid]
                    if not row.empty:
                        r = row.iloc[0]
                        pick_rows.append({
                            "Week": w,
                            "Network": net,
                            "Matchup": f"{r['away_team']} @ {r['home_team']}",
                            "Viewers (M)": f"{r['predicted_viewers_millions']:.2f}",
                        })
            st.dataframe(pd.DataFrame(pick_rows), use_container_width=True, hide_index=True)
