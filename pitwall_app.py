import os
import time

import fastf1
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


CACHE_DIR = "f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

sns.set_style("darkgrid")
plt.rcParams["figure.facecolor"] = "#111111"
plt.rcParams["axes.facecolor"] = "#111111"
plt.rcParams["axes.edgecolor"] = "#444444"
plt.rcParams["text.color"] = "#eeeeee"
plt.rcParams["axes.labelcolor"] = "#eeeeee"
plt.rcParams["xtick.color"] = "#cccccc"
plt.rcParams["ytick.color"] = "#cccccc"

TIRE_COLORS = {
    "SOFT": "#DA291C",
    "MEDIUM": "#FFF200",
    "HARD": "#FFFFFF",
    "INTERMEDIATE": "#43B02A",
    "WET": "#0067AD",
    "UNKNOWN": "#888888",
}

st.set_page_config(page_title="F1 Pitwall Replay", layout="wide")

st.sidebar.title("🏁 Session Select")

year = st.sidebar.selectbox("Season", list(range(2024, 2017, -1)), index=0)


@st.cache_data(show_spinner="Loading event schedule...")
def get_schedule(yr):
    sched = fastf1.get_event_schedule(yr)
    sched = sched[sched["EventFormat"] != "testing"]
    # Only show races that have actually happened (avoid picking a future
    # event with no session data yet)
    sched = sched[sched["EventDate"] < pd.Timestamp.now()]
    return sched


schedule = get_schedule(year)
event_names = schedule["EventName"].tolist()
event_name = st.sidebar.selectbox("Race", event_names)

session_type = st.sidebar.selectbox(
    "Session", ["Race", "Qualifying", "Sprint", "FP1", "FP2", "FP3"], index=0
)

load_btn = st.sidebar.button("Load Session", type="primary")

st.sidebar.markdown("---")
st.sidebar.subheader("Focus Driver")
driver_placeholder = st.sidebar.empty()


@st.cache_data(show_spinner="Loading session data from FastF1 (first load can take a minute)...")
def load_session(yr, event, sess_type):
    session = fastf1.get_session(yr, event, sess_type)
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    if session.laps is None or session.laps.empty:
        raise ValueError(
            f"No lap data available for {yr} {event} - {sess_type}. "
            "This usually means that session didn't take place at this event "
            "(e.g. no Sprint that weekend), or the race hasn't happened yet / "
            "isn't in FastF1's archive yet. Try a different session type or an "
            "earlier race."
        )

    laps = session.laps.copy()
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

    # Gap to leader per lap (based on cumulative race time)
    laps["CumTimeSeconds"] = laps.groupby("Driver")["LapTimeSeconds"].cumsum()
    leader_time_per_lap = laps.groupby("LapNumber")["CumTimeSeconds"].transform("min")
    laps["GapToLeader"] = laps["CumTimeSeconds"] - leader_time_per_lap

    results = session.results[["Abbreviation", "TeamName", "Position"]].copy()
    # Track outline: X/Y from the fastest lap of the session (clean single trace)
    fastest_lap = session.laps.pick_fastest()
    track_outline = fastest_lap.get_telemetry()[["X", "Y"]].dropna()

    # Per-driver position-over-time, keyed by driver abbreviation, so we can
    # look up "where was this car at time T" for any point in the race.
    # Filter to Status == 'OnTrack' only, so pit lane / garage telemetry
    # (which sits at unrelated paddock coordinates) never gets picked as the
    # "nearest" point and doesn't create stray dots off the circuit.
    driver_positions = {}
    for drv_num in session.drivers:
        abbr = session.get_driver(drv_num)["Abbreviation"]
        if drv_num in session.pos_data:
            pos_df = session.pos_data[drv_num][["SessionTime", "X", "Y", "Status"]].dropna(
                subset=["SessionTime", "X", "Y"]
            )
            if "Status" in pos_df.columns:
                pos_df = pos_df[pos_df["Status"] == "OnTrack"]
            if not pos_df.empty:
                driver_positions[abbr] = pos_df.reset_index(drop=True)

    return session, laps, results, track_outline, driver_positions


def nearest_position(pos_df, target_time):
    """Return the (X, Y) row in pos_df whose SessionTime is closest to target_time."""
    if pos_df is None or pos_df.empty or pd.isna(target_time):
        return None
    idx = (pos_df["SessionTime"] - target_time).abs().idxmin()
    return pos_df.loc[idx]


if load_btn or "laps_df" in st.session_state:
    if load_btn:
        try:
            session, laps_df, results_df, track_outline, driver_positions = load_session(
                year, event_name, session_type
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Couldn't load this session: {e}")
            st.stop()

        st.session_state["laps_df"] = laps_df
        st.session_state["results_df"] = results_df
        st.session_state["track_outline"] = track_outline
        st.session_state["driver_positions"] = driver_positions
        st.session_state["session_meta"] = f"{year} {event_name} - {session_type}"
        st.session_state["frame"] = 0

    laps_df = st.session_state["laps_df"]
    results_df = st.session_state["results_df"]
    track_outline = st.session_state["track_outline"]
    driver_positions = st.session_state["driver_positions"]

    # Driver list ordered by finishing position (leader first)
    driver_order = (
        results_df.sort_values("Position")["Abbreviation"].tolist()
        if "Position" in results_df.columns
        else sorted(laps_df["Driver"].unique().tolist())
    )

    if "focus_driver" not in st.session_state or st.session_state["focus_driver"] not in driver_order:
        st.session_state["focus_driver"] = driver_order[0]

    focus_driver = driver_placeholder.selectbox(
        "Driver",
        driver_order,
        index=driver_order.index(st.session_state["focus_driver"]),
        key="focus_driver_select",
    )
    st.session_state["focus_driver"] = focus_driver

    total_laps = int(laps_df["LapNumber"].max())
    BIN_SIZE = 5
    total_frames = max(1, (total_laps + BIN_SIZE - 1) // BIN_SIZE)

    st.title(f"🏎️ Pitwall Replay — {st.session_state['session_meta']}")
    st.markdown(f"**Focus driver:** `{focus_driver}`")

    # ------------------------------------------------------------------
    # Frame controls
    # ------------------------------------------------------------------
    col_prev, col_slider, col_next, col_play = st.columns([1, 6, 1, 2])

    if "frame" not in st.session_state:
        st.session_state["frame"] = 0

    with col_prev:
        if st.button("◀ Prev"):
            st.session_state["frame"] = max(0, st.session_state["frame"] - 1)

    with col_next:
        if st.button("Next ▶"):
            st.session_state["frame"] = min(total_frames - 1, st.session_state["frame"] + 1)

    with col_slider:
        st.session_state["frame"] = st.slider(
            "Race progress",
            min_value=0,
            max_value=total_frames - 1,
            value=st.session_state["frame"],
            format="Block %d",
        )

    with col_play:
        autoplay = st.toggle("▶ Autoplay")

    frame = st.session_state["frame"]
    lap_cutoff = min((frame + 1) * BIN_SIZE, total_laps)
    st.caption(f"Showing laps 1–{lap_cutoff} of {total_laps}")

    # Slice the master dataframe up to current lap cutoff
    window = laps_df[laps_df["LapNumber"] <= lap_cutoff]

    # Filter everything down to just the focus driver from here on
    driver_window = window[window["Driver"] == focus_driver]

    if not driver_window.empty:
        latest = driver_window.iloc[-1]

        # Look at the previous frame's last lap for this driver, to show deltas
        prev_cutoff = max(0, lap_cutoff - BIN_SIZE)
        prev_window = laps_df[
            (laps_df["Driver"] == focus_driver) & (laps_df["LapNumber"] <= prev_cutoff)
        ]
        prev_latest = prev_window.iloc[-1] if not prev_window.empty else None

        hud1, hud2, hud3, hud4 = st.columns(4)

        with hud1:
            position = latest.get("Position")
            pos_display = f"P{int(position)}" if pd.notna(position) else "—"
            pos_delta = None
            if prev_latest is not None and pd.notna(prev_latest.get("Position")) and pd.notna(position):
                change = int(prev_latest["Position"]) - int(position)
                if change != 0:
                    pos_delta = f"{change:+d}"
            st.metric("Position", pos_display, delta=pos_delta)

        with hud2:
            compound = latest.get("Compound")
            tyre_age = latest.get("TyreLife")
            tyre_display = compound if pd.notna(compound) else "—"
            age_display = f"{int(tyre_age)} laps old" if pd.notna(tyre_age) else None
            st.metric("Current Tire", tyre_display, delta=age_display, delta_color="off")

        with hud3:
            lap_time = latest.get("LapTimeSeconds")
            lt_display = f"{lap_time:.3f}s" if pd.notna(lap_time) else "—"
            lt_delta = None
            if prev_latest is not None and pd.notna(prev_latest.get("LapTimeSeconds")) and pd.notna(lap_time):
                diff = lap_time - prev_latest["LapTimeSeconds"]
                lt_delta = f"{diff:+.3f}s"
            st.metric("Last Lap Time", lt_display, delta=lt_delta, delta_color="inverse")

        with hud4:
            st.metric("Current Lap", f"{int(latest['LapNumber'])} / {total_laps}")

        st.markdown("---")
    else:
        st.warning(
            f"{focus_driver} has no lap data yet at this point in the session "
            "(retired, DNS, or hasn't started this stint)."
        )

    st.markdown("---")

    st.subheader(f"Lap Time Evolution — {focus_driver}")
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(
        driver_window["LapNumber"],
        driver_window["LapTimeSeconds"],
        marker="o",
        markersize=4,
        color="#00D2FF",
    )
    ax1.set_xlabel("Lap")
    ax1.set_ylabel("Lap Time (s)")
    ax1.set_xlim(0, total_laps)
    st.pyplot(fig1)

    st.subheader(f"Tire Stint Timeline — {focus_driver}")
    stints = (
        driver_window.groupby(["Stint", "Compound"])
        .agg(StartLap=("LapNumber", "min"), EndLap=("LapNumber", "max"))
        .reset_index()
    )
    fig2, ax2 = plt.subplots(figsize=(12, 2.2))
    for _, row in stints.iterrows():
        ax2.barh(
            focus_driver,
            row["EndLap"] - row["StartLap"] + 1,
            left=row["StartLap"],
            color=TIRE_COLORS.get(row["Compound"], "#888888"),
            edgecolor="black",
            height=0.6,
        )
        ax2.text(
            row["StartLap"] + (row["EndLap"] - row["StartLap"]) / 2,
            focus_driver,
            row["Compound"][:1],
            ha="center",
            va="center",
            fontsize=9,
            color="black",
        )
    ax2.set_xlabel("Lap")
    ax2.set_xlim(0, total_laps)
    st.pyplot(fig2)

    st.subheader(f"Tire Degradation — {focus_driver}")
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    sns.scatterplot(
        data=driver_window,
        x="TyreLife",
        y="LapTimeSeconds",
        hue="Compound",
        palette=TIRE_COLORS,
        ax=ax3,
        s=40,
    )
    # Trendline per stint so degradation direction is easy to read
    for stint_id, stint_data in driver_window.groupby("Stint"):
        if len(stint_data) >= 2:
            ax3.plot(
                stint_data["TyreLife"],
                stint_data["LapTimeSeconds"],
                color="white",
                alpha=0.3,
                linewidth=1,
            )
    ax3.set_xlabel("Tire Age (laps)")
    ax3.set_ylabel("Lap Time (s)")
    st.pyplot(fig3)
    if autoplay:
        if st.session_state["frame"] < total_frames - 1:
            time.sleep(1.2)
            st.session_state["frame"] += 1
            st.rerun()
        else:
            st.toast("Replay finished")

else:
    st.info("Pick a season, race, and session, then click **Load Session** to begin.")
