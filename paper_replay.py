"""HW7 Part C: reproduce the supplied four Blackjack game histories.

Run from the repository root:
    python paper_replay.py

This uses only the Python standard library and the original CSVs.
The fixed-policy replay never samples replacement histories.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data"
OUTPUT = ROOT / "outputs" / "paper-replay"
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save(name, rows, columns):
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


states_data = read_csv(INPUT / "states.csv")
labels = [row["label"] for row in states_data]
player_sums = {row["label"]: int(row["player_sum"]) for row in states_data}
policy = {s: 1 if player_sums[s] < 20 else 0 for s in labels}

history = read_csv(INPUT / "paper-games.csv")
history.sort(key=lambda row: (int(row["game"]), int(row["step"])))
for row in history:
    row["game"] = int(row["game"])
    row["step"] = int(row["step"])
    row["action"] = int(row["action"])
    row["reward"] = float(row["reward"])
    assert row["action"] == policy[row["state"]]

paper_td = (
    [0, .1, -.1, 0, 0, 0, -.08, -.01],
    [-.073, .091, -.1657, 0, 0, 0, -.14003, -.0181]
)
paper_q = (
    [0, .1, -.1, 0, 0, 0, -.08, -.01],
    [-.072, .091, -.1648, 0, 0, 0, -.13922, -.0181]
)
comparison = []


def add_comparison(method, state, place, on_paper, computed):
    comparison.append({
        "Method": method, "State / Action": state,
        "Sweep / Pass / Update": place,
        "Paper result": on_paper, "Program result": computed,
        "Difference": computed - on_paper
    })

# 1. Model-based: estimate probabilities and rewards from the supplied data.
counts = Counter((r["state"], r["action"], r["next_state"]) for r in history)
totals = Counter((r["state"], r["action"]) for r in history)
reward_sums = defaultdict(float)
for r in history:
    reward_sums[(r["state"], r["action"], r["next_state"])] += r["reward"]

model_rows = []
for (s, a, ns), count in sorted(counts.items()):
    model_rows.append({
        "State": s, "Action": "Hit" if a == 1 else "Stick",
        "Next": ns, "Count": count,
        "Probability": count / totals[(s, a)],
        "Mean reward": reward_sums[(s, a, ns)] / count
    })
save("transition_model.csv", model_rows, list(model_rows[0]))

coverage = [{"State": s, "Action": "Hit" if a else "Stick",
             "Visits": totals[(s, a)],
             "Status": "Observed" if totals[(s, a)] else "Unobserved"}
            for s in labels for a in (0, 1)]
save("action_coverage.csv", coverage, list(coverage[0]))

V = {s: 0.0 for s in labels}
model_sweeps = []
for sweep in range(3):
    model_sweeps.append({"Sweep": sweep, **V})
    for s in labels:
        target_paper = (-1 / 3 if s == "A" else 0.0) if sweep else 0.0
        add_comparison("Model-Based", s, f"Sweep {sweep}", target_paper, V[s])
    if sweep == 2:
        break
    old = V.copy()
    for s in labels:
        a = policy[s]
        assert totals[(s, a)] > 0  # All fixed-policy state-action pairs occur here.
        value = 0.0
        for (s1, a1, ns), count in counts.items():
            if (s1, a1) != (s, a):
                continue
            probability = count / totals[(s, a)]
            average_reward = reward_sums[(s, a, ns)] / count
            continuation = 0.0 if ns == "END" else old[ns]
            value += probability * (average_reward + continuation)
        V[s] = value
save("model_based_sweeps.csv", model_sweeps, ["Sweep"] + labels)

# 2. Every-visit Monte Carlo: work backward inside each finished episode.
returns = defaultdict(list)
by_game = defaultdict(list)
for row in history:
    by_game[row["game"]].append(row)
for game in sorted(by_game):
    G = 0.0
    for row in reversed(by_game[game]):
        G = row["reward"] + G  # gamma = 1
        returns[row["state"]].append(G)
mc_rows = []
for s in labels:
    observed = returns[s]
    mean = sum(observed) / len(observed)
    mc_rows.append({"State": s, "Returns": str(observed),
                    "Visits": len(observed), "MC Value": mean})
    add_comparison("Monte Carlo", s, "Pass 1",
                   -1 / 3 if s == "A" else 0.0, mean)
save("monte_carlo_returns.csv", mc_rows, list(mc_rows[0]))

# 3. TD(0): update after each transition, replay twice without resetting.
alpha, gamma = 0.1, 1.0
V = {s: 0.0 for s in labels}
td_rows = []
for pass_no in (1, 2):
    for i, row in enumerate(history):
        s, ns, r = row["state"], row["next_state"], row["reward"]
        old = V[s]
        next_value = 0.0 if ns == "END" else V[ns]
        target = r + gamma * next_value
        new = old + alpha * (target - old)
        V[s] = new
        on_paper = paper_td[pass_no - 1][i]
        td_rows.append({"Method": "TD(0)", "Pass": pass_no,
                        "Game": row["game"], "Step": row["step"],
                        "State": s, "Old": old, "Reward": r,
                        "Next V": next_value, "Target": target,
                        "New V": new, "Paper": on_paper,
                        "Difference": new - on_paper})
        add_comparison("TD(0)", s,
                       f"Pass {pass_no}, Game {row['game']}, Step {row['step']}",
                       on_paper, new)
save("td0_updates.csv", td_rows, list(td_rows[0]))

# 4. Q-learning: compare both actions for the next state; new Q-table.
Q = {(s, a): 0.0 for s in labels for a in (0, 1)}
q_rows = []
for pass_no in (1, 2):
    for i, row in enumerate(history):
        s, a, ns, r = row["state"], row["action"], row["next_state"], row["reward"]
        old = Q[(s, a)]
        next_max = 0.0 if ns == "END" else max(Q[(ns, 0)], Q[(ns, 1)])
        target = r + gamma * next_max
        new = old + alpha * (target - old)
        Q[(s, a)] = new
        on_paper = paper_q[pass_no - 1][i]
        q_rows.append({"Pass": pass_no, "Game": row["game"],
                       "Step": row["step"], "State": s,
                       "Action": "Hit" if a else "Stick", "Old Q": old,
                       "Reward": r, "Max next Q": next_max,
                       "Target": target, "New Q": new, "Paper": on_paper,
                       "Difference": new - on_paper})
        add_comparison("Q-learning", f"{s} / {'Hit' if a else 'Stick'}",
                       f"Pass {pass_no}, Game {row['game']}, Step {row['step']}",
                       on_paper, new)
save("q_learning_updates.csv", q_rows, list(q_rows[0]))
final_q = []
for s in labels:
    hit, stick = Q[(s, 1)], Q[(s, 0)]
    best = "Tie" if hit == stick else "Hit" if hit > stick else "Stick"
    final_q.append({"State": s, "Hit": hit, "Stick": stick, "Greedy action": best})
save("final_q_table.csv", final_q, list(final_q[0]))

save("paper_vs_program.csv", comparison, list(comparison[0]))
max_diff = max(abs(row["Difference"]) for row in comparison)
print("Paper replay completed using the supplied 4 fixed game histories.")
print(f"Compared {len(comparison)} results; max absolute difference: {max_diff:.12g}")
print("Saved tables and updates in outputs/paper-replay/")
if max_diff > 1e-10:
    raise RuntimeError("A paper-versus-program discrepancy needs investigation")
