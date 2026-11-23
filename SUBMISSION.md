# HW7 submission — Blackjack

## AI-use declaration

I used ChatGPT to understand the assignment, explain reinforcement-learning methods, help write and debug Python code, organize notebook explanations, and check results. I performed the paper calculations on my tablet and ran the experiments in Google Colab. I reviewed the generated outputs.

## Setup and reproduction

- Environment: Python 3.13.18; pandas 2.2.3; Gymnasium 1.4.0, as recorded in [`outputs/dependency_versions.txt`](outputs/dependency_versions.txt).
- Blackjack environment: `Blackjack-v1`, `natural=False`, `sab=False`.
- Install dependencies: `python -m pip install -r requirements.txt`.
- Reproduce Part C from the supplied fixed histories with **one command**: `python paper_replay.py`. This writes all fixed-history results to `outputs/paper-replay/`.
- For Part D, open [`HW7_Blackjack.ipynb`](HW7_Blackjack.ipynb) in Google Colab and run cells in order, starting from the original HW7 input archive. The notebook is included in this package. The D.1 data collection, D.2–D.5 evaluation and comparisons, D.6–D.7 training, and D.8–D.9 frozen evaluation are kept in the notebook. Training-reward curves are generated after D.6–D.7 and before the D.8 evaluation. These experiments use three training seeds (11, 22, 33) and evaluation seeds (1011, 1022, 1033).

## Part A — Count states and actions

Original handwritten answers: [`paper/Part_A.pdf`](paper/Part_A.pdf). The paper includes examples and calculations for Blackjack, Durak, tic-tac-toe and Nim.

| Game | Representation | State-count formula | Type of count | Action representation | Legal actions in example | Hidden information |
| --- | --- | --- | --- | --- | --- | --- |
| Blackjack | `(player sum, dealer shown card, usable ace)` | `10 × 10 × 2 = 200` | Restricted observations | Hit or stick | 2 | Dealer's hidden card |
| Durak | Hand plus public table, trump, phase and turn | `C(36,6) = 1,947,792`; opponent candidate hands `C(29,6) = 475,020` | Hand-count components, not full states | Play a card, pick up, end attack | 6 opening; 4 in defense example | Opponent hand, stock order |
| Tic-tac-toe | 3×3 board, turn implied by legal counts | `3^9 = 19,683` | Board encodings, not exact legal positions | Put a mark in a square | 5 | None |
| Nim | `(three labeled pile sizes, player to move)` | `3 × 4 × 5 × 2 = 120` | Pile-and-turn encodings | `(pile, stones removed)` | 9 initially | None |

The number of action labels is not the same as the number of legal actions in every position. Durak especially needs a much larger state representation because of cards, hidden information and the changing table.

## Part B — Paper calculations

Original handwritten paper: [`paper/Part_B.pdf`](paper/Part_B.pdf). This includes model-based transition counts and two Bellman sweeps, every-visit Monte Carlo returns, two TD(0) passes, and two Q-learning passes. The handwritten originals are preserved unchanged here. The course requires a separate earlier Git commit for this paper work; the student must verify the commit history in their fork.

## Part C — Reproduce the calculations

- Standalone code: [`paper_replay.py`](paper_replay.py); supplied data: [`data/paper-games.csv`](data/paper-games.csv), [`data/states.csv`](data/states.csv).
- All outputs: [`outputs/paper-replay/`](outputs/paper-replay/).
- Detailed comparison: [`outputs/paper-replay/paper_vs_program.csv`](outputs/paper-replay/paper_vs_program.csv). It includes 52 comparison rows, including all intermediate TD and Q updates.
- Maximum absolute paper-versus-program difference: **2.78e-17**, which is floating-point rounding, not a material disagreement.

| Method | State / action | Sweep / pass / update | Paper result | Program result | Difference | Explanation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Model-based | A | Sweep 2 | -0.333333 | -0.333333 | 0 | Two synchronous Bellman sweeps |
| Monte Carlo | A | One pass | -0.333333 | -0.333333 | 0 | Average of three observed returns |
| TD(0) | A | End of pass 2 | -0.14003 | -0.14003 | 0 | Step-by-step state-value updates |
| Q-learning | A, hit | End of pass 2 | -0.13922 | -0.13922 | 0 | Greedy successor action value |
| Q-learning | B, stick | End of pass 2 | -0.01810 | -0.01810 | 0 | Learned action value |

The replay tests mathematical updates on prescribed examples; it does not test exploration or learned action selection.

## Part D — Simulated games

### Fixed-policy value estimates

- Shared game histories: [`outputs/fixed-policy/`](outputs/fixed-policy/), 10,000 games each for seeds 11, 22, 33.
- Model-based estimates, learned transition tables and convergence summaries: [`outputs/model-based/`](outputs/model-based/).
- Monte Carlo checkpoint estimates and visit counts: [`outputs/monte-carlo/`](outputs/monte-carlo/).
- TD(0) checkpoint estimates, visit counts and individual updates: [`outputs/td0/`](outputs/td0/).
- Values are saved at 100, 1,000 and 10,000 games per seed. All nine model-based evaluations converged within the 10,000-sweep cap at tolerance 1e-6.
- Unvisited states are marked `Unobserved`, with missing estimates represented by `NaN`, not zero. Potential missing successor states are tracked separately; all nine recorded runs have zero such successors.
- Methods compared at several states in [`outputs/comparison/method_comparison.csv`](outputs/comparison/method_comparison.csv), with a graph at [`outputs/comparison/method_comparison.png`](outputs/comparison/method_comparison.png). Model-based learns probabilities and mean rewards; Monte Carlo uses full returns; TD(0) learns by bootstrapping after every step.

Fixed-policy game summaries:

| Seed | Wins | Losses | Draws | Cut-offs | Transitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 2991 | 6415 | 594 | 0 | 17958 |
| 22 | 2940 | 6447 | 613 | 0 | 18036 |
| 33 | 2904 | 6489 | 607 | 0 | 17912 |

### Q-learning and baseline evaluation

Both Q-learning experiments used alpha = 0.1, gamma = 1.0, 50,000 training games per seed, and independent freshly initialized Q-tables. For epsilon = 0.2, exploration selected uniformly between hit and stick; for epsilon = 0, the policy used current greedy estimates with random tie-breaking. In all evaluations, the learned policies were frozen, with no Q-updates or epsilon exploration. An unexpected cut-off was recorded separately rather than treated as a completed game.

| Training seed | Policy / epsilon | Evaluation seed | Wins | Losses | Draws | Cut-offs | Mean reward |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 11 | baseline | 1011 | 2962 | 6462 | 576 | 0 | -0.3500 |
| 11 | 0.2 | 1011 | 4204 | 4894 | 902 | 0 | -0.0690 |
| 11 | 0 | 1011 | 4107 | 4905 | 988 | 0 | -0.0798 |
| 22 | baseline | 1022 | 2935 | 6481 | 584 | 0 | -0.3546 |
| 22 | 0.2 | 1022 | 4209 | 4847 | 944 | 0 | -0.0638 |
| 22 | 0 | 1022 | 4145 | 4926 | 929 | 0 | -0.0781 |
| 33 | baseline | 1033 | 2950 | 6496 | 554 | 0 | -0.3546 |
| 33 | 0.2 | 1033 | 4178 | 4941 | 881 | 0 | -0.0763 |
| 33 | 0 | 1033 | 4227 | 4839 | 934 | 0 | -0.0612 |

Average frozen-evaluation reward over the three seed pairs:

| Policy | Average reward |
| --- | ---: |
| Baseline | -0.3531 |
| Q-learning eps=0.2 | -0.0697 |
| Q-learning eps=0 | -0.0730 |

### Plots and learned policies

- Training curves for ε = 0.2: [`outputs/q-learning/epsilon_0.2/training_rewards.png`](outputs/q-learning/epsilon_0.2/training_rewards.png).
- Training curves for ε = 0: [`outputs/q-learning/epsilon_0.0/training_rewards.png`](outputs/q-learning/epsilon_0.0/training_rewards.png).
- Training comparison: [`outputs/q-learning/epsilon_comparison.png`](outputs/q-learning/epsilon_comparison.png).
- Frozen evaluation bar chart: [`outputs/q-learning/frozen-evaluation/final_policy_comparison.png`](outputs/q-learning/frozen-evaluation/final_policy_comparison.png).
- Per-seed Q-values, hit/stick counts, and policy tables separated by usable ace: [`outputs/q-learning/epsilon_0.2/`](outputs/q-learning/epsilon_0.2/) and [`outputs/q-learning/epsilon_0.0/`](outputs/q-learning/epsilon_0.0/).

### Explain the results

**Prediction versus control.** Model-based policy evaluation, Monte Carlo and TD(0) estimated values under the fixed hit-below-20 policy. Q-learning estimates values of actions and can learn a new greedy policy.

**Terminal transitions.** An `END` state has continuation value zero. Nonterminal hits normally have reward zero, while wins, losses and draws are scored when games finish.

**Usable ace.** The flag tells us whether an ace is currently worth 11 without busting. This changes the risk of hitting and must be included in the observation.

**Exploration and coverage.** Epsilon = 0.2 tries random actions some of the time. Epsilon = 0 has no explicit random exploration but can still break equal Q-values randomly. Rarely visited states and actions have uncertain estimates, and zero initialization does not prove an untried action is good.

**Variation and improvement.** Frozen evaluation shows a clear improvement over the fixed baseline on all three seed pairs. The average reward was -0.3531 for baseline, -0.0697 for Q-learning with ε = 0.2 and -0.0730 for Q-learning with ε = 0. The ε = 0 variant performed better on seed 33, so the ε comparison is not identical across seeds. These results are from finite simulation and do not prove a globally optimal policy.
