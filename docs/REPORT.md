# CartPole report

Replace each **TODO** with your own observations. Keep this report short; link code,
compact results, and images instead of pasting logs.
For each stage, explain which tool or function uses your inputs, where the
results come from, and what uses them next. Point to the relevant code or
documentation; a few sentences or an annotated arrow diagram is enough.

To link a file in Markdown, use `[link text](relative/path/to/file)`.
For example, `[Environment code](../onboarding/env.py)` becomes
[Environment code](../onboarding/env.py). Links are relative to this report's
`docs/` folder; use `../` to reach the repository root.
To display an image, add `!`: `![Model screenshot](../results/model.png)`.

## Setup

- Member: **Kevin Liu**
- OS: **Windows 10**
- Setup diagnostic, viewer, and RGB results; any fix needed: **All passed; no fix needed.**
- Fork URL and working branch: **[Fork URL](https://github.com/joejoemallianjoe/BSRA-RL)**
- Before starting, read the [toolchain overview](../resources/toolchain.md).
  What role does each of MuJoCo, Gymnasium, Stable-Baselines3, and TensorBoard
  play in this exercise? Describe how they work together in your own words: **MuJoCo handles the complex physics simulation logic and running it at a fast speed. Gymnasium handles the actual "MDP" part by standardizing the interaction between the environment and the agent (states, actions, rewards, dones, etc.). Stable-Baselines3 provides standard implementations for popular RL algorithms like PPO, DQN, etc. TensorBoard is responsible for providing an interface for statistics like loss, total episodic reward, etc.**

## Model and task

### Model (Stage 2)

- Which tool loads `scene.xml` and its included `cartpole.xml`, and what does it
  create from them? Point to the loading call in `scripts/view_model.py`. Which
  tool computes the motion you see in the viewer when a control is applied? **The "model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))" line loads the MJCF files to create an MjModel object representing like the compiled simulation model. The view_model(model, args. control, args.seconds) passes off the control to onboarding/viewer.py where mujoco.mj_step(model, data) actually steps the simulation forward with the control applied.**
- Which XML file owns the mechanism, and how does the include connect it to the
  scene? Explain the slide/hinge axes, unactuated pole, box half-extents, and
  degrees versus radians. Link a small model screenshot (`../results/model.png`): **scene.xml controls the environment like the world including the floor, the light, the camera, etc. cartpole.xml controls the cartpole. The slider joint acts on the cart, allowing movement only along the X-axis. The hinge joint acts on the pole, allowing only rotation around the Y-axis. The pole is unactuated because the CartPole task is to balance the pole. When defining the size attribute for a box geom, you need to provide half extents (e.g., length / 2, width / 2, height / 2) because MuJuCo has some weird optimizationw ith that. The actuator targets the slider instead of the pole. The <compiler angle="degree"/> specifies that the angles listed in the XML are in degrees. Note that MuJoCo ultimately converts everything to radians in the end though. [model screeenshot](https://github.com/joejoemallianjoe/BSRA-RL/blob/onboarding/cartpole/results/model.png)**

### Environment (Stage 3)

Answer each row in a few sentences, using the source linked in
[Stage 3](ONBOARDING.md#3-environment-setup-and-inspection). Name the function or
setting that supports your answer, and distinguish MuJoCo's role from Gymnasium's.

| Observations | What are the four values returned to the policy, in order and with units? Where do those values come from, and how does `_get_obs()` assemble them? | **cart position, vertical angle of pole, cart linear velocity, pole angular velocity. `_get_obs()` returns an np array of `self.data.qpos` and `self.data.qvel`. These are obtained from the MuJoCo simulation.** |
| --- | --- | --- |
| Actions | What does the action control, what is its allowed range, and how does it reach the motor? | **The action space is a 1D continuous box in the range [-3, 3]. `step(action)` passes a control value to `do_simulation` → `_step_mujoco_simulation(ctrl, n_frames)`, which updates `ctrl` and advances the actual simulation using `mujoco.mj_step(self.model, self.data)`.** |
| Physics | Which tool computes motion from your XML model, and how does Gymnasium ask it to advance? How much simulated time passes per action? | **`step()` → `do_simulation()` → `_step_mujoco_simulation()` → `mujoco.mj_step(...)` (this is where motion is computed). 0.02 seconds per physics step × a frame skip of 2 = 0.04 seconds per action.** |
| Reset | What does `reset_model()` change at the start of an episode? How are the starting values randomized, and what does using the same reset seed reproduce? | **`qpos` and `qvel` are randomized based on `self._reset_noise_scale`. Using the same reset seed reproduces the same `qpos` and `qvel` values.** |
| Reward | What is the exact reward rule, which code computes it, and does it use the state before or after the action? | **After the action is taken, the new state is observed and checked to see whether any state values are non-finite or the absolute angle between the vertical and the pole exceeds 0.2 radians. If healthy, the reward is +1; otherwise, it is 0.** |

- Environment test and random-rollout results; action-range warning:
- **test/testenv.py passed all 11 tests.**
- **Random-rollout: Mean episode length was 5.67 steps (std. dev. of 0.47) and mean return was 4.67 (std.dev. 0.47). No episodes were truncated. Video generation was successful.**
-  **Action-range warning: SB3 gave me a warning that the action space [-3, 3] was not normalized to [-1, 1] (but I kept it as [-3, 3]).**

## Training

- Run name / source commit / training seed: **first-run / 9143c17 / 0**
- Requested and actual steps / elapsed time / device: **requested: 100000 / actual: 100352 / elapsed_time: 231.42941495100968 / device: cpu**
- Configuration and versions: **[Link](https://github.com/joejoemallianjoe/BSRA-RL/blob/onboarding/cartpole/results/first-run/run.json)** link `../results/<run>/run.json`
- Learning curve: **[Learning curve](https://github.com/joejoemallianjoe/BSRA-RL/blob/onboarding/cartpole/results/first-run/learning-curve.png)** embed `../results/<run>/learning-curve.png`

- What do SB3, PyTorch, and `Monitor` each do here? How does SB3 use the environment
  and settings you pass to PPO? **SB3 provides the reinforcement learning algorithm implementaitons, which relies on PyTorch under the hood for the actual neural networks. Monitor is a wrapper for the environment that allows it to collect episode statistics like total reward and episode length. When SB3's PPO is initialized, SB3 receives the environment + hyperparameters. When model.learn() is called, SB3 automatically interacts with gym to collect observations, actions, rewards, and episode outcomes to train the policy.**
- What do the curve's axes, averaging, and smoothing mean, and what evidence of
  improvement, variability, or a plateau do you see? **The x-axis on each graph shown in the learning curve above represents environment steps, while the y axis represents mean episode length over the rollouts (top graph) and the mean total reward over the rollouts. There is a clear direct, positive increasing trend in improvement, with no evidence of a plateau yet. However, a plateau is to be expected as there will be a max amount a policy can learn once it reaches a local extrema.**

## Evaluation

Use reset seeds 10000–10019, deterministic PPO actions, and random action seeds
`reset seed + 20000`. Standard deviations describe episodes (`ddof=0`).

| Agent | Episodes | Return mean ± std | Length mean ± std | Time-limit fraction |
| --- | --- | --- | --- | --- |
| Random | 20 | 4.8 +- 2.6570660511172846 | 5.8 +- 2.6570660511172846 | 0.0 |
| PPO | 20 | 1000.0 +- 0.0 | 1000.0 +- 0.0 | 1.0 |

- Per-episode CSV and summary JSON: **[Per-episode CSV](https://github.com/joejoemallianjoe/BSRA-RL/blob/onboarding/cartpole/results/first-run/episodes.csv) [Summary JSON](https://github.com/joejoemallianjoe/BSRA-RL/blob/onboarding/cartpole/results/first-run/summary.json)** links
- Predetermined PPO rollout (seed 10000): **The policy briefly corrected the pole angle, moving to the right, and then it stood still forever, perfectly balancing the pole with 0 angular velocity...**
- Limitation or failure, and a specific diagnosis if learning was weak: **Learning was not weak; PPO policy showed strong results, with 100% of the seeds ending by truncation (cuz it's just that cracked)**

- Why does this evaluation differ from a training curve, and what can one
  trained seed not establish? What carries over to humanoid soccer, and what is
  missing (for example, contacts, partial observations, or sim-to-real transfer)? **Validation is crucial because training curves can be deceiving. An agent can overfit onto specific seeds or environment setups and can happen to look like it is performing well. It can also get trapped in a suboptimal policy. Validating it on new seeds ensures that it is robust against different initial conditions. One training seed only establishes how well your algorithm performs on a specific seed. The resulting PPO policy can be much different on another training seed because it affects the explored states, actions taken, etc. What carries over to humanoid soccer is the core RL loop: Observe state --> take action --> receive reward and new state with policy pi. The same idea for validation applies too: test the agent's policy on new seeds and compare it to a random baseline OR a human baseline to see if it has measurable improvement, rather than relying on training curves alone. What's missing is definitely the sim-to-real transfer. The simulation doesn't perfectly model the real world, which can cause issues where it works well in simulation but it doesn't work well in the real world. Additionally, in humanoid soccer, there are multiple agents working together, and there are much more complex joints and geoms.**

## Review

**To reproduce setup:**
```bash
# Setup command
# Install git, uv, fork and clone, the whole shebang then run uv sync
uv sync

# Test that XML files are correct
uv run python scripts/view_model.py
uv run python scripts/view_model.py --control 0.1 --seconds 1
uv run python scripts/view_model.py --control -0.1 --seconds 1

# To check the MuJoCo gym env
uv run python -m pytest test/test_env.py
uv run python scripts/evaluate.py --random-only --run-name random-debug --episodes 3 --trace
uv run python scripts/evaluate.py --random-only --run-name random-debug --episodes 3 --video

# Run final tests
uv run python -m pytest test/

# Train PPO
uv run python scripts/train.py --run-name first-run --seed 0 --steps 100000

# Evaluating trained policy and recording a video
uv run python scripts/evaluate.py --policy models/first-run/policy.zip --run-name first-run --video
```
Keep the saved policy, TensorBoard logs, and rollout video locally and share them
directly with the RL lead if requested. No file uploads, download links, or GitHub
release are required for these files.

- Automated test results: **17 passed (1 warning about the control range being [-3.0 to 3.0])**
- Manual model/render/video checks: **All passed:
- 
- MuJoco model: uv run python -m pytest test/test_model.py (2/2 tests passed)
- Gym environment: uv run python -m pytest test/test_env.py (11/11 passed; 1 warning about control range)
- Render: uv run python scripts/view_model.py -- cartpole appears with rail, floor, lighting, and camera
- uv run python scripts/evaluate.py --random-only --run-name random-debug --episodes 3 --trace: trace prints results for 3 random episodes
- uv run python scripts/evaluate.py --random-only --run-name random-debug --episodes 3 --video: video was generated**
- Fork PR link and review notes (record merge in the PR): **TODO**

## Feedback

- Did you learn anything from this onboarding? What was new, or what became
  clearer? If little was new, say so: **A LOT was new. I didn't know how MJCF or MuJoCo really worked or how git worked in general. I spent the most time on parts 1 and 2. The format of the instructions was a little confusing, since it was like a lot of shorthand and some parts were kinda out of order, but I did eventually get the flow of it. On the initial git setup part, I was definitely confused for a while since I didn't know what the difference was between a fork and branch was, or what a repository was and why we needed a remote and a local one and what the workflow was. I also got confused on the MJCF part since I wasn't really familiar with the idea of using a file to represent robots, but it definitely makes sense.**
- What improvements would make the onboarding easier to follow or more useful? **I would say that in some sections it's a little bit confusing where the "start" point is since it seems out of order. I definitely felt this way when I saw the first section, since it's a lot of information and it would help to keep the part about explaining the purpose of the project and the overview and then jumping straight to like "first do this" and then gradually introduce stuff like "look for the TO-DOs inside the files and edit them," but idk maybe that's just me.**
  Mention any confusing instructions, missing background, or unnecessary work: **I definitely felt a bit confused about what the uv stuff was for. I know it probably wasn't necessary for the project, but I sort of felt like I was executing commands without really understanding what was happening at the start. I did understand the later stuff though with MJCF, Tensorboard, SB3, etc.**
