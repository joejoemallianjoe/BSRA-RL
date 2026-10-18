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
- Fork URL and working branch: **[TODO](https://github.com/joejoemallianjoe/BSRA-RL)**
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
  degrees versus radians. Link a small model screenshot (`../results/model.png`): **scene.xml controls the environment like the world including the floor, the light, the camera, etc. cartpole.xml controls the cartpole. The slider joint acts on the cart, allowing movement only along the X-axis. The hinge joint acts on the pole, allowing only rotation around the Y-axis. The pole is unactuated because the CartPole task is to balance the pole. When defining the size attribute for a box geom, you need to provide half extents (e.g., length / 2, width / 2, height / 2) because MuJuCo has some weird optimizationw ith that. The actuator targets the slider instead of the pole. The <compiler angle="degree"/> specifies that the angles listed in the XML are in degrees. Note that MuJoCo ultimately converts everything to radians in the end though. Link: https://github.com/joejoemallianjoe/BSRA-RL/blob/onboarding/cartpole/results/model.png **

### Environment (Stage 3)

Answer each row in a few sentences, using the source linked in
[Stage 3](ONBOARDING.md#3-environment-setup-and-inspection). Name the function or
setting that supports your answer, and distinguish MuJoCo's role from Gymnasium's.

| Observations | What are the four values returned to the policy, in order and with units? Where do those values come from, and how does `_get_obs()` assemble them?     | **cart position, vertical angle of pole, cart linear velocity, pole angular velocity. _get_obs() returns an np array of self.data.qpos and self.data.qvel. These are obtained from the MuJoCo simulation.**    |
| Actions      | What does the action control, what is its allowed range, and how does it reach the motor?                                                               | **The action space is a 1D continuous box in the range [-3, 3]. step(action) passes a control value to do_simulation --> _step_mujoco_simulation(ctrl, n_frames) which updates ctrl and advances the actual simulation mujoco.mj_step(self.model, self.data)**    |
| Physics      | Which tool computes motion from your XML model, and how does Gymnasium ask it to advance? How much simulated time passes per action?                    | **gym step() --> do_simulation() --> _step_mujoco_simulation() --> mujoco.mj_step(...) (this is where motion is computed). 0.02 seconds / physics step * frame skip of 2 = 0.04 seconds / action.**    |
| Reset        | What does `reset_model()` change at the start of an episode? How are the starting values randomized, and what does using the same reset seed reproduce? | **q_pos and q_vel are randomized based on self._reset_noise_scale. Using the same reset seed produces the same q_pos and q_vel numbers.**    |
| Reward       | What is the exact reward rule, which code computes it, and does it use the state before or after the action?                                            | **After the action is taken, the new state is observed and checked to see if any state space values no longer finite or the |angle between the vertical and the pole| > 0.2. If healthy then +1 reward, otherwise 0.**    |

- Environment test and random-rollout results; action-range warning:
- **test/testenv.py passed all 11 tests.**
- **Random-rollout: Mean episode length was 5.67 steps (std. dev. of 0.47) and mean return was 4.67 (std.dev. 0.47). No episodes were truncated. Video generation was successful.**
-  **Action-range warning: SB3 gave me a warning that the action space [-3, 3] was not normalized to [-1, 1] (but I kept it as [-3, 3]).**

## Training

- Run name / source commit / training seed: **first-run / 9143c17 / 0**
- Requested and actual steps / elapsed time / device: **requested: 100000 / actual: 100352 / elapsed_time: 231.42941495100968 / device: cpu**
- Configuration and versions: **[Link here](https://github.com/joejoemallianjoe/BSRA-RL/blob/onboarding/cartpole/results/first-run/run.json)** link `../results/<run>/run.json`
- Learning curve: **[TODO](https://github.com/joejoemallianjoe/BSRA-RL/blob/onboarding/cartpole/results/first-run/learning-curve.png)** embed `../results/<run>/learning-curve.png`

- What do SB3, PyTorch, and `Monitor` each do here? How does SB3 use the environment
  and settings you pass to PPO? **SB3 provides the reinforcement learning algorithm implementaitons, which relies on PyTorch under the hood for the actual neural networks. Monitor is a wrapper for the environment that allows it to collect episode statistics like total reward and episode length. When SB3's PPO is initialized, SB3 receives the environment + hyperparameters. When model.learn() is called, SB3 automatically interacts with gym to collect observations, actions, rewards, and episode outcomes to train the policy.**
- What do the curve's axes, averaging, and smoothing mean, and what evidence of
  improvement, variability, or a plateau do you see? **The x-axis on each graph shown in the learning curve above represents environment steps, while the y axis represents mean episode length over the rollouts (top graph) and the mean total reward over the rollouts. There is a clear direct, positive increasing trend in improvement, with no evidence of a plateau yet. However, a plateau is to be expected as there will be a max amount a policy can learn once it reaches a local extrema.**

## Evaluation

Use reset seeds 10000–10019, deterministic PPO actions, and random action seeds
`reset seed + 20000`. Standard deviations describe episodes (`ddof=0`).

| Agent | Episodes | Return mean ± std | Length mean ± std | Time-limit fraction |
| --- | --- | --- | --- | --- |
| Random | TODO | TODO | TODO | TODO |
| PPO | TODO | TODO | TODO | TODO |

- Per-episode CSV and summary JSON: **TODO** links
- Predetermined PPO rollout (seed 10000): **TODO** briefly describe the behavior you observed
- Limitation or failure, and a specific diagnosis if learning was weak: **TODO**

- Why does this evaluation differ from a training curve, and what can one
  trained seed not establish? What carries over to humanoid soccer, and what is
  missing (for example, contacts, partial observations, or sim-to-real transfer)? **TODO**

## Review

**TODO:** Record exact setup, test, train, and evaluate commands, plus any rendering
environment variable.

Keep the saved policy, TensorBoard logs, and rollout video locally and share them
directly with the RL lead if requested. No file uploads, download links, or GitHub
release are required for these files.

- Automated test results: **TODO**
- Manual model/render/video checks: **TODO**
- Fork PR link and review notes (record merge in the PR): **TODO**

## Feedback

- Did you learn anything from this onboarding? What was new, or what became
  clearer? If little was new, say so: **TODO**
- What improvements would make the onboarding easier to follow or more useful?
  Mention any confusing instructions, missing background, or unnecessary work: **TODO**
