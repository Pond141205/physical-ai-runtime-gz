# Physical AI Runtime

## Mission
Build universal developer infrastructure for Physical AI.

AI may perceive, reason, and propose semantic or spatial intent.

RobotRuntime owns safety and execution authority.
MoveIt owns collision-aware motion planning.
Robot-specific adapters/controllers own embodiment-specific and low-level control.

## Work Rules
Before changing code:
1. Inspect the current repository.
2. Read this AGENTS.md.
3. Read `PHYSICAL_AI_RUNTIME_TROUBLESHOOTING.md` and check whether the symptom already has a documented cause, verification, or recovery procedure.
4. Inspect current tests and execution entrypoints.
5. Reproduce the current failure.
6. Establish a measured baseline.
7. Do not assume prior chat results are still valid.

Prefer the smallest correct patch.

After resolving a problem, append an evidence-backed troubleshooting entry with the date, symptom, affected embodiment/component, root cause, smallest corrective change, exact verification commands and measured result, and any remaining limitation. Do not record a workaround as a known-good fix without stated evidence. Read the troubleshooting file again before a new diagnosis.

## Architecture Boundary
Preferred architecture:

AI / Agent / VLA
-> Semantic Skill Contracts
-> RobotRuntime
-> Capability Negotiation
-> Frame / Workspace Translation
-> BaseRobotAdapter
-> Robot-specific Adapter
-> ROS / SDK / Simulator
-> Robot

Task logic should remain embodiment-agnostic.

MuJoCo, Panda joint names, qpos/qvel, sites, Jacobians,
and simulator-specific APIs belong in the embodiment/backend layer.

Do not patch task logic with Panda-specific coordinate offsets.

## Safety
Safety First.

- Fail closed when safety context cannot be verified.
- Never let AI/VLM output directly authorize motion.
- Never bypass PlanningScene collision validation.
- Never execute autonomous arm motion when required TF is unavailable.
- Never execute if robot state differs materially from planned start state.
- Separate PLAN-ONLY testing from EXECUTION testing.
- Never fabricate unavailable force or torque information.

## Panda Milestone Acceptance
Goal: close the Panda embodiment milestone.

Required:
- Final TCP Euclidean position error <= 0.020 m
- No timeout
- Repeated-run regression passes with zero failures
- Do not declare success after a single passing run

For every benchmark run report:
- target TCP XYZ
- final TCP XYZ
- Euclidean error
- timeout status
- convergence iterations/time
- pass/fail

Repeated-run summary:
- passed / total
- max TCP error
- mean TCP error
- timeout count

Stop only when:

max(final_tcp_error) <= 0.020 m
AND failures == 0
AND timeouts == 0

## Product KPI and Portability Evidence
The project must prove its central claim: developers can write a task once and reuse it across multiple robot models with little or no task-level change.

### North Star: Robot Portability Rate
```
Robot Portability Rate = unchanged task-level code / total task-level code * 100
```
MVP target: **>= 95% unchanged task-level code** when the same task runs on Panda and UR5e (or xArm) in simulation. Measure a named shared task program and document changed lines/files. Adapter, description, controller, and robot configuration code do not count as task-level code.

### MVP KPI
- Portability: task-level reuse >= 95%; support >= 2 robot models.
- Integration: second adapter <= 3 working days; robot-specific code <= 20% of implementation LOC.
- Reliability: pick-and-place success >= 90 / 100 task runs.
- Accuracy: final TCP Euclidean error <= 0.020 m.
- Stability: timeout rate <= 5 / 100 normal-condition task runs.
- Safety: unsafe workspace, joint-limit, and collision commands rejected 100%.
- Verification: reported task failure matches actual failure >= 95%.
- Recovery: recoverable timeout/obstruction failures recover >= 80%.
- Maintainability: duplicated adapter code <= 15%.
- Usability: a new user runs a basic example in <= 15 minutes.
- Observability: every failure has a structured code and diagnostic detail.

### Compatibility Suite
Every supported embodiment must pass the same compatibility suite: initialize, read joint/TCP state, home, valid move (<= 0.020 m), repeated move (>= 90/100), unreachable-target rejection, joint-limit rejection, workspace rejection, timeout reporting, emergency-stop behavior, pick-and-place, and state/result verification. Keep the task program shared; robot differences belong in the adapter or embodiment configuration.

### Milestone Gates
- **M1 Panda Foundation:** target `[0.45, 0.15, 0.45]`, final error <= 0.020 m, >= 95/100 repeated runs, >= 20 consecutive normal runs with no timeout, BaseRobotAdapter separated from PandaAdapter, no Panda/MuJoCo APIs in task code, and standardized `success`, `timeout`, `unreachable`, `unsafe` results.
- **M2 Cross-Robot Proof:** add UR5e or xArm; reuse >= 95% of the named task; finish the second adapter <= 3 working days; contain robot changes to adapter/config; both robots pass `home -> move -> pick -> place`, >= 90% success, and 100% unsafe-command rejection.
- **M3 Developer SDK:** first example <= 30 minutes, adapter skeleton <= 10 minutes, public API documentation complete, compatibility coverage >= 90% of core functions, third adapter <= 1 day, and adapter common-code reuse >= 80%.
- **M4 Real Robot Validation:** at least one real robot, >= 85/100 physical pick-and-place success, controller-compliant emergency stop, no command outside configured safety envelope, >= 90% simulation-to-real task reuse, and complete execution logs.

## Investigation Requirements
Explicitly determine:
1. Current execution flow
2. Panda coupling
3. MuJoCo coupling
4. Root cause of IK error
5. Root cause of timeout
6. Baseline against KPI
7. Minimal fix required

Investigate before tuning.

Check frame conventions, TCP site selection, target frame,
joint limits, Jacobian construction, damping, numerical step size,
controller convergence, simulation stepping, actuator semantics,
and stale-state issues as applicable.

## Testing
Use real repository tests and simulator execution.

Do not claim a fix from static inspection alone.

After a fix:
1. run focused regression
2. run repeated-run benchmark
3. inspect measured TCP error
4. report exact commands
5. report modified files
6. report git diff summary
