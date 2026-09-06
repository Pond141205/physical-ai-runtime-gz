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
3. Inspect current tests and execution entrypoints.
4. Reproduce the current failure.
5. Establish a measured baseline.
6. Do not assume prior chat results are still valid.

Prefer the smallest correct patch.

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
