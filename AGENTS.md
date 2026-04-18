# AGENTS.md

This folder contains Python simulations for network-link throughput calculations and related upper-bound experiments.

## Default behavior

- Read code first, then implement.
- Prefer editing existing scripts over adding new abstractions.
- Add new functions or runnable scripts when needed.
- Assume the user will run expensive simulations on a server.
- Do not ask questions that can be answered from local code context.

## Coding rules

- Keep solutions simple and script-friendly.
- Use `if __name__ == "__main__":` for runnable entry points.
- Reuse local utilities and existing data conventions.
- Keep simulation parameters explicit and easy to edit.
- Avoid hardcoded machine-specific paths unless already required by the code.

## Do not

- run long or expensive jobs unless explicitly asked
- rename core files or rewrite the folder structure without need
- delete notebooks, data, or cached outputs unless asked
- introduce heavy frameworks or unnecessary abstractions

## Ask the user only if

- the math or experiment design is genuinely ambiguous
- the change would overwrite important results
- credentials, remote access, or missing infrastructure details are required

Otherwise, make a reasonable assumption, implement the change, and state that assumption briefly.
