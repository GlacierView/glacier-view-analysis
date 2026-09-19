# General rules

- When sources disagree, precedence is: the checkpoints and the data on disk, then the paper
  (`.agents/references/paper.md`), then these `.agents/` files, then the code, then
  `docs/low_level_design.md`, then `README.md`. The LLD is authoritative on *rationale* and stale on
  specifics; `README.md` is the oldest and partly predates the current tree.
- Verify before recommending. Several documented facts here were wrong until checked against the
  bytes — the model's channel count among them.
- Do not add instructions to agent-specific files. `.agents/` is the source of truth and
  `AGENTS.md` / `CLAUDE.md` are thin entrypoints onto it.
