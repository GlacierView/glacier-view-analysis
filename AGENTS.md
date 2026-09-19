# Agent instructions

GlacierView measures mountain glacier surface-area change over time from Landsat imagery, using a
ResNet-50-encoder U-Net. Agent guidance lives under `.agents/`.

## Always follow

- `.agents/rules/general.md`
- `.agents/rules/code.md`
- `.agents/rules/data.md`

## Consult when relevant

| Need | Read |
|---|---|
| What the project is and what it found | `.agents/context/overview.md` |
| How data flows, stage by stage | `.agents/context/pipeline.md` |
| Landing zone, `data_label`, Athena tables | `.agents/context/data-model.md` |
| The model's input contract and checkpoints | `.agents/context/model.md` |
| What is present on disk vs missing | `.agents/context/state.md` |

## Skills

Load one only when doing that task:

- `.agents/skills/setup-environment/SKILL.md`
- `.agents/skills/run-inference/SKILL.md`
- `.agents/skills/train-model/SKILL.md`

## Reference material

`.agents/references/` — the paper's numbers, the AWS inventory, and pointers to the long-form docs.
See `.agents/references/README.md`.

## Important

- This is research code: no tests, no CI. When something fails, assume the code has drifted from the
  data before assuming you are using it wrong.
- Prefer existing implementations over introducing new patterns.
- Do not read every file under `.agents/`. Retrieve what the task needs.
- When instructions conflict, the more specific one wins.
