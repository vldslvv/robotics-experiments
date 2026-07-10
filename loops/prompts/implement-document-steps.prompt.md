You are performing one isolated Codex run over an implementation document.

Implementation document:

`{{ITEM_PATH}}`

The path is relative to the project root.

## Goal

Work through every unimplemented step in `{{ITEM_PATH}}`. For each step, plan
and implement it, run the complete verification routine, create one commit, and
mark the step completed. Repeat until the document contains no unimplemented
steps.

Do not stop after producing a plan or after completing only one step. The shell
runner invokes this prompt once for the document, so iteration and completion
decisions belong to this Codex run.

## Step selection

Read `AGENTS.md`, the complete implementation document, and the current
repository before selecting work.

Choose the next genuinely unimplemented step using the document's priorities,
suggested sequence, dependencies, and the repository's actual state. Do not
assume that the first unchecked line is necessarily the correct next action. A
checkbox or status label is evidence of intent, not a substitute for inspecting
the implementation.

Prefer the documented sequence when multiple steps are ready. If the repository
already satisfies a nominally incomplete step, verify that conclusion, update
the document to describe the current behavior, and complete the same verification
and commit routine before marking it done.

## Routine for each step

1. Identify one next unimplemented step and read its complete description.
2. Read only the area-specific documentation required by that step.
3. Inspect the relevant implementation and make a concrete plan for completing
   the step. Continue directly into implementation; do not pause for approval.
4. Implement the complete step, including production code, tests, configuration,
   and documentation required by the repository instructions.
5. Update the step's detailed description so it states the resulting current
   behavior rather than leaving stale recommendations or observations. Keep its
   completion marker unchecked until verification succeeds.
6. Run all of the following commands:
   - `uv run pytest`
   - `uv run pre-commit run --all-files`
   - `make test-integration-all`
7. Fix every failure and rerun the affected commands. All three commands must
   pass for the final implementation of this step. If pre-commit changes files,
   review the changes and rerun affected verification.
8. If a required command cannot pass, stop the entire workflow. Do not mark the
   step completed and do not commit the incomplete step. Report the blocker and
   current working-tree state.
9. After verification passes, mark only this step completed using the document's
   existing completion convention. Use `[x]` for Markdown task-list entries.
10. Run `uv run pre-commit run --all-files` again to validate the completion and
    documentation changes. Fix failures or restore the step to incomplete before
    stopping.
11. Create exactly one Git commit for this step. Include its implementation,
    tests, configuration, documentation, and completion marker. Use a concise
    imperative commit subject derived from the step.
12. Confirm that the working tree is clean, reread `{{ITEM_PATH}}`, and begin the
    routine again with the next unimplemented step.

Do not create intermediate commits within a step. Do not combine multiple steps
in one commit. Do not amend, squash, or modify commits that existed before this
run. Do not skip, weaken, or substitute the required verification commands.

When no unimplemented steps remain, make no additional commit. Confirm that the
working tree is clean and finish.

## Final response

Summarize each completed step, its verification results, and its commit hash. If
the workflow stopped early, identify the first remaining step and the blocker.
