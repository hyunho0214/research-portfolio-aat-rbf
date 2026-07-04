# Codex-Assisted, Human-in-the-Loop Workflow

This project is framed as an AI-assisted semiconductor yield-defect detection
program. The intent is not to pretend that every line was manually typed from
scratch. Instead, the repository documents how the project owner used Codex
actively: defining the semiconductor problem, setting validation criteria,
asking for implementation options, approving the direction, and checking the
results.

## Portfolio Message

> I used Codex as an engineering co-pilot for a semiconductor ML project. I did
> not passively accept generated code. I supplied the process-domain problem,
> required leakage-safe preprocessing, asked Codex to expose validation commands,
> reviewed results after each experiment, and redirected the project toward a
> more useful yield-defect detection workflow.

## Human Responsibilities

- Define the semiconductor manufacturing context.
- Identify the statistical traps: class imbalance, missing sensor values,
  high-dimensional noise, and misleading accuracy.
- Decide which metrics matter for production: fail recall, fail F1, PR-AUC, and
  missed-fail counts.
- Approve each major project direction before Codex expands the implementation.
- Interpret whether the output makes sense for process engineering.

## Codex Responsibilities

- Scaffold reusable Python code.
- Convert requirements into scikit-learn / imbalanced-learn pipelines.
- Add model comparison, report generation, and hyperparameter tuning.
- Catch implementation risks such as SMOTE leakage and XGBoost label handling.
- Produce validation commands and summarize evidence after each run.

## Direction-Approval Rule

This project intentionally uses a "direction approval" pattern:

1. Codex proposes the next technical direction.
2. The user confirms, corrects, or narrows that direction.
3. Codex implements the approved step.
4. Codex runs the validation harness.
5. The user reviews the result and decides the next direction.

This is important because AI-generated code can look plausible while still
missing the actual engineering objective. For this project, the objective is not
only to get a higher score; it is to build an auditable semiconductor
fail-detection workflow.

Codex is therefore treated as an implementation and validation partner, not as
an autonomous author. Major steps should pause at an approval gate: problem
definition, preprocessing policy, model family, evaluation metric, threshold
policy, and final GitHub artifact selection.

## Key User Corrections

- The project should be presented as AI-assisted development, not as pure
  hand-written code.
- The GitHub repository should include the prompts, commands, and validation
  harness used with Codex.
- Codex should not run far ahead without exposing the direction and letting the
  user steer the project.
- The repository should make the user's technical judgment visible: what was
  requested, what was approved, what was corrected, and what evidence was used.

## Guardrails Used

- Treat fail (`1`) as the positive class.
- Do not use accuracy as the primary score.
- Apply SMOTE only inside the training fold.
- Keep feature selection inside the cross-validation pipeline.
- Preserve confusion matrices so missed-fail counts remain visible.
- Keep raw public data and generated intermediate reports out of Git.
- Save prompts and validation commands as portfolio evidence.
