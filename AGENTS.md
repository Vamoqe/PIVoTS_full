# Research workflow

The user requires version control at each completed research stage.

- Record each completed stage in README.md's version history: concrete changes, validation results, remaining assumptions and limitations.
- Run the relevant tests and scientific checks before committing the stage. Keep experiment protocols, source hashes and compact summaries reproducible; do not add large raw simulation outputs to Git.
- Create a commit and a unique version tag for each completed stage, then push the research branch and that tag to origin. The current research branch is codex/research-stages. Use ordinary fast-forward pushes; do not force-push or move existing tags.
- Preserve and identify unfinished research. Monte Carlo checks do not establish a general theorem, and sample rank does not establish population identification.
- In pilot comparisons, include the cost of pilot data, respect the common calendar horizon, and report identification failures alongside conditional error summaries. Keep DGP-level summaries separate.
- The repository has a flat layout. Run Python entry points and unittest discovery from the repository root.
