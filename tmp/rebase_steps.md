# Rebase a Feature Branch

Use these steps to update a feature branch with the latest `main` changes.

## 1. Check the Worktree

```bash
git status --short --branch
git branch --list main random_elastic_deform
```

Do not continue if there are uncommitted changes that are not ready to
rebase. Commit them or stash them first.

## 2. Switch to the Feature Branch

```bash
git switch random_elastic_deform
```

## 3. Rebase onto Main

```bash
git rebase main
```

This replays the feature branch commits on top of the current local `main`.

## 4. Resolve Conflicts

If Git pauses:

```bash
git status
git diff --name-only --diff-filter=U
```

Resolve each conflict, then stage the resolved files:

```bash
git add <resolved-file>
git rm <file-to-keep-deleted>
git rebase --continue
```

Repeat until Git reports that the rebase is complete. Use
`git rebase --abort` only when the rebase should be cancelled.

## 5. Verify the Result

```bash
git status --short --branch
git log --oneline --decorate --graph -5
git merge-base --is-ancestor main random_elastic_deform
```

The final command should succeed, confirming that `main` is an ancestor of
the rebased feature branch. Run the relevant tests before pushing.

## 6. Update the Remote Branch

Because rebasing rewrites commit IDs, update the remote with the safer force
option:

```bash
git push --force-with-lease origin random_elastic_deform
```

`--force-with-lease` refuses to overwrite remote commits that appeared after
the last fetch, unlike an unconditional `--force` push.
