# Version History Layout

Every change agentopt applies to the user's files creates one version under
`.agentopt/history/`, snapshotted **before** the write — there is never a
moment where an applied change exists without its undo.

```
.agentopt/history/
  v1/
    meta.json          # what changed and the evidence
    files/<relpath>    # each touched file's content BEFORE this change
  v2/
    ...
```

## meta.json

| Field | Meaning |
|---|---|
| `version` | `v<n>`, monotonically increasing |
| `ts` | ISO timestamp of the apply |
| `description` | one-line change description |
| `entry_id` | manifest entry the change targeted |
| `files` | touched files, relative paths |
| `score_before` / `score_after` | holdout/suite scores in percent |
| `post_hashes` | sha256 of each touched file AFTER the change |

## Rollback semantics

`rollback vN` undoes versions latest → N, restoring each version's pre-images
in reverse order. Before restoring a file, its current sha256 must equal the
`post_hashes` entry of the version being undone (or of the latest version that
touched it): a mismatch means the user hand-edited the file since — the
rollback stops and reports instead of destroying their work (`--force`
overrides).
