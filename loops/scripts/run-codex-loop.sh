#!/usr/bin/env bash
set -Eeuo pipefail

# Generic Codex loop runner.
#
# Usage:
#   ./loops/scripts/run-codex-loop.sh [options] [items-file] [prompt-template]
#
# Defaults:
#   items-file:      loops/inputs/modules.txt
#   prompt-template: loops/prompts/refactor-public-api.prompt.md
#
# Examples:
#   ./loops/scripts/run-codex-loop.sh
#   ./loops/scripts/run-codex-loop.sh --resume
#   ./loops/scripts/run-codex-loop.sh --resume --run-id 20260706-143000
#   ./loops/scripts/run-codex-loop.sh --start-after src/foo/bar
#   ./loops/scripts/run-codex-loop.sh --only src/foo/bar
#   ./loops/scripts/run-codex-loop.sh --max-items 3
#
# Directory layout:
#   loops/
#     scripts/
#       run-codex-loop.sh
#     prompts/
#       refactor-public-api.prompt.md
#     inputs/
#       modules.txt
#     runs/
#       <run-id>/
#         run.log
#         results.jsonl
#         prompts/
#         logs/
#       latest -> <run-id>
#
# Supported prompt placeholders:
#   {{ITEM}}
#   {{ITEM_PATH}}
#   {{MODULE_PATH}}
#
# Environment overrides:
#   PROJECT_ROOT_NAME=robotics-experiments
#   CODEX_SANDBOX=workspace-write
#   CODEX_APPROVAL=never
#   REQUIRE_CLEAN_START=1
#   REQUIRE_CLEAN_BETWEEN_RUNS=1
#   VALIDATE_ITEM_PATHS=1

ITEMS_FILE=""
PROMPT_TEMPLATE=""

PROJECT_ROOT_NAME="${PROJECT_ROOT_NAME:-robotics-experiments}"
CODEX_SANDBOX="${CODEX_SANDBOX:-workspace-write}"
CODEX_APPROVAL="${CODEX_APPROVAL:-never}"
REQUIRE_CLEAN_START="${REQUIRE_CLEAN_START:-1}"
REQUIRE_CLEAN_BETWEEN_RUNS="${REQUIRE_CLEAN_BETWEEN_RUNS:-1}"
VALIDATE_ITEM_PATHS="${VALIDATE_ITEM_PATHS:-1}"

RUNS_ROOT="loops/runs"
RUN_ID=""
RUN_DIR=""
RUN_LOG=""
RESULTS_FILE=""
PROMPTS_DIR=""
ITEM_LOGS_DIR=""

RESUME=0
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
SKIP_COMPLETED_EXPLICIT=0
START_AFTER=""
ONLY_ITEM=""
MAX_ITEMS=""

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat >&2 <<EOF
Usage:
  $0 [options] [items-file] [prompt-template]

Options:
  --resume
      Resume an existing run. If --run-id is omitted, resumes loops/runs/latest.
      By default, this also skips items that already have status=success.

  --run-id <id>
      Use a specific run id under loops/runs/<id>.
      For a new run, the id is created.
      With --resume, the id must already exist.

  --skip-completed
      Skip items that already have a success record in results.jsonl for this run.

  --no-skip-completed
      Do not skip completed items, even when using --resume.

  --start-after <item>
      Skip items until this exact item is seen, then start with the following item.

  --only <item>
      Run only this exact item.

  --max-items <n>
      Process at most n runnable items.

  -h, --help
      Show this help.

Defaults:
  items-file:      loops/inputs/modules.txt
  prompt-template: loops/prompts/refactor-public-api.prompt.md
EOF
  exit 2
}

now_iso() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  local line
  line="[$(now_iso)] $*"

  if [[ -n "${RUN_LOG:-}" ]]; then
    printf '%s\n' "$line" | tee -a "$RUN_LOG"
  else
    printf '%s\n' "$line"
  fi
}

trim_and_strip_comment() {
  sed \
    -e 's/#.*$//' \
    -e 's/^[[:space:]]*//' \
    -e 's/[[:space:]]*$//'
}

slugify() {
  local value="$1"

  printf '%s' "$value" \
    | sed \
        -e 's#[^A-Za-z0-9._-]#-#g' \
        -e 's#--*#-#g' \
        -e 's#^-##' \
        -e 's#-$##' \
    | cut -c 1-120
}

assert_project_root() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "current directory is not inside a Git repository"

  local git_root
  git_root="$(git rev-parse --show-toplevel)"

  if [[ "$PWD" != "$git_root" ]]; then
    die "script must be run from the project root. Current pwd: $PWD; Git root: $git_root"
  fi

  if [[ "$(basename "$git_root")" != "$PROJECT_ROOT_NAME" ]]; then
    die "invalid project root. Expected directory named '$PROJECT_ROOT_NAME', got: $(basename "$git_root")"
  fi
}

git_tree_dirty() {
  [[ -n "$(git status --porcelain)" ]]
}

assert_clean_git_tree() {
  local reason="$1"

  if git_tree_dirty; then
    echo "ERROR: working tree is dirty: $reason" >&2
    git status --short >&2
    exit 1
  fi
}

assert_item_path_valid() {
  local item="$1"

  if [[ "$item" = /* ]]; then
    die "item must be a relative path, got: $item"
  fi

  if [[ "$item" =~ (^|/)\.\.(/|$) ]]; then
    die "item must not contain '..', got: $item"
  fi

  if [[ ! -e "$item" ]]; then
    die "item path does not exist: $item"
  fi
}

render_prompt() {
  local item="$1"
  local template
  template="$(< "$PROMPT_TEMPLATE")"

  template="${template//\{\{ITEM\}\}/$item}"
  template="${template//\{\{ITEM_PATH\}\}/$item}"
  template="${template//\{\{MODULE_PATH\}\}/$item}"

  printf '%s\n' "$template"
}

append_result() {
  local item="$1"
  local status="$2"
  local commit="$3"
  local prompt_path="$4"
  local item_log="$5"
  local message="$6"
  local started_at="$7"
  local finished_at="$8"
  local duration_seconds="$9"

  RESULT_TIMESTAMP="$(now_iso)" \
  RESULT_RUN_ID="$RUN_ID" \
  RESULT_ITEM="$item" \
  RESULT_STATUS="$status" \
  RESULT_COMMIT="$commit" \
  RESULT_PROMPT="$prompt_path" \
  RESULT_LOG="$item_log" \
  RESULT_MESSAGE="$message" \
  RESULT_STARTED_AT="$started_at" \
  RESULT_FINISHED_AT="$finished_at" \
  RESULT_DURATION_SECONDS="$duration_seconds" \
  python3 - "$RESULTS_FILE" <<'PY'
import json
import os
import sys

path = sys.argv[1]

def env(name: str) -> str:
    return os.environ.get(name, "")

duration = env("RESULT_DURATION_SECONDS")

record = {
    "timestamp": env("RESULT_TIMESTAMP"),
    "run_id": env("RESULT_RUN_ID"),
    "item": env("RESULT_ITEM"),
    "status": env("RESULT_STATUS"),
    "commit": env("RESULT_COMMIT") or None,
    "prompt": env("RESULT_PROMPT") or None,
    "log": env("RESULT_LOG") or None,
    "message": env("RESULT_MESSAGE") or None,
    "started_at": env("RESULT_STARTED_AT") or None,
    "finished_at": env("RESULT_FINISHED_AT") or None,
    "duration_seconds": int(duration) if duration else None,
}

with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, sort_keys=True) + "\n")
PY
}

is_item_completed() {
  local item="$1"

  [[ -f "$RESULTS_FILE" ]] || return 1

  RESULT_ITEM="$item" python3 - "$RESULTS_FILE" <<'PY'
import json
import os
import sys

path = sys.argv[1]
item = os.environ["RESULT_ITEM"]

try:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("item") == item and record.get("status") == "success":
                sys.exit(0)
except FileNotFoundError:
    pass

sys.exit(1)
PY
}

init_run_dir() {
  mkdir -p "$RUNS_ROOT"

  if [[ "$RESUME" == "1" ]]; then
    if [[ -n "$RUN_ID" ]]; then
      RUN_DIR="$RUNS_ROOT/$RUN_ID"

      [[ -d "$RUN_DIR" ]] \
        || die "cannot resume missing run directory: $RUN_DIR"
    else
      [[ -e "$RUNS_ROOT/latest" ]] \
        || die "cannot resume: $RUNS_ROOT/latest does not exist"

      RUN_DIR="$(readlink -f "$RUNS_ROOT/latest")"
      RUN_ID="$(basename "$RUN_DIR")"

      [[ -d "$RUN_DIR" ]] \
        || die "cannot resume: latest does not point to a directory"
    fi
  else
    if [[ -z "$RUN_ID" ]]; then
      RUN_ID="$(date +"%Y%m%d-%H%M%S")"
    fi

    [[ "$RUN_ID" != *"/"* ]] \
      || die "run id must not contain '/': $RUN_ID"

    RUN_DIR="$RUNS_ROOT/$RUN_ID"

    if [[ -s "$RUN_DIR/results.jsonl" ]]; then
      die "run directory already has results: $RUN_DIR. Use --resume --run-id $RUN_ID"
    fi

    mkdir -p "$RUN_DIR"

    if [[ -e "$RUNS_ROOT/latest" && ! -L "$RUNS_ROOT/latest" ]]; then
      die "$RUNS_ROOT/latest exists and is not a symlink"
    fi

    ln -sfn "$RUN_ID" "$RUNS_ROOT/latest"
  fi

  PROMPTS_DIR="$RUN_DIR/prompts"
  ITEM_LOGS_DIR="$RUN_DIR/logs"
  RUN_LOG="$RUN_DIR/run.log"
  RESULTS_FILE="$RUN_DIR/results.jsonl"

  mkdir -p "$PROMPTS_DIR" "$ITEM_LOGS_DIR"
  touch "$RUN_LOG" "$RESULTS_FILE"

  log "Run id: $RUN_ID"
  log "Run directory: $RUN_DIR"
  log "Items file: $ITEMS_FILE"
  log "Prompt template: $PROMPT_TEMPLATE"
  log "Resume: $RESUME"
  log "Skip completed: $SKIP_COMPLETED"
}

run_codex_for_item() {
  local item="$1"
  local line_index="$2"

  if [[ "$VALIDATE_ITEM_PATHS" == "1" ]]; then
    assert_item_path_valid "$item"
  fi

  if [[ "$REQUIRE_CLEAN_BETWEEN_RUNS" == "1" ]]; then
    if git_tree_dirty; then
      log "ERROR: working tree is dirty before processing item: $item"
      git status --short | tee -a "$RUN_LOG"
      return 1
    fi
  fi

  local item_no
  item_no="$(printf '%03d' "$line_index")"

  local slug
  slug="$(slugify "$item")"
  [[ -n "$slug" ]] || slug="item"

  local prompt_path="$PROMPTS_DIR/${item_no}-${slug}.md"
  local item_log="$ITEM_LOGS_DIR/${item_no}-${slug}.log"

  render_prompt "$item" > "$prompt_path"

  local started_at
  local finished_at
  local start_epoch
  local end_epoch
  local duration_seconds
  local before_head
  local after_head
  local commit=""

  started_at="$(now_iso)"
  start_epoch="$(date +%s)"
  before_head="$(git rev-parse HEAD)"

  log "Starting item: $item"
  log "Rendered prompt: $prompt_path"
  log "Item log: $item_log"

  if {
    echo "=== Codex run for item: $item ==="
    echo "Started at: $started_at"
    echo "Prompt: $prompt_path"
    echo

    codex --ask-for-approval "$CODEX_APPROVAL" exec \
      --cd . \
      --sandbox "$CODEX_SANDBOX" \
      < "$prompt_path"
  } 2>&1 | tee -a "$RUN_LOG" "$item_log"; then
    after_head="$(git rev-parse HEAD)"

    if [[ "$after_head" != "$before_head" ]]; then
      commit="$after_head"
    fi

    finished_at="$(now_iso)"
    end_epoch="$(date +%s)"
    duration_seconds="$((end_epoch - start_epoch))"

    if [[ "$REQUIRE_CLEAN_BETWEEN_RUNS" == "1" ]] && git_tree_dirty; then
      append_result \
        "$item" \
        "failed" \
        "$commit" \
        "$prompt_path" \
        "$item_log" \
        "codex completed but left the working tree dirty" \
        "$started_at" \
        "$finished_at" \
        "$duration_seconds"

      log "ERROR: Codex completed but left the working tree dirty after item: $item"
      git status --short | tee -a "$RUN_LOG"
      return 1
    fi

    append_result \
      "$item" \
      "success" \
      "$commit" \
      "$prompt_path" \
      "$item_log" \
      "completed successfully" \
      "$started_at" \
      "$finished_at" \
      "$duration_seconds"

    log "Finished item successfully: $item"

    if [[ -n "$commit" ]]; then
      log "Commit: $commit"
    else
      log "Commit: none"
    fi

    return 0
  else
    local exit_code="$?"

    after_head="$(git rev-parse HEAD)"

    if [[ "$after_head" != "$before_head" ]]; then
      commit="$after_head"
    fi

    finished_at="$(now_iso)"
    end_epoch="$(date +%s)"
    duration_seconds="$((end_epoch - start_epoch))"

    append_result \
      "$item" \
      "failed" \
      "$commit" \
      "$prompt_path" \
      "$item_log" \
      "codex exec failed with exit code $exit_code" \
      "$started_at" \
      "$finished_at" \
      "$duration_seconds"

    log "ERROR: Codex failed for item: $item"
    log "Exit code: $exit_code"

    return "$exit_code"
  fi
}

parse_args() {
  local positional=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --resume)
        RESUME=1
        shift
        ;;

      --run-id)
        [[ $# -ge 2 ]] || die "--run-id requires a value"
        RUN_ID="$2"
        shift 2
        ;;

      --skip-completed)
        SKIP_COMPLETED=1
        SKIP_COMPLETED_EXPLICIT=1
        shift
        ;;

      --no-skip-completed)
        SKIP_COMPLETED=0
        SKIP_COMPLETED_EXPLICIT=1
        shift
        ;;

      --start-after)
        [[ $# -ge 2 ]] || die "--start-after requires a value"
        START_AFTER="$2"
        shift 2
        ;;

      --only)
        [[ $# -ge 2 ]] || die "--only requires a value"
        ONLY_ITEM="$2"
        shift 2
        ;;

      --max-items)
        [[ $# -ge 2 ]] || die "--max-items requires a value"
        MAX_ITEMS="$2"
        [[ "$MAX_ITEMS" =~ ^[0-9]+$ ]] || die "--max-items must be a non-negative integer"
        shift 2
        ;;

      -h|--help)
        usage
        ;;

      --)
        shift
        positional+=("$@")
        break
        ;;

      -*)
        die "unknown option: $1"
        ;;

      *)
        positional+=("$1")
        shift
        ;;
    esac
  done

  if [[ "${#positional[@]}" -gt 2 ]]; then
    die "too many positional arguments"
  fi

  ITEMS_FILE="${positional[0]:-loops/inputs/modules.txt}"
  PROMPT_TEMPLATE="${positional[1]:-loops/prompts/refactor-public-api.prompt.md}"

  if [[ "$RESUME" == "1" && "$SKIP_COMPLETED_EXPLICIT" == "0" ]]; then
    SKIP_COMPLETED=1
  fi
}

main() {
  parse_args "$@"

  assert_project_root

  [[ -f "$ITEMS_FILE" ]] \
    || die "input file not found: $ITEMS_FILE"

  [[ -f "$PROMPT_TEMPLATE" ]] \
    || die "prompt template not found: $PROMPT_TEMPLATE"

  command -v codex >/dev/null 2>&1 \
    || die "codex command not found in PATH"

  command -v python3 >/dev/null 2>&1 \
    || die "python3 command not found in PATH"

  init_run_dir

  if [[ "$REQUIRE_CLEAN_START" == "1" ]]; then
    assert_clean_git_tree "before starting"
  fi

  local line_index=0
  local processed_count=0
  local start_after_seen=0
  local start_after_found=0

  if [[ -z "$START_AFTER" ]]; then
    start_after_seen=1
  fi

  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line_index="$((line_index + 1))"

    local item
    item="$(printf '%s' "$raw_line" | trim_and_strip_comment)"

    [[ -z "$item" ]] && continue

    if [[ -n "$ONLY_ITEM" && "$item" != "$ONLY_ITEM" ]]; then
      continue
    fi

    if [[ "$start_after_seen" == "0" ]]; then
      if [[ "$item" == "$START_AFTER" ]]; then
        start_after_seen=1
        start_after_found=1
        log "Found --start-after marker, skipping marker item: $item"
      fi

      continue
    fi

    if [[ "$SKIP_COMPLETED" == "1" ]] && is_item_completed "$item"; then
      log "Skipping completed item: $item"

      append_result \
        "$item" \
        "skipped" \
        "" \
        "" \
        "" \
        "already completed in this run" \
        "$(now_iso)" \
        "$(now_iso)" \
        "0"

      continue
    fi

    if [[ -n "$MAX_ITEMS" && "$processed_count" -ge "$MAX_ITEMS" ]]; then
      log "Reached --max-items limit: $MAX_ITEMS"
      break
    fi

    processed_count="$((processed_count + 1))"

    if ! run_codex_for_item "$item" "$line_index"; then
      log "Stopping after failed item: $item"
      exit 1
    fi
  done < "$ITEMS_FILE"

  if [[ -n "$START_AFTER" && "$start_after_found" == "0" ]]; then
    die "--start-after item was not found: $START_AFTER"
  fi

  log "All requested Codex runs completed."
  log "Processed runnable items: $processed_count"
  log "Results: $RESULTS_FILE"
}

main "$@"
