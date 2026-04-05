#!/usr/bin/env fish

function usage
    echo "Usage: scripts/dataset_caption_edit.fish [options] --pass 'SEARCH=>REPLACE' [--pass 'SEARCH=>REPLACE' ...]"
    echo
    echo "Options:"
    echo "  -d, --dataset=PATH     Dataset path (default: rara)"
    echo "  -o, --output=DIR       Output dir override (default: empty)"
    echo "  -s, --server=URL       ComfyUI server URL (default: http://127.0.0.1:8188)"
    echo "  -a, --apply            Write changes (default is dry-run)"
    echo "  -n, --dry-run          Force dry-run mode"
    echo "  -p, --pass=PAIR        Replacement pair in the form SEARCH=>REPLACE"
    echo "  -h, --help             Show this help"
    echo
    echo "Examples:"
    echo "  scripts/dataset_caption_edit.fish --dataset rara --pass '  ,=>,' --pass '  => '"
    echo "  scripts/dataset_caption_edit.fish --dataset rara --apply --pass 'old=>new'"
end

set -l server "http://127.0.0.1:8188"
set -l dataset_path "rara"
set -l output_dir ""
set -l dry_run true
set -l passes

set -l i 1
while test $i -le (count $argv)
    set -l arg "$argv[$i]"
    switch "$arg"
        case '-h' '--help'
            usage
            exit 0
        case '-a' '--apply'
            set dry_run false
        case '-n' '--dry-run'
            set dry_run true
        case '-d' '--dataset'
            set i (math "$i + 1")
            if test $i -gt (count $argv)
                echo "Error: missing value for $arg" >&2
                exit 2
            end
            set dataset_path "$argv[$i]"
        case '-o' '--output'
            set i (math "$i + 1")
            if test $i -gt (count $argv)
                echo "Error: missing value for $arg" >&2
                exit 2
            end
            set output_dir "$argv[$i]"
        case '-s' '--server'
            set i (math "$i + 1")
            if test $i -gt (count $argv)
                echo "Error: missing value for $arg" >&2
                exit 2
            end
            set server "$argv[$i]"
        case '-p' '--pass'
            set i (math "$i + 1")
            if test $i -gt (count $argv)
                echo "Error: missing value for $arg" >&2
                exit 2
            end
            set passes $passes "$argv[$i]"
        case '--dataset=*'
            set dataset_path (string replace -r '^--dataset=' '' -- "$arg")
        case '--output=*'
            set output_dir (string replace -r '^--output=' '' -- "$arg")
        case '--server=*'
            set server (string replace -r '^--server=' '' -- "$arg")
        case '--pass=*'
            set passes $passes (string replace -r '^--pass=' '' -- "$arg")
        case '*'
            echo "Error: unknown option '$arg'" >&2
            usage
            exit 2
    end
    set i (math "$i + 1")
end

if not command -q jq
    echo "Error: jq is required. Install jq and retry." >&2
    exit 1
end

if test (count $passes) -eq 0
    echo "Error: provide at least one --pass 'SEARCH=>REPLACE'." >&2
    usage
    exit 2
end

for raw_pass in $passes
    if not string match -rq '.+=>.*' -- "$raw_pass"
        echo "Error: invalid --pass format: $raw_pass" >&2
        echo "Expected: SEARCH=>REPLACE" >&2
        exit 2
    end
end

set -l mode_label "DRY-RUN"
set -l dry_run_json true
if test "$dry_run" = "false"
    set mode_label "APPLY"
    set dry_run_json false
end

echo "Running "(count $passes)" pass(es) in $mode_label mode"
echo "Server: $server"
echo "Dataset: $dataset_path"
if test -n "$output_dir"
    echo "Output dir: $output_dir"
end

echo

set -l pass_index 1
for pass_entry in $passes
    set -l parts (string split -m 1 '=>' -- "$pass_entry")
    set -l find_text "$parts[1]"
    set -l replace_text ""
    if test (count $parts) -gt 1
        set replace_text "$parts[2]"
    end

    set -l payload (jq -nc \
        --arg dataset_path "$dataset_path" \
        --arg output_dir "$output_dir" \
        --arg find_text "$find_text" \
        --arg replace_text "$replace_text" \
        --argjson dry_run $dry_run_json \
        '{dataset_path:$dataset_path, output_dir:$output_dir, find_text:$find_text, replace_text:$replace_text, dry_run:$dry_run}')

    set -l response_file (mktemp)
    set -l http_code (curl -sS \
        -o "$response_file" \
        -w "%{http_code}" \
        -X POST "$server/fbtools/dataset_caption/edit" \
        -H "Content-Type: application/json" \
        -d "$payload")

    set -l body (cat "$response_file")
    rm -f "$response_file"

    if test "$http_code" != "200"
        echo "Pass $pass_index failed with HTTP $http_code" >&2
        echo "$body" >&2
        exit 1
    end

    set -l ok (echo "$body" | jq -r '.ok // false')
    if test "$ok" != "true"
        echo "Pass $pass_index returned ok=false" >&2
        echo "$body" | jq . >&2
        exit 1
    end

    set -l edited_count (echo "$body" | jq -r '.edited_count // 0')
    set -l total_txt (echo "$body" | jq -r '.total_txt_files // 0')
    echo "[$pass_index] edited_count=$edited_count total_txt_files=$total_txt"

    if test "$dry_run" = "true"
        set -l preview (echo "$body" | jq -r '.preview_changes[0] | if . then (.before + " => " + .after) else "" end')
        if test -n "$preview"
            echo "    sample: $preview"
        end
    end

    set pass_index (math "$pass_index + 1")
end

echo
echo "Finished $mode_label run successfully."
