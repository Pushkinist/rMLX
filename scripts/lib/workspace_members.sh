# The workspace members of a Cargo.toml, parsed and normalised once for the tools that read them.
#
# Sourced, not executed.
#
# `workspace_members <Cargo.toml>` prints one member per line, normalised, for the
# `[workspace] members = [ ... ]` array in its one-member-per-line layout. A member is
# a path, and the same crate is spelled several ways: `crates/fa/`, `./crates/fa`,
# `crates//fa`, `crates/./fa`. Every path a tool builds from a member and compares with
# another path (the files the classifier lists, the directories the partition producer
# reads) must be one spelling, so the spelling is settled here and nowhere else:
#
#   * a `#` comment, a whole line or the end of one, is dropped before anything else is
#     read: a `]` in it does not end the array. A `#` inside a quoted path is part of
#     the path;
#   * a leading `./`, a trailing `/` and a trailing `/.` are dropped, repeatedly;
#   * `//` and `/./` inside the path become `/`, repeatedly;
#   * a member that is then empty or `.` is the root member, written `.`.
#
# No spelling is refused: the plausibility floor of the classifier compares the parsed
# members with the crates on disk. `member_prefix <member>` is what a path under the member
# starts with, relative to the root: empty for the root member, `<member>/` otherwise.

# normalize_member <member>: the member in the one spelling.
normalize_member() {
    local m="$1" prev=""
    while [ "$m" != "$prev" ]; do
        prev="$m"
        m="${m#./}"
        m="${m%/}"
        m="${m%/.}"
        m="${m//\/\///}"
        m="${m//\/.\///}"
    done
    case "$m" in
    "" | .) m=. ;;
    esac
    printf '%s' "$m"
}

# member_prefix <member>: what a path under the member starts with, relative to the root.
member_prefix() {
    if [ "$1" = . ]; then
        printf ''
    else
        printf '%s/' "$1"
    fi
}

# workspace_members <Cargo.toml>: the members, one per line, normalised.
workspace_members() {
    local m
    awk '
        /^[[:space:]]*members[[:space:]]*=[[:space:]]*\[/ { inm = 1; next }
        inm {
            line = $0
            quoted = 0
            for (k = 1; k <= length(line); k++) {
                c = substr(line, k, 1)
                if (c == "\"") quoted = !quoted
                else if (c == "#" && !quoted) { line = substr(line, 1, k - 1); break }
            }
            if (line ~ /\]/) { inm = 0 }
            gsub(/[",]/, "", line)
            gsub(/[[:space:]]/, "", line)
            gsub(/\[/, "", line)
            gsub(/\]/, "", line)
            if (line != "") { print line }
        }
    ' "$1" | while IFS= read -r m; do
        printf '%s\n' "$(normalize_member "$m")"
    done
}
