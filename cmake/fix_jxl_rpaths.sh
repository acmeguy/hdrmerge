#!/bin/bash
# Rewrite Homebrew/rpath dylib references to @executable_path/../Frameworks/
# in the main executable and all bundled dylibs.
#
# Handles version mismatches: a load command referencing libjxl.0.11.dylib
# matches a bundled libjxl.0.11.1.dylib via prefix matching.
#
# Usage: fix_jxl_rpaths.sh <executable> <frameworks_dir>

set -e

EXE="$1"
FW_DIR="$2"

if [ -z "$EXE" ] || [ -z "$FW_DIR" ]; then
    echo "Usage: $0 <executable> <frameworks_dir>" >&2
    exit 1
fi

fix_paths() {
    local binary="$1"
    otool -L "$binary" | tail -n +2 | awk '{print $1}' | while read -r dep; do
        # Only fix Homebrew absolute paths and @rpath references
        case "$dep" in
            /opt/homebrew/*|@rpath/*)
                local dep_base
                dep_base=$(basename "$dep")
                # Strip .dylib, use remainder as prefix to find bundled file
                local prefix="${dep_base%.dylib}"
                local match
                match=$(ls "$FW_DIR"/${prefix}*.dylib 2>/dev/null | head -1)
                if [ -n "$match" ]; then
                    local name
                    name=$(basename "$match")
                    local new_path="@executable_path/../Frameworks/$name"
                    if [ "$dep" != "$new_path" ]; then
                        install_name_tool -change "$dep" "$new_path" "$binary" 2>/dev/null || true
                    fi
                fi
                ;;
        esac
    done
}

# Fix the main executable
fix_paths "$EXE"

# Fix cross-references between bundled dylibs
for lib in "$FW_DIR"/*.dylib; do
    [ -f "$lib" ] || continue
    fix_paths "$lib"
done
