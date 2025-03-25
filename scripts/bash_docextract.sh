#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 filename.py"
  exit 1
fi

filename="$1"

# Remove cython lines
tmpfile=$(mktemp)
grep -vE '^\s*(cdef|cpdef|cimport|ctypedef)' "$filename" > "$tmpfile"

# Use awk to extract function names and docstrings
awk '
/^def[ \t]+[a-zA-Z0-9_]+\(/ {
    in_func = 1;
    in_doc = 0;
    doc = "";

    # Extract the function name manually
    func = $0;
    sub(/^def[ \t]+/, "", func);
    sub(/\(.*/, "", func);
    next;
}

/^[ \t]*"""/ && in_func {
    count = gsub(/"""/, "&");
    if (count >= 2) {
        line = $0;
        sub(/^[ \t]*"""/, "", line);
        sub(/""".*$/, "", line);
        doc = line;
        print "Function: " func;
        print "Docstring:";
        print doc;
        print "---------------------------------";
        in_func = 0;
    } else {
        in_doc = 1;
        doc = "";
        line = $0;
        sub(/^[ \t]*"""/, "", line);
        doc = line "\n";
    }
    next;
}

in_doc {
    doc = doc $0 "\n";
    if ($0 ~ /"""/) {
        sub(/""".*$/, "", doc);
        print "Function: " func;
        print "Docstring:";
        print doc;
        print "---------------------------------";
        in_doc = 0;
        in_func = 0;
    }
}
' "$tmpfile"

rm "$tmpfile"

