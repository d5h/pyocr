#!/bin/bash

tn="$(mktemp -t n.XXXXXX)" || exit 1
ty="$(mktemp -t y.XXXXXX)" || exit 1
tg="$(mktemp -t g.XXXXXX)" || exit 1

egrep '^n' "$1" | cut -d, -f2- >"$tn"
egrep '^y' "$1" | cut -d, -f2- >"$ty"

cat >"$tg" <<EOF
set xlabel "x"
set ylabel "y"
plot "$tn", "$ty"
EOF

echo "Working around crappy gnuplot / scim bug..."
echo "Close graph window and hit Ctrl+D to exit!"
echo
cat "$tg" - | gnuplot
rm "$tn" "$ty" "$tg"
