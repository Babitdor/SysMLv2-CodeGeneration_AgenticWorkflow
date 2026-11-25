#!/bin/sh
set -eu

for directory in users state uploads current artifacts; do
    path="/workspace/$directory"
    test ! -L "$path"
    mkdir -p "$path"
    chown 10001:10001 "$path"
    chmod u+rwx "$path"
done
