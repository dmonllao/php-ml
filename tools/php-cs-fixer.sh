#!/bin/bash

exitcode=0

echo "Fixing src/ folder"
php-cs-fixer fix src/

csexitcode="$?"
if [ "$csexitcode" != "0" ]; then
    exitcode=$csexitcode
fi

echo "Fixing tests/ folder"
php-cs-fixer fix tests/

csexitcode="$?"
if [ "$csexitcode" != "0" ]; then
    # Overwrite the previous exitcode yes, there are worst things in life.
    exitcode=$csexitcode
fi

# Forward php-cs-fixer exit codes.
exit $exitcode
