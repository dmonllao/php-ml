#!/bin/bash

exitcode=0

echo "Fixing src/ folder"
php-cs-fixer fix src/
if [ "$?" -ne "0" ]; then
    exitcode=1
fi

echo "Fixing tests/ folder"
php-cs-fixer fix tests/
if [ "$?" -ne "0" ]; then
    exitcode=1
fi

exit $exitcode
