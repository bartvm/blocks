SUCCESS=0
flake8 --filename=* bin || SUCCESS=1
flake8 blocks doctests tests || SUCCESS=1
pep257 bin --numpy --ignore=D100,D101,D102,D103 --match='.*' || SUCCESS=1
pep257 blocks --numpy --ignore=D100,D101,D102,D103 || SUCCESS=1
pep257 doctests tests --numpy --ignore=D100,D101,D102,D103 --match='.*\.py' || SUCCESS=1
exit $SUCCESS
