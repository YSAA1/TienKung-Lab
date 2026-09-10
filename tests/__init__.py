# Make ``tests`` a real package: IsaacLab sources ship their own regular
# ``tests`` package, which would otherwise shadow this directory as a namespace
# package on PYTHONPATHs that include IsaacLab source roots.
