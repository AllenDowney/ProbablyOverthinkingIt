# pip install jupyter-book
# pip install ghp-import

# Build the Jupyter book version

# copy the notebooks
cp ../notebooks/gaussian.ipynb .
cp ../notebooks/inspection.ipynb .
cp ../notebooks/preston.ipynb .
cp ../notebooks/lognormal.ipynb .
cp ../notebooks/nbue.ipynb .
cp ../notebooks/berkson.ipynb .
cp ../notebooks/overton_irt.ipynb .

# build the HTML version
jb build .

# push it to GitHub
ghp-import -n -p -f _build/html
