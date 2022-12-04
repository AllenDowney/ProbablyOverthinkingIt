# pip install jupyter-book
# pip install ghp-import

# Build the Jupyter book version

# copy the notebooks
cp ../notebooks/gaussian.ipynb .
cp ../notebooks/overton_irt.ipynb .

# build the HTML version
jb build .

# push it to GitHub
ghp-import -n -p -f _build/html
