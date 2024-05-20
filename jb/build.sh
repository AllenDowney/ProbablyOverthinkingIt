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
cp ../notebooks/birthweight.ipynb .
cp ../notebooks/longtail.ipynb .
cp ../notebooks/base_rate.ipynb .
cp ../notebooks/simpson.ipynb .
cp ../notebooks/progress.ipynb .
cp ../notebooks/overton.ipynb .
cp ../notebooks/overton_irt.ipynb .
cp ../examples/ansur_pca.ipynb .
cp ../examples/bertrand.ipynb .
cp ../examples/ideology_gap.ipynb .

# build the HTML version
jb build .

# push it to GitHub
ghp-import -n -p -f _build/html
