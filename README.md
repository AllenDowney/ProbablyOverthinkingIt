# Probably Overthinking It

This site contains the Jupyter notebooks I used in my preparation of *[Probably Overthinking It: How to Use Data to Answer Questions, Avoid Statistical Traps, and Make Better Decisions](https://greenteapress.com/wp/probably-overthinking-it/)*.

If you've read the book and you want to know how the analysis works, this is the place.
And if you read these notebooks, and you want to get the book, you can order from [Bookshop.org](https://bookshop.org/a/98697/9780226822587) and [Amazon](https://amzn.to/3Kp629E) (affiliate links).

Before you read these notebooks, please keep in mind:

* There is some explanatory text in the notebooks, but some of the examples will not make sense until you have read the corresponding chapter in the book.

* While preparing these notebooks, I made some changes to improve the readability of the code. There might be small differences between what appears in the book and what you get when you run the code.

**Chapter 1: Are You Normal? Hint: No.**

[Run the code on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/gaussian.ipynb)

[Run the code that prepares the BRFSS data](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/clean_brfss.ipynb)

[Run the code that prepares the Big Five data](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/clean_big_five.ipynb)


**Chapter 2: Relay Races and Revolving Doors**

[Run the code on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/inspection.ipynb)


**Chapter 3: Defy Tradition, Save the World**

[Run the code on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/preston.ipynb)


**Chapter 4: Extremes, Outliers, and GOATs**

[Run the code on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/lognormal.ipynb)

[Run the code that prepares the BRFSS data](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/clean_brfss.ipynb)

[Run the code that prepares the NSFG data](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/nsfg_clean.ipynb)


**Chapter 5: Bettter Than New**

[Run the code on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/nbue.ipynb)


**Chapter 6: Jumping to Conclusions**

[Run the code on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/berkson.ipynb)


**Chapter 7: Causation, Collision, and Confusion**

[Run the code on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/birthweight.ipynb)

[Run the code that prepares the NCHS data](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/clean_nchs.ipynb)


**Chapter 8: The Long Tail of Disaster**

[Run the code on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/longtail.ipynb)

[Run the code that prepares the earthquake data](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/clean_quake.ipynb)

[Run the code that prepares the solar flare data](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/clean_flare.ipynb)

**Chapter 9: Fairness and Fallacy**

[Run the code on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/base_rate.ipynb)


**Chapter 10: Penguins, Pessimists, and Paradoxes**

[Run the code on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/simpson.ipynb)

[Run the code that prepares the GSS data](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/clean_simpson.ipynb)


**Chapter 11: Changing Hearts and Minds**

[Run the code on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/progress.ipynb)


**Chapter 12: Chasing the Overton Window**

[Run the code on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/notebooks/overton.ipynb)

## Getting Started

### Downloading the Repository

To download this repository and run the notebooks locally:

```bash
git clone https://github.com/AllenDowney/ProbablyOverthinkingIt.git
cd ProbablyOverthinkingIt
```

### Setting Up the Environment

If you have conda installed, you can create the environment using the Makefile:

```bash
make create_environment
conda activate ProbablyOverthinkingIt
make requirements
```

The Makefile installs both the main requirements and development dependencies.

### Running the Notebooks

Once you have the environment set up, you can start Jupyter:

```bash
jupyter notebook
```

Or if you prefer JupyterLab:

```bash
jupyter lab
```

Then navigate to the `notebooks/` directory to find and run the notebooks for each chapter.

### Data Files

The notebooks use data files stored in the `data/` directory. These files are included in the repository and should be available when you run the notebooks locally.

### Alternative: Using Google Colab

If you prefer not to set up a local environment, you can run the notebooks directly in Google Colab using the links provided above for each chapter.

## Repository Sitemap

### Main Directories

- **`notebooks/`** - Main Jupyter notebooks for each chapter of the book
- **`data/`** - Data files used by the notebooks (CSV, HDF5, Excel files)
- **`examples/`** - Additional examples and supplementary notebooks
- **`jb/`** - Jupyter Book configuration and built documentation

### Configuration Files

- **`environment.yml`** - Conda environment specification
- **`requirements.txt`** - Python package requirements
- **`requirements-dev.txt`** - Development dependencies
- **`Makefile`** - Build automation and environment setup
- **`LICENSE`** - Project license

