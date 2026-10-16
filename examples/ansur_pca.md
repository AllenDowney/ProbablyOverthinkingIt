---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# The Principal Components of Anthropometry

In a previous post I explored the correlations between measurements in the ANSUR-II dataset, which includes 93 measurements from a sample of U.S. military personnel.
I found that measurements of the head were weakly correlated with measurements from other parts of the body -- and in particular the protrusion of the ears is almost entirely uncorrelated with anything else.

A friend of mine, and co-developer of the [Modeling and Simulation](https://greenteapress.com/wp/modsimpy/) class I taught at Olin, asked whether I had tried running principal component analysis (PCA).
I had not, but now I have. Let's look at the results.

<!-- #region tags=["remove-cell"] -->
This notebook is an offshoot of Chapter 1 of [*Probably Overthinking It: How to Use Data to Answer Questions, Avoid Statistical Traps, and Make Better Decisions*](https://probablyoverthinking.it).
<!-- #endregion -->

[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ProbablyOverthinkingIt/blob/book/examples/ansur_pca.ipynb).

```python tags=["remove-cell"]
# Install empirical dist if we don't already have it

try:
    import empiricaldist
except ImportError:
    !pip install empiricaldist
```

```python tags=["remove-cell"]
# download utils.py

from os.path import basename, exists

def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print('Downloaded ' + local)
        
download("https://github.com/AllenDowney/ProbablyOverthinkingIt/raw/book/notebooks/utils.py")
```

```python tags=["hide-cell"]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## The data

Here's the ANSUR data, originally downloaded from [The OPEN Design Lab](https://www.openlab.psu.edu/ansur2/).

```python tags=["remove-cell"]
download("https://github.com/AllenDowney/ProbablyOverthinkingIt/raw/book/data/ANSURIIFEMALEPublic.csv")
download("https://github.com/AllenDowney/ProbablyOverthinkingIt/raw/book/data/ANSURIIMALEPublic.csv")
```

```python tags=["hide-cell"]
ansur_female = pd.read_csv("ANSURIIFEMALEPublic.csv", encoding="ISO-8859-1")
ansur_female.shape
```

```python tags=["hide-cell"]
ansur_male = pd.read_csv("ANSURIIMALEPublic.csv", encoding="ISO-8859-1")
ansur_male.shape
```

Since heights are in mm, I converted to cm.

```python tags=["hide-cell"]
height_male = ansur_male["stature"] / 10
height_female = ansur_female["stature"] / 10
```

```python tags=["hide-cell"]
def add_earratio(df):
    df['earratio'] = df['earprotrusion'] / df['earbreadth']
```

```python tags=["hide-cell"]
measurements = [column for column in ansur_female.columns if column.lower() == column]
measurements
```

```python tags=["hide-cell"]
len(measurements)
```

```python tags=["hide-input"]
ansur_male[measurements]
```

```python tags=["hide-cell"]
df = ansur_male[measurements]
```

## Principle Component Analysis

As you might guess from the comments below, I used ChatGPT to generate a lot of the code.
It took some revision to get it working, but the head start definitely saved me some time.

```python tags=["hide-cell"]
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In this example, we will first standardize the data and then perform PCA.

# Step 3.1: Create a StandardScaler step
scaler = StandardScaler()

# Step 3.2: Create a PCA step
pca = PCA(n_components=10) 

# Step 4: Create the pipeline
steps = [
    ('scaler', scaler),  # Standardize the data
    ('pca', pca)        # Perform PCA
]

pipeline = Pipeline(steps)

# Fit the pipeline to your data
pipeline.fit(df)

# Now, you can access the transformed data and principal components if needed.
transformed_data = pipeline.transform(df)  # Transformed data after PCA
principal_components = pca.components_    # Principal components (eigenvectors)

# You can also access the explained variance ratio to see how much variance is explained by each component.
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_ratio
```

Here's a visualization of explained variance versus number of components.
I lifted this from [The Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html).

```python tags=["hide-cell"]
ys = np.cumsum(pca.explained_variance_ratio_)
xs = np.arange(len(ys)) + 1
ys
```

```python tags=["hide-input"]
plt.bar(xs, ys)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

plt.savefig('ansur_pca1.png', dpi=300)
```

With one component, we can capture 44% of the variation in the measurements.
With two components, we're up to 62%.
After that, the gains are smaller (as we expect), but with 10 measurements, we get up to 78%.


## Loadings

Looking at the loadings, we can see which measurements contribute the most to each of the components, so we can get a sense of which characteristics each component captures. 

```python tags=["hide-input"]
loadings = pca.components_  # Get the loadings (eigenvectors)

# Create a DataFrame to store the loadings with feature names as columns
loadings_df = pd.DataFrame(loadings, columns=measurements)
loadings_df
```

```python tags=["hide-input"]
loadings_transposed = loadings_df.transpose()
loadings_transposed
```

```python tags=["hide-cell"]
num_top_features = 5

# Iterate through each principal component
for i, col in loadings_transposed.items():
    print(f"Principal Component {i+1}:")
    
    # Sort the features by absolute loading values for the current principal component
    sorted_features = col.sort_values(key=abs, ascending=False)
    
    # Select the top N features with the highest absolute loadings
    top_features = sorted_features[:num_top_features].round(3)
    
    # Print the top correlated factors for the current principal component
    for name, val in top_features.items():
        print(-val, '\t', name)
    print()
```

I won't explain all of the measurements, but if there are any you are curious about, you can look them up in [*The Measurer’s Handbook*](http://tools.openlab.psu.edu/publicData/ANSURII-TR11-017.pdf) which includes details on "sampling strategy and measuring techniques" as well as descriptions and diagrams of the landmarks and measurements between them.

* Not surprisingly, the first component is loaded with measurements of height. If you want to predict someone's measurements, and can only use one number, choose height.

* The second component is loaded with measurements of girth. Again, no surprises so far.

* The third component seems to capture torso length. And that makes sense, too. One you know how tall someone is, it helps to know how that height is split between torso and legs.

* The fourth component seems to capture hand and foot size (with sitting height thrown in just to remind us that PCA is not obligated to find components that align perfectly with the axes we expect).

* Component 5 is all about the shoulders.

* Component 6 is all about the head.

After that, things are not so neat. But two things are worth noting:

* Component 7 is mostly related to the dimensions of the pelvis, but...

* Components 7, 8, and 10 are surprisingly loaded up with ear measurements.

As we saw in the previous article, there seems to be something special about ears.
Once you have exhausted the information carried by the most obvious measurements, the dimensions of the ear seem to be strangely salient.


## Using the pca module

I found [this module](https://erdogant.github.io/pca/pages/html/index.html) and decided to experiment with it.

```python tags=["remove-cell"]
try:
    from pca import pca
except ImportError:
    !pip install pca
    from pca import pca
```

```python tags=["hide-cell"]
model = pca(normalize=True, n_components=10)
model.fit_transform(df);
```

```python tags=["hide-input"]
model.plot();
```

Unfortunately, the most interesting plots don't seem to work well with the number of dimensions in this dataset.
I'd be curious to know if they can be tuned to work better.

```python tags=["hide-input"]
model.biplot3d();
```

<!-- #region tags=["remove-print"] -->
Copyright 2023 Allen Downey 

The code in this notebook and `utils.py` is under the [MIT license](https://mit-license.org/).
<!-- #endregion -->

```python tags=["hide-cell"]

```
