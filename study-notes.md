# Study notes by Olivier

## Run ess.ipynb

- Write functions for things have to do repeatedly, like select columns
- When importing data file, see if some variables need to convert to categorical
- Check NANs, and replace them properly with `df.variable.replace()`, check `inplace=True` if needed to save it in the original data
- After replacing NANs, check the `value_counts().sort_index()` to see the values are replaced correctly
- Demean some variables for better explanation (why? still need to look into it)
- Look for exceptional values in variables, replace them with more sensible value if needed ( **not a good idea**, using rank is another option)
- Generate new variables with exist variables like this: `df['hasrelig'] = (df.rlgblg==1).astype(int)`, the data type can be changed on spot with `.astype()`
- The `fill_var()` function is interesting, it drops the NANs and resample the data to get a sample of the same size
- Learn to use functions like `extract_res()`, access the regression results and make easy printouts
- Learn to use the **try-except** method

There seems to be a little mistake in the notebook, the `formula` has to be redefined before performing national regression, since `smf.logit` only take 0-1 variables

