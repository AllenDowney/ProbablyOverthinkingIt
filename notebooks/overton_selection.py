"""Question selection for the Overton update.

Step 0: rerun the book's find_questions() ranking on the book's data, to
document how the original fifteen questions ranked. Same candidate list
and model as Chapter 12: for each variable, code y = 1 for the most common
response, fit logit(y ~ year + cohort + polviews), and rank by the absolute
value of the polviews coefficient.

Writes overton_figs/selection_step0.csv.

Usage: python overton_selection.py
"""

import pandas as pd
import statsmodels.formula.api as smf

from overton_pipeline import BOOK_FILE, FIG_DIR, ORDER, load_gss
from utils import prepare_yvar

# the candidate list from Chapter 12 (overton.md)
BOOK_CANDIDATES = [
    "abany", "abdefect", "abhlth", "abnomore", "abpoor", "abrape", "absingle",
    "affrmact", "bible", "cappun", "colath", "colcom", "colhomo", "colmil",
    "colmslm", "colrac", "compuse", "conarmy", "conbus", "conclerg", "coneduc",
    "confed", "confinan", "conjudge", "conlabor", "conlegis", "conmedic",
    "conpress", "consci", "contv", "discaffm", "discaffw", "divlaw", "divorce",
    "fair", "fear", "fechld", "fefam", "fehire", "fejobaff", "fepol",
    "fepresch", "finrela", "fund", "god", "grass", "gunlaw", "hapmar", "happy",
    "health", "helpful", "homosex", "hunt", "letdie1", "libath", "libcom",
    "libhomo", "libmil", "libmslm", "librac", "life", "meovrwrk", "nataid",
    "natarms", "natchld", "natcity", "natcrime", "natdrug", "nateduc",
    "natenrgy", "natenvir", "natfare", "natheal", "natmass", "natpark",
    "natrace", "natroad", "natsci", "natsoc", "natspac", "pornlaw", "postlife",
    "pray", "prayer", "premarsx", "pres04", "pres08", "pres12", "racmar",
    "racopen", "racpres", "reborn", "relexp", "reliten", "relpersn", "satfin",
    "satjob", "savesoul", "sexeduc", "spanking", "spkath", "spkcom", "spkhomo",
    "spkmil", "spkmslm", "spkrac", "sprtprsn", "teensex", "trust", "union_",
    "xmarsex", "owngun", "pistol", "attend", "relactiv",
]


def find_questions(gss, varnames):
    """The book's ranking: logit of the modal response on year, cohort, polviews."""
    rows = []
    for varname in varnames:
        if varname not in gss:
            rows.append(dict(question=varname, missing=True))
            continue
        prepare_yvar(gss, varname)
        subset = gss.dropna(subset=[varname, "polviews"])
        results = smf.logit("y ~ year + cohort + polviews", data=subset).fit(disp=False)
        years = subset["year"]
        rows.append(
            dict(
                question=varname,
                param=results.params["polviews"],
                abs_param=abs(results.params["polviews"]),
                modal_value=gss[varname].value_counts().idxmax(),
                n=len(subset),
                waves=years.nunique(),
                first=years.min(),
                last=years.max(),
                in_book=varname in ORDER,
                missing=False,
            )
        )
    table = pd.DataFrame(rows).set_index("question")
    table = table.sort_values("abs_param", ascending=False)
    table["rank"] = range(1, len(table) + 1)
    return table


def main():
    gss = load_gss(BOOK_FILE, "gss0")
    table = find_questions(gss, BOOK_CANDIDATES)
    table.to_csv(FIG_DIR / "selection_step0.csv")
    with pd.option_context("display.width", 150, "display.max_rows", 200):
        print(table.round(3))


if __name__ == "__main__":
    main()
