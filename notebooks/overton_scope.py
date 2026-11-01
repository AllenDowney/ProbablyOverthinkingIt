"""Which candidates are in scope for the Overton question selection.

The candidates are the variables that pass the easy gates in
overton_candidates.py. This module classifies them as:

- opinion: a view on a public question or social norm; in scope.
- borderline: needed a decision; the decisions are in BORDERLINE_EXCLUDED
  (everything else in BORDERLINE is in scope).
- excluded, with a reason: demographic or household, behavior or
  experience, religion, personal circumstances or well-being,
  expectation, knowledge, interview or administrative, circular.

The lists are explicit so every decision can be reviewed. No responses or
trends are used here.

Writes overton_figs/candidates_scope.csv.

Usage: python overton_scope.py
"""

import pandas as pd

from overton_pipeline import FIG_DIR

OPINION = """
natspac natenvir natheal natcity natcrime natdrug nateduc natrace natarms nataid natfare
natroad natsoc natmass natpark natchld natsci
eqwlth helppoor helpnot helpsick helpblk jobsall goveqinc
spkath colath libath spkrac colrac librac spkcom colcom libcom spkmil colmil libmil
spkhomo colhomo libhomo spkmslm colmslm libmslm
cappun gunlaw courts wirtap grass prayer
racmar racpush racseg racopen racfew busing racpres racchng affrmact wrkwayup
racdif1 racdif2 racdif3 racdif4 liveblks livewhts marblk marasian marhisp marwht discaff
fehome fework fepres fepol fechld fehelp fepresch fefam fejobaff fehire meovrwrk
abdefect abnomore abhlth abpoor abrape absingle abany
pillok sexeduc divlaw premarsx teensex xmarsex homosex pornlaw porninf pornmorl pornrape pornout
spanking letdie1 suicide1 suicide2 suicide3 suicide4
hitok hitmarch hitdrunk hitchild hitbeatr hitrobbr polhitok polabuse polmurdr polescap polattak
letin1a usintl usun commun
confinan conbus conclerg coneduc confed conlabor conpress conmedic contv conjudge consci
conlegis conarmy
""".split()

BORDERLINE = {
    # alternate wordings of the spending items, asked on other ballots
    **{
        v: "alternate wording of a spending item (split ballot)"
        for v in "natspacy natenviy nathealy natcityy natcrimy natdrugy nateducy natracey natarmsy nataidy natfarey".split()
    },
    "trust": "social trust",
    "fair": "social trust",
    "helpful": "social trust",
    "obey": "child-rearing values",
    "popular": "child-rearing values",
    "thnkself": "child-rearing values",
    "workhard": "child-rearing values",
    "helpoth": "child-rearing values",
    "aged": "family norm",
    "getahead": "belief about how people get ahead",
    "anomia5": "anomie",
    "anomia6": "anomie",
    "anomia7": "anomie",
    "world1": "worldview",
    "wlthwhts": "group stereotype",
    "wlthblks": "group stereotype",
    "workwhts": "group stereotype",
    "workblks": "group stereotype",
    "intlwhts": "group stereotype",
    "intlblks": "group stereotype",
    "discaffm": "perceived likelihood of discrimination",
    "discaffw": "perceived likelihood of discrimination",
    "nextgen": "attitude toward science",
    "toofast": "attitude toward science",
    "advfront": "attitude toward science",
    "balpos": "attitude toward science",
    "tax": "own taxes too high",
    "rachaf": "follow-up asked depending on the answer to racfew",
    "racmost": "follow-up asked depending on the answer to racfew",
}

# decisions on the borderline items (2026-10-30): all in scope except the
# racfew follow-ups
BORDERLINE_EXCLUDED = ["rachaf", "racmost"]

# alternate wordings asked on different ballots, so no one answers both and
# correlation can't detect redundancy; for redundancy they count as one topic
SAME_TOPIC = [
    ("natspac", "natspacy"),
    ("natenvir", "natenviy"),
    ("natheal", "nathealy"),
    ("natcity", "natcityy"),
    ("natcrime", "natcrimy"),
    ("natdrug", "natdrugy"),
    ("nateduc", "nateducy"),
    ("natrace", "natracey"),
    ("natarms", "natarmsy"),
    ("nataid", "nataidy"),
    ("natfare", "natfarey"),
]

EXCLUDED = {
    "religion": """denom fund reliten postlife pray mapa mastersp judgeluv frndking popespks denom16
        fund16 spden spfund sprel16 spfund16 bible god reborn savesoul relpersn sprtprsn reltrad
        reltrad16""",
    "behavior or experience": """drink drunk smoke quitsmk evsmoke hit hitage gun gunage burglr robbry
        owngun pistol shotgun rifle rowngun hunt news xmovie racclos raclive racdis racinteg rachome
        racchurh racwork socrel socommun socfrend socbar matesex frndsex acqntsex pikupsex paidsex
        othersex sexsex sexfreq sexsex5 evpaidsx evstray condom relatsex evidu evcrack hivtest hivtest2
        compuse usewww memfrat memserv memvet mempolit memunion memsport memyouth memschl memhobby
        memgreek memnat memfarm memlit memprof memchurh memother fear divorce5 hosdis5 padeath madeath
        chlddth sibdeath spdeath trauma1 trauma5 union""",
    "personal circumstances or well-being": """happy hapmar health life satcity sathobby satfam satfrnd
        sathealt joblose jobfind satjob richwork jobinc jobsec jobhour jobpromo jobmeans class satfin
        finalter finrela parsol kidssol goodlife stress trynewjb chldmore""",
    "expectation": "uswar uswary",
    "knowledge": "worda wordb wordc wordd worde wordf wordg wordh wordi wordj viruses",
    "circular": "vote_ts pres_ts",
}


def classify(variable):
    if variable in OPINION:
        return "opinion", ""
    if variable in BORDERLINE:
        if variable in BORDERLINE_EXCLUDED:
            return "excluded", BORDERLINE[variable]
        return "opinion", "borderline, included: " + BORDERLINE[variable]
    for reason, names in EXCLUDED.items():
        if variable in names.split():
            return "excluded", reason
    return "excluded", "demographic, household, interview, or administrative"


def main():
    table = pd.read_csv(FIG_DIR / "candidates_pass1.csv", index_col=0)
    table = table[table["passes"]].copy()
    scope = [classify(v) for v in table.index]
    table["scope"] = [s for s, _ in scope]
    table["scope_note"] = [n for _, n in scope]

    unknown = set(OPINION) - set(table.index)
    assert not unknown, unknown

    table.to_csv(FIG_DIR / "candidates_scope.csv")
    print(table["scope"].value_counts())
    print(table.loc[table["scope"] == "excluded", "scope_note"].value_counts())


if __name__ == "__main__":
    main()
