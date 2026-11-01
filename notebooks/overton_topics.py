"""Topic groups for the Overton question selection (draft for review).

At most one question per topic is selected: the highest ranked. Hand
assigned, because correlation doesn't capture the redundancy we care
about (eqwlth and goveqinc, near-duplicates, have r = 0.58; homosex and
premarsx, separate topics in the book, have r = 0.72).

Questions not listed are their own topic.
"""

TOPICS = {
    "abortion": "abany abdefect abnomore abpoor abrape absingle abhlth",
    "welfare and the poor": "helppoor natfare natfarey",
    "size of government": "helpnot",
    "health care": "helpsick natheal nathealy",
    "income inequality": "eqwlth goveqinc",
    "government help for Black Americans": "helpblk natrace natracey",
    "causes of racial inequality": "racdif1 racdif2 racdif3 racdif4",
    "affirmative action": "affrmact discaff",
    "racial stereotypes": "wlthblks wlthwhts workblks workwhts intlblks intlwhts",
    "teen sexuality": "teensex pillok",
    "end of life": "letdie1 suicide1",
    "suicide": "suicide2 suicide3 suicide4",
    "police use of force": "polhitok polabuse polmurdr polescap polattak",
    "working mothers": "fechld fepresch",
    "women in politics": "fepol fepres",
    "child-rearing values": "obey popular thnkself workhard helpoth",
    "social trust": "trust fair helpful",
    "civil liberties: homosexual": "spkhomo colhomo libhomo",
    "civil liberties: communist": "spkcom colcom libcom",
    "civil liberties: anti-religionist": "spkath colath libath",
    "civil liberties: militarist": "spkmil colmil libmil",
    "civil liberties: racist": "spkrac colrac librac",
    "civil liberties: Muslim clergy": "spkmslm colmslm libmslm",
}

TOPIC_OF = {v: topic for topic, names in TOPICS.items() for v in names.split()}


def topic(variable):
    return TOPIC_OF.get(variable, variable)
