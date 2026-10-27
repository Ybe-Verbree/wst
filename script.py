import re
from functools import lru_cache
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from itertools import combinations


VOORBEELD = "A-, A+, B+, C+, (C-, B-)"

MAX_FACTOR_PER_TOESTAND = 18
MAX_KANDIDAAT_TESTS = 300_000
MAX_DEKKINGSTOESTANDEN = 100_000


# ============================================================
# INVOER EN CILINDERSTANDEN
# ============================================================

def lees_volgorde(tekst):
    tekst = tekst.strip().upper()

    if not tekst:
        raise ValueError("Voer een bewegingsvolgorde in.")

    beweging = r"[A-Z]\s*[+-]"
    groep = rf"\(\s*{beweging}(?:\s*,\s*{beweging})*\s*\)"
    stap = rf"(?:{beweging}|{groep})"
    patroon = rf"{stap}(?:\s*,\s*{stap})*"

    if not re.fullmatch(patroon, tekst):
        raise ValueError(
            "Ongeldige invoer. Gebruik bijvoorbeeld: "
            "A-, A+, B+, C+, (C-, B-). "
            "Scheid stappen met komma's. "
            "Geneste haakjes zijn niet toegestaan."
        )

    onderdelen = re.findall(
        rf"\([^()]*\)|{beweging}",
        tekst,
    )

    stappen = []

    for nummer, onderdeel in enumerate(onderdelen, start=1):
        bewegingen = re.findall(
            r"([A-Z])\s*([+-])",
            onderdeel,
        )

        cilinders = [
            cilinder
            for cilinder, _ in bewegingen
        ]

        if len(cilinders) != len(set(cilinders)):
            raise ValueError(
                f"Stap {nummer}: een cilinder mag maar één "
                "keer in dezelfde stap voorkomen."
            )

        stappen.append(bewegingen)

    return stappen


def staptekst(bewegingen):
    tekst = ", ".join(
        cilinder + richting
        for cilinder, richting in bewegingen
    )

    return f"({tekst})" if len(bewegingen) > 1 else tekst


def bitwaarde(cilinder):
    return 1 << (ord(cilinder) - ord("A"))


def is_cyclisch(data, cilinders):
    return all(
        int(data.iloc[0][cilinder])
        == int(data.iloc[-1][cilinder])
        for cilinder in cilinders
    )


def maak_diagramdata(stappen):
    cilinders = sorted({
        cilinder
        for bewegingen in stappen
        for cilinder, _ in bewegingen
    })

    beginstanden = {}

    for bewegingen in stappen:
        for cilinder, richting in bewegingen:
            if cilinder not in beginstanden:
                beginstanden[cilinder] = (
                    0 if richting == "+" else 1
                )

    posities = dict(beginstanden)

    rijen = [{
        "Stapgrens": 0,
        **posities,
    }]

    for nummer, bewegingen in enumerate(stappen, start=1):
        for cilinder, richting in bewegingen:
            doel = 1 if richting == "+" else 0

            if posities[cilinder] == doel:
                toestand = (
                    "uitgeschoven"
                    if doel == 1
                    else "ingeschoven"
                )

                raise ValueError(
                    f"Stap {nummer}: {cilinder}{richting} "
                    f"is niet mogelijk. Cilinder {cilinder} "
                    f"staat al {toestand}."
                )

        for cilinder, richting in bewegingen:
            posities[cilinder] = (
                1 if richting == "+" else 0
            )

        rijen.append({
            "Stapgrens": nummer,
            **posities,
        })

    data = pd.DataFrame(rijen)

    data["Bitgetal"] = sum(
        data[cilinder] * bitwaarde(cilinder)
        for cilinder in cilinders
    )

    return data, beginstanden, cilinders

# ============================================================
# AUTOMATISCHE FASEBESTURING
# ============================================================

MAX_FASE_STAPPEN = 16


def maak_fasebesturing(basisdata, stappen, cilinders):
    """
    Ontwerp fasebesturing voor een gesloten cyclus.

    Ontwerpmodel:
    - Iedere cilinder komt maximaal één keer in een fase voor.
    - Fase 0 heeft geen eigen geheugen.
    - Fase 1 gebruikt G1, fase 2 gebruikt G2, enzovoort.
    - Geheugens zijn in de bedoelde stabiele toestanden one-hot.
    - SET/RESET wordt verwerkt voordat uitgangen worden bepaald.
    - Alleen beweging op invoerstap 1 krijgt 'st'.

    Optimalisatie:
    1. Minimaal aantal fasen.
    2. Minimaal aantal contactverwijzingen binnen de
       gegenereerde formulevorm.
    """
    n = len(stappen)

    if n > MAX_FASE_STAPPEN:
        raise ValueError(
            "De exacte fasezoeker ondersteunt maximaal "
            f"{MAX_FASE_STAPPEN} stappen."
        )

    if not is_cyclisch(basisdata, cilinders):
        raise ValueError(
            "Deze fasezoeker ondersteunt alleen gesloten cycli. "
            "Laat alle cilinders eindigen in hun beginstand."
        )

    stapcilinders = [
        {cilinder for cilinder, _ in bewegingen}
        for bewegingen in stappen
    ]

    # Een factor is (signaalnaam, positief).
    # Voorbeeld: ('a0', True), ('g1', False), ('st', True).

    def uniek(factoren):
        return list(dict.fromkeys(factoren))

    def latex_factor(factor):
        naam, positief = factor

        if naam == "st":
            symbool = r"\mathrm{st}"
        elif naam.startswith("g"):
            symbool = rf"g_{{{naam[1:]}}}"
        else:
            symbool = rf"{naam[0]}_{{{naam[1:]}}}"

        return (
            symbool
            if positief
            else rf"\overline{{{symbool}}}"
        )

    def latex_term(factoren):
        factoren = uniek(factoren)

        return (
            r" \cdot ".join(
                latex_factor(factor)
                for factor in factoren
            )
            if factoren
            else "1"
        )

    def latex_of(termen):
        termen = list(dict.fromkeys(
            tuple(uniek(term))
            for term in termen
        ))

        if not termen:
            return "0"

        return " + ".join(
            latex_term(term)
            for term in termen
        )

    def eindcontacten(stapindex):
        return [
            (
                f"{cilinder.lower()}"
                f"{1 if richting == '+' else 0}",
                True,
            )
            for cilinder, richting in stappen[stapindex]
        ]

    def contacten_waar_op_rij(contacten, rij):
        for naam, positief in contacten:
            if naam == "st" or naam.startswith("g"):
                raise ValueError(
                    "Interne fout: hier worden alleen "
                    "cilindersensoren verwacht."
                )

            cilinder = naam[0].upper()
            stand = int(naam[1:])

            waarde = int(rij[cilinder]) == stand

            if waarde != positief:
                return False

        return True

    def maak_groepen(knippen):
        grenzen = sorted(knippen)
        groepen = []

        for positie, begin in enumerate(grenzen):
            einde = grenzen[(positie + 1) % len(grenzen)]

            groep = []
            gezien = set()
            stapindex = begin

            while True:
                if gezien & stapcilinders[stapindex]:
                    return None

                gezien.update(stapcilinders[stapindex])
                groep.append(stapindex)

                stapindex = (stapindex + 1) % n

                if stapindex == einde:
                    break

            groepen.append(groep)

        # De fase die invoerstap 1 bevat, wordt fase 0.
        nulpositie = next(
            index
            for index, groep in enumerate(groepen)
            if 0 in groep
        )

        return groepen[nulpositie:] + groepen[:nulpositie]

    def beoordeel(groepen):
        aantal_fasen = len(groepen)
        aantal_geheugens = aantal_fasen - 1

        fase_per_stap = {}

        for fase, groep in enumerate(groepen):
            for stapindex in groep:
                fase_per_stap[stapindex] = fase

        def fasecontacten(fase):
            if fase == 0:
                return [
                    (f"g{nummer}", False)
                    for nummer in range(1, aantal_fasen)
                ]

            return [(f"g{fase}", True)]

        def overgangsvoorwaarde(van_fase, naar_fase):
            laatste_stap = groepen[van_fase][-1]
            sensoren = eindcontacten(laatste_stap)

            # Een overgangssignaal mag zonder fasecontact
            # worden gebruikt wanneer het op de berekende
            # grenzen alleen bij bron- of doelfase voorkomt.
            passende_fasen = set()

            for grens in range(n):
                if contacten_waar_op_rij(
                    sensoren,
                    basisdata.iloc[grens],
                ):
                    passende_fasen.add(
                        fase_per_stap[grens]
                    )

            if passende_fasen <= {van_fase, naar_fase}:
                return sensoren

            return uniek(
                fasecontacten(van_fase) + sensoren
            )

        bewegingstermen = {
            (cilinder, richting): []
            for cilinder in cilinders
            for richting in ("+", "-")
        }

        details = []

        for stapindex, bewegingen in enumerate(stappen):
            fase = fase_per_stap[stapindex]
            groep = groepen[fase]

            eerste_in_fase = stapindex == groep[0]
            vorige_stap = (stapindex - 1) % n

            factoren = fasecontacten(fase)

            if stapindex == 0:
                # Start alleen op het startpunt.
                factoren = (
                    [("st", True)]
                    + factoren
                    + eindcontacten(vorige_stap)
                )

            elif not eerste_in_fase:
                factoren = (
                    factoren
                    + eindcontacten(vorige_stap)
                )

            elif fase == 0:
                # Fase 0 is impliciet. De eindmelding van
                # de vorige fase blijft een extra voorwaarde.
                factoren = (
                    factoren
                    + eindcontacten(vorige_stap)
                )

            # Eerste beweging van een expliciete fase
            # wordt direct door het fasegeheugen vrijgegeven.
            factoren = uniek(factoren)

            for cilinder, richting in bewegingen:
                bewegingstermen[
                    (cilinder, richting)
                ].append(factoren)

                details.append({
                    "Beweging": cilinder + richting,
                    "Stap": stapindex + 1,
                    "Startstramienlijn": stapindex,
                    "Fase": fase,
                    "Voorwaarde": latex_term(factoren),
                })

        geheugenontwerp = []
        geheugencontacten = 0

        for fase in range(1, aantal_fasen):
            setfactoren = overgangsvoorwaarde(
                fase - 1,
                fase,
            )

            if fase < aantal_fasen - 1:
                # De volgende fase reset de voorgaande.
                resetfactoren = [
                    (f"g{fase + 1}", True)
                ]
            else:
                resetfactoren = overgangsvoorwaarde(
                    fase,
                    0,
                )

            geheugencontacten += (
                len(uniek(setfactoren))
                + len(uniek(resetfactoren))
            )

            geheugenontwerp.append({
                "Geheugen": f"G{fase}",
                "SET": (
                    rf"S_{{g_{{{fase}}}}}"
                    rf" = {latex_term(setfactoren)}"
                ),
                "RESET": (
                    rf"R_{{g_{{{fase}}}}}"
                    rf" = {latex_term(resetfactoren)}"
                ),
                "Beginstand": 0,
                "Contacten": (
                    len(uniek(setfactoren))
                    + len(uniek(resetfactoren))
                ),
            })

        cilinderformules = []
        cilindercontacten = 0

        for cilinder in cilinders:
            for richting in ("+", "-"):
                termen = bewegingstermen[
                    (cilinder, richting)
                ]

                unieke_termen = list(dict.fromkeys(
                    tuple(term)
                    for term in termen
                ))

                aantal_contacten = sum(
                    len(term)
                    for term in unieke_termen
                )

                cilindercontacten += aantal_contacten

                cilinderformules.append({
                    "Cilinder": cilinder,
                    "Beweging": cilinder + richting,
                    "Formule": (
                        rf"\mathrm{{{cilinder}}}^{{{richting}}}"
                        rf" = {latex_of(termen)}"
                    ),
                    "Contacten": aantal_contacten,
                    "Termen": len(unieke_termen),
                })

        return {
            "Groepen": groepen,
            "Fase per stap": fase_per_stap,
            "Cilinderformules": cilinderformules,
            "Geheugenformules": geheugenontwerp,
            "Details": details,
            "Contacten": (
                cilindercontacten + geheugencontacten
            ),
            "Aantal geheugens": aantal_geheugens,
        }

    beste = None

    # Eerst zo weinig mogelijk fasen zoeken.
    for aantal_fasen in range(1, n + 1):
        for knippen in combinations(range(n), aantal_fasen):
            groepen = maak_groepen(knippen)

            if groepen is None:
                continue

            ontwerp = beoordeel(groepen)

            if (
                beste is None
                or ontwerp["Contacten"] < beste["Contacten"]
            ):
                beste = ontwerp

        if beste is not None:
            break

    if beste is None:
        raise ValueError(
            "Er kon geen fase-indeling worden gevonden."
        )

    data = basisdata.copy()
    geheugens = [
        f"G{nummer}"
        for nummer in range(
            1,
            beste["Aantal geheugens"] + 1,
        )
    ]

    fase_per_stap = beste["Fase per stap"]

    # De fase op een grens is de fase van de volgende stap.
    fases = [
        fase_per_stap[grens]
        for grens in range(n)
    ] + [fase_per_stap[0]]

    data["Fase"] = fases
    data["Cilinderbitgetal"] = data["Bitgetal"].copy()
    data["Geheugenbitgetal"] = 0

    gewichten = {
        cilinder: bitwaarde(cilinder)
        for cilinder in cilinders
    }

    eerste_geheugenbit = max(
        ord(cilinder) - ord("A")
        for cilinder in cilinders
    ) + 1

    for index, geheugen in enumerate(geheugens):
        fase = index + 1

        data[geheugen] = [
            int(actieve_fase == fase)
            for actieve_fase in fases
        ]

        gewicht = 1 << (eerste_geheugenbit + index)
        gewichten[geheugen] = gewicht

        data["Geheugenbitgetal"] += (
            data[geheugen] * gewicht
        )

    data["Bitgetal"] = (
        data["Cilinderbitgetal"]
        + data["Geheugenbitgetal"]
    )

    schakelingen = []

    for grens in range(1, len(data)):
        for geheugen in geheugens:
            oud = int(data.iloc[grens - 1][geheugen])
            nieuw = int(data.iloc[grens][geheugen])

            if oud != nieuw:
                schakelingen.append({
                    "Stramienlijn": grens,
                    "Geheugen": geheugen,
                    "Actie": "SET" if nieuw else "RESET",
                    "Nieuwe stand": nieuw,
                })

    schakelingen = pd.DataFrame(
        schakelingen,
        columns=[
            "Stramienlijn",
            "Geheugen",
            "Actie",
            "Nieuwe stand",
        ],
    )

    fasetabel = pd.DataFrame([
        {
            "Fase": fase,
            "Geheugen": (
                "Alle geheugens uit"
                if fase == 0
                else f"G{fase}"
            ),
            "Invoerstappen": ", ".join(
                str(stapindex + 1)
                for stapindex in groep
            ),
            "Bewegingen": " → ".join(
                staptekst(stappen[stapindex])
                for stapindex in groep
            ),
        }
        for fase, groep in enumerate(beste["Groepen"])
    ])

    for detail in beste["Details"]:
        rij = data.iloc[detail["Startstramienlijn"]]

        detail["Cilinderbitgetal"] = int(
            rij["Cilinderbitgetal"]
        )

        detail["Totaal bitgetal"] = int(
            rij["Bitgetal"]
        )

    return {
        "Data": data,
        "Geheugens": geheugens,
        "Gewichten": gewichten,
        "Schakelingen": schakelingen,
        "Fasetabel": fasetabel,
        "Cilinderformules": beste["Cilinderformules"],
        "Geheugenformules": beste["Geheugenformules"],
        "Details": beste["Details"],
        "Contacten": beste["Contacten"],
    }


# ============================================================
# GEHEUGENINDELING
# ============================================================

def voeg_geheugens_toe(data, cilinders):
    """
    Minimaliseer het aantal geheugenbits onder de regel:
    alleen schakelen op unieke cilinderstanden.
    """
    data = data.copy()

    standen = [
        tuple(int(rij[c]) for c in cilinders)
        for _, rij in data.iterrows()
    ]

    cyclisch = standen[0] == standen[-1]
    werkstanden = standen[:-1] if cyclisch else standen

    aantal = len(werkstanden)
    voorkomens = {}

    for grens, stand in enumerate(werkstanden):
        voorkomens.setdefault(stand, []).append(grens)

    hoogste_aantal = max(
        len(grenzen)
        for grenzen in voorkomens.values()
    )

    toegestane_schakellijnen = {
        grenzen[0]
        for grenzen in voorkomens.values()
        if len(grenzen) == 1
    }

    ouders = list(range(aantal))

    def vind(index):
        while ouders[index] != index:
            ouders[index] = ouders[ouders[index]]
            index = ouders[index]
        return index

    def verenig(a, b):
        wortel_a = vind(a)
        wortel_b = vind(b)

        if wortel_a != wortel_b:
            ouders[wortel_b] = wortel_a

    for grens in range(aantal):
        if grens == 0 and not cyclisch:
            continue

        if grens not in toegestane_schakellijnen:
            verenig((grens - 1) % aantal, grens)

    wortels = sorted({
        vind(grens)
        for grens in range(aantal)
    })

    componentnummer = {
        wortel: nummer
        for nummer, wortel in enumerate(wortels)
    }

    componenten = [
        componentnummer[vind(grens)]
        for grens in range(aantal)
    ]

    buren = [set() for _ in wortels]

    for stand, grenzen in voorkomens.items():
        betrokken = [
            componenten[grens]
            for grens in grenzen
        ]

        if len(betrokken) != len(set(betrokken)):
            getal = sum(
                waarde * bitwaarde(cilinder)
                for cilinder, waarde in zip(cilinders, stand)
            )

            raise ValueError(
                "Geen oplossing binnen de ontwerpregel "
                "'alleen schakelen bij unieke cilinderstanden'. "
                f"Cilinderbitgetal {getal} komt meerdere keren "
                "voor binnen hetzelfde geheugengebied. "
                "Hiervoor is een uitgebreidere "
                "geheugenontwerpmethode nodig."
            )

        for index, a in enumerate(betrokken):
            for b in betrokken[index + 1:]:
                buren[a].add(b)
                buren[b].add(a)

    def zoek_codering(max_codes):
        kleuren = [-1] * len(wortels)
        kleuren[componenten[0]] = 0

        def zoek():
            open_knopen = [
                knoop
                for knoop, kleur in enumerate(kleuren)
                if kleur == -1
            ]

            if not open_knopen:
                return kleuren.copy()

            knoop = max(
                open_knopen,
                key=lambda kandidaat: (
                    len({
                        kleuren[buur]
                        for buur in buren[kandidaat]
                        if kleuren[buur] != -1
                    }),
                    len(buren[kandidaat]),
                ),
            )

            verboden = {
                kleuren[buur]
                for buur in buren[knoop]
                if kleuren[buur] != -1
            }

            bovengrens = min(
                max_codes,
                max(kleuren) + 2,
            )

            for kleur in range(bovengrens):
                if kleur in verboden:
                    continue

                kleuren[knoop] = kleur
                resultaat = zoek()

                if resultaat is not None:
                    return resultaat

                kleuren[knoop] = -1

            return None

        return zoek()

    minimum_bits = (hoogste_aantal - 1).bit_length()
    maximum_bits = (len(wortels) - 1).bit_length()

    componentcodes = None

    for aantal_geheugens in range(
        minimum_bits,
        maximum_bits + 1,
    ):
        componentcodes = zoek_codering(
            1 << aantal_geheugens
        )

        if componentcodes is not None:
            break

    if componentcodes is None:
        raise ValueError(
            "Er kon geen geldige geheugenindeling worden gevonden."
        )

    codes = [
        componentcodes[component]
        for component in componenten
    ]

    if cyclisch:
        codes.append(codes[0])

    geheugens = [
        f"G{index + 1}"
        for index in range(aantal_geheugens)
    ]

    gewichten = {
        cilinder: bitwaarde(cilinder)
        for cilinder in cilinders
    }

    eerste_geheugenbit = max(
        ord(cilinder) - ord("A")
        for cilinder in cilinders
    ) + 1

    data["Cilinderbitgetal"] = data["Bitgetal"].copy()
    data["Geheugenbitgetal"] = 0

    for index, geheugen in enumerate(geheugens):
        data[geheugen] = [
            (code >> index) & 1
            for code in codes
        ]

        gewicht = 1 << (eerste_geheugenbit + index)
        gewichten[geheugen] = gewicht

        data["Geheugenbitgetal"] += (
            data[geheugen] * gewicht
        )

    data["Bitgetal"] = (
        data["Cilinderbitgetal"]
        + data["Geheugenbitgetal"]
    )

    controle = data.iloc[:-1] if cyclisch else data

    if not controle["Bitgetal"].is_unique:
        raise ValueError(
            "De geheugenindeling bevat dubbele totaaltoestanden."
        )

    schakelingen = []

    for grens in range(1, len(data)):
        for geheugen in geheugens:
            oud = int(data.iloc[grens - 1][geheugen])
            nieuw = int(data.iloc[grens][geheugen])

            if oud == nieuw:
                continue

            voorwaarde = " AND ".join(
                f"{c.lower()}{int(data.iloc[grens][c])}"
                for c in cilinders
            )

            schakelingen.append({
                "Stramienlijn": int(
                    data.iloc[grens]["Stapgrens"]
                ),
                "Cilinderbitgetal": int(
                    data.iloc[grens]["Cilinderbitgetal"]
                ),
                "Geheugen": geheugen,
                "Actie": "SET" if nieuw else "RESET",
                "Nieuwe stand": nieuw,
                "Volledige voorwaarde": voorwaarde,
            })

    schakelingen = pd.DataFrame(
        schakelingen,
        columns=[
            "Stramienlijn",
            "Cilinderbitgetal",
            "Geheugen",
            "Actie",
            "Nieuwe stand",
            "Volledige voorwaarde",
        ],
    )

    return (
        data,
        geheugens,
        gewichten,
        schakelingen,
        cyclisch,
        hoogste_aantal,
    )


# ============================================================
# NOTATIE
# ============================================================

def sensor_latex(cilinder, stand):
    return rf"{cilinder.lower()}_{{{int(stand)}}}"


def geheugen_latex(geheugen, stand):
    symbool = rf"g_{{{geheugen[1:]}}}"

    return (
        symbool
        if int(stand)
        else rf"\overline{{{symbool}}}"
    )


def toestand_naar_latex(rij, cilinders, geheugens):
    factoren = [
        sensor_latex(c, rij[c])
        for c in cilinders
    ]

    factoren.extend(
        geheugen_latex(g, rij[g])
        for g in geheugens
    )

    return r" \cdot ".join(factoren) if factoren else "1"


def maak_signaaldefinities(cilinders, geheugens):
    definities = []

    for cilinder in cilinders:
        for stand in (0, 1):
            definities.append({
                "Signaal": cilinder,
                "Stand": stand,
                "Latex": sensor_latex(cilinder, stand),
            })

    for geheugen in geheugens:
        for stand in (0, 1):
            definities.append({
                "Signaal": geheugen,
                "Stand": stand,
                "Latex": geheugen_latex(geheugen, stand),
            })

    return definities


def rij_naar_contacten(rij, definities):
    return frozenset(
        index
        for index, definitie in enumerate(definities)
        if int(rij[definitie["Signaal"]])
        == definitie["Stand"]
    )


# ============================================================
# EXACTE CONTACTMINIMALISATIE
# ============================================================

def minimaliseer_contactformule(
    toestanden,
    aan_indices,
    uit_indices,
    definities,
):
    """
    Minimaliseer het totale aantal contactverwijzingen
    in een OF-van-EN-formule.

    Bij gelijke contactkosten: minimaliseer het aantal termen.

    De controle geldt uitsluitend voor de opgegeven toestanden.
    """
    aan = sorted(set(aan_indices))
    uit = sorted(set(uit_indices))

    if set(aan) & set(uit):
        raise ValueError(
            "Een toestand moet zowel AAN als UIT zijn."
        )

    if not aan:
        return {
            "Latex": "0",
            "Contacten": 0,
            "Termen": 0,
        }

    positief = [
        toestanden[index]
        for index in aan
    ]

    negatief = [
        toestanden[index]
        for index in uit
    ]

    kandidaten_per_dekking = {}
    getest = set()
    aantal_tests = 0

    for positieve_toestand in positief:
        contacten = sorted(positieve_toestand)

        if len(contacten) > MAX_FACTOR_PER_TOESTAND:
            raise ValueError(
                "Te veel factoren voor exacte minimalisatie. "
                f"De limiet is {MAX_FACTOR_PER_TOESTAND} "
                "factoren per toestand."
            )

        geldige_subtermen = []

        for lengte in range(len(contacten) + 1):
            for combinatie in combinations(contacten, lengte):
                term = frozenset(combinatie)

                if any(
                    subterm.issubset(term)
                    for subterm in geldige_subtermen
                ):
                    continue

                if term in getest:
                    continue

                getest.add(term)
                aantal_tests += 1

                if aantal_tests > MAX_KANDIDAAT_TESTS:
                    raise ValueError(
                        "De contactminimalisatie bereikt "
                        "de kandidaatzoeklimiet. "
                        "Er wordt geen onbewezen minimum getoond."
                    )

                if any(
                    term.issubset(toestand)
                    for toestand in negatief
                ):
                    continue

                geldige_subtermen.append(term)

                dekking = 0

                for positie, toestand in enumerate(positief):
                    if term.issubset(toestand):
                        dekking |= 1 << positie

                bestaande = kandidaten_per_dekking.get(dekking)

                if (
                    bestaande is None
                    or len(term) < len(bestaande)
                    or (
                        len(term) == len(bestaande)
                        and tuple(sorted(term))
                        < tuple(sorted(bestaande))
                    )
                ):
                    kandidaten_per_dekking[dekking] = term

    kandidaten = sorted(
        kandidaten_per_dekking.items(),
        key=lambda item: (
            len(item[1]),
            -item[0].bit_count(),
            tuple(sorted(item[1])),
        ),
    )

    opties = [[] for _ in positief]

    for kandidaat_index, (dekking, _) in enumerate(kandidaten):
        for positie in range(len(positief)):
            if dekking & (1 << positie):
                opties[positie].append(kandidaat_index)

    if any(not mogelijkheden for mogelijkheden in opties):
        raise ValueError(
            "Geen onderscheidende formule mogelijk met "
            "de beschikbare signalen en geheugenstanden."
        )

    volledig = (1 << len(positief)) - 1
    aantal_dekkingstests = 0

    @lru_cache(maxsize=None)
    def zoek(gedekt):
        nonlocal aantal_dekkingstests

        aantal_dekkingstests += 1

        if aantal_dekkingstests > MAX_DEKKINGSTOESTANDEN:
            raise ValueError(
                "De contactminimalisatie bereikt "
                "de dekkingszoeklimiet. "
                "Er wordt geen onbewezen minimum getoond."
            )

        if gedekt == volledig:
            return 0, 0, ()

        open_posities = [
            positie
            for positie in range(len(positief))
            if not (gedekt & (1 << positie))
        ]

        positie = min(
            open_posities,
            key=lambda p: len(opties[p]),
        )

        beste = None

        for kandidaat_index in opties[positie]:
            dekking, term = kandidaten[kandidaat_index]
            vervolg = zoek(gedekt | dekking)

            resultaat = (
                len(term) + vervolg[0],
                1 + vervolg[1],
                (kandidaat_index,) + vervolg[2],
            )

            if beste is None or resultaat[:2] < beste[:2]:
                beste = resultaat

        return beste

    contacten, aantal_termen, keuze = zoek(0)

    gekozen_termen = [
        kandidaten[index][1]
        for index in keuze
    ]

    # Controleer de gekozen formule opnieuw.
    def evalueer(toestand):
        return any(
            term.issubset(toestand)
            for term in gekozen_termen
        )

    if (
        not all(evalueer(t) for t in positief)
        or any(evalueer(t) for t in negatief)
    ):
        raise ValueError(
            "Interne controle van de minimale formule mislukt."
        )

    latex_termen = []

    for term in gekozen_termen:
        factoren = [
            definities[index]["Latex"]
            for index in sorted(term)
        ]

        latex_termen.append(
            r" \cdot ".join(factoren)
            if factoren
            else "1"
        )

    return {
        "Latex": " + ".join(latex_termen),
        "Contacten": contacten,
        "Termen": aantal_termen,
    }


# ============================================================
# CILINDER- EN GEHEUGENFORMULES
# ============================================================

def maak_cilinderformules(data, stappen, cilinders, geheugens):
    cyclisch = is_cyclisch(data, cilinders)
    werkdata = data.iloc[:-1] if cyclisch else data

    definities = maak_signaaldefinities(
        cilinders,
        geheugens,
    )

    toestanden = [
        rij_naar_contacten(rij, definities)
        for _, rij in werkdata.iterrows()
    ]

    aan_per_beweging = {
        (c, richting): set()
        for c in cilinders
        for richting in ("+", "-")
    }

    details = []

    for stapnummer, bewegingen in enumerate(stappen, start=1):
        startgrens = stapnummer - 1
        rij = data.iloc[startgrens]

        voorwaarde = toestand_naar_latex(
            rij,
            cilinders,
            geheugens,
        )

        for cilinder, richting in bewegingen:
            aan_per_beweging[
                (cilinder, richting)
            ].add(startgrens)

            details.append({
                "Beweging": cilinder + richting,
                "Stap": stapnummer,
                "Startstramienlijn": startgrens,
                "Cilinderbitgetal": int(
                    rij["Cilinderbitgetal"]
                ),
                "Totaal bitgetal": int(rij["Bitgetal"]),
                "Voorwaarde": voorwaarde,
            })

    alle_indices = set(range(len(toestanden)))
    formules = []

    for cilinder in cilinders:
        for richting in ("+", "-"):
            aan = aan_per_beweging[(cilinder, richting)]

            minimaal = minimaliseer_contactformule(
                toestanden,
                aan,
                alle_indices - aan,
                definities,
            )

            formules.append({
                "Cilinder": cilinder,
                "Beweging": cilinder + richting,
                "Formule": (
                    rf"\mathrm{{{cilinder}}}^{{{richting}}}"
                    rf" = {minimaal['Latex']}"
                ),
                "Contacten": minimaal["Contacten"],
                "Contacten oorspronkelijk": (
                    len(aan)
                    * (len(cilinders) + len(geheugens))
                ),
                "Termen": minimaal["Termen"],
            })

    return formules, details


def maak_geheugenformules(data, cilinders, geheugens):
    definities = maak_signaaldefinities(cilinders, [])

    toestanden = [
        rij_naar_contacten(rij, definities)
        for _, rij in data.iterrows()
    ]

    alle_indices = set(range(len(toestanden)))
    formules = []

    def gelijke_toestanden_meenemen(indices):
        geselecteerd = {
            toestanden[index]
            for index in indices
        }

        return {
            index
            for index, toestand in enumerate(toestanden)
            if toestand in geselecteerd
        }

    for geheugen in geheugens:
        set_indices = set()
        reset_indices = set()

        for grens in range(1, len(data)):
            oud = int(data.iloc[grens - 1][geheugen])
            nieuw = int(data.iloc[grens][geheugen])

            if oud == nieuw:
                continue

            if nieuw:
                set_indices.add(grens)
            else:
                reset_indices.add(grens)

        set_aan = gelijke_toestanden_meenemen(set_indices)
        reset_aan = gelijke_toestanden_meenemen(reset_indices)

        set_minimaal = minimaliseer_contactformule(
            toestanden,
            set_aan,
            alle_indices - set_aan,
            definities,
        )

        reset_minimaal = minimaliseer_contactformule(
            toestanden,
            reset_aan,
            alle_indices - reset_aan,
            definities,
        )

        nummer = geheugen[1:]

        formules.append({
            "Geheugen": geheugen,
            "SET": (
                rf"S_{{g_{{{nummer}}}}}"
                rf" = {set_minimaal['Latex']}"
            ),
            "RESET": (
                rf"R_{{g_{{{nummer}}}}}"
                rf" = {reset_minimaal['Latex']}"
            ),
            "Beginstand": int(data.iloc[0][geheugen]),
            "Contacten": (
                set_minimaal["Contacten"]
                + reset_minimaal["Contacten"]
            ),
            "Contacten oorspronkelijk": (
                (len(set_indices) + len(reset_indices))
                * len(cilinders)
            ),
        })

    return formules


# ============================================================
# DIAGRAM
# ============================================================

def teken_diagram(data, stappen, cilinders, geheugens):
    aantal_stappen = len(stappen)
    signalen = cilinders + geheugens

    figuur, assen = plt.subplots(
        len(signalen),
        1,
        sharex=True,
        figsize=(
            max(9, aantal_stappen * 1.5),
            max(3.5, len(signalen) * 1.9),
        ),
        squeeze=False,
    )

    grenzen = data["Stapgrens"].to_numpy()

    for index, signaal in enumerate(signalen):
        as_ = assen[index, 0]
        geheugen = signaal in geheugens

        if geheugen:
            as_.step(
                grenzen,
                data[signaal],
                where="post",
                color="tab:purple",
                linewidth=2,
            )

            as_.plot(
                grenzen,
                data[signaal],
                linestyle="none",
                marker="o",
                markersize=4,
                color="tab:purple",
            )

            for grens in range(1, len(data)):
                oud = int(data.iloc[grens - 1][signaal])
                nieuw = int(data.iloc[grens][signaal])

                if oud == nieuw:
                    continue

                laatste = grens == aantal_stappen

                as_.annotate(
                    f"{signaal}{'+' if nieuw else '-'}",
                    xy=(grenzen[grens], 1.12),
                    xytext=(-5 if laatste else 5, 0),
                    textcoords="offset points",
                    ha="right" if laatste else "left",
                    va="center",
                    fontsize=10,
                    color="tab:purple",
                )

        else:
            as_.plot(
                grenzen,
                data[signaal],
                color="tab:blue",
                linewidth=2,
                marker="o",
                markersize=4,
            )

            for nummer, bewegingen in enumerate(stappen, start=1):
                for cilinder, richting in bewegingen:
                    if cilinder == signaal:
                        as_.text(
                            nummer - 0.5,
                            1.12,
                            cilinder + richting,
                            ha="center",
                            va="center",
                            fontsize=10,
                        )

        as_.set_ylabel(
            signaal,
            rotation=0,
            fontsize=14,
            labelpad=20,
        )

        as_.set_ylim(-0.2, 1.35)
        as_.set_yticks([0, 1])

        as_.set_yticklabels(
            ["0 — uit", "1 — aan"]
            if geheugen
            else ["0 — in", "1 — uit"]
        )

        as_.set_xticks(grenzen)
        as_.grid(True, which="major", alpha=0.35)

        as_.tick_params(
            axis="x",
            which="major",
            labelbottom=False,
        )

        for nummer, bewegingen in enumerate(stappen, start=1):
            if len(bewegingen) > 1:
                as_.axvspan(
                    nummer - 1,
                    nummer,
                    color="tab:orange",
                    alpha=0.10,
                )

    onderste_as = assen[-1, 0]
    onderste_as.set_xlim(0, aantal_stappen)

    onderste_as.set_xticks(
        [
            nummer - 0.5
            for nummer in range(1, aantal_stappen + 1)
        ],
        minor=True,
    )

    onderste_as.set_xticklabels(
        [
            str(nummer)
            for nummer in range(1, aantal_stappen + 1)
        ],
        minor=True,
    )

    onderste_as.tick_params(
        axis="x",
        which="minor",
        labelbottom=True,
        length=0,
        pad=8,
    )

    onderste_as.set_xlabel(
        "Bewegingsvolgorde",
        labelpad=12,
    )

    bovenste_as = assen[0, 0].secondary_xaxis("top")
    bovenste_as.set_xticks(grenzen)

    bovenste_as.set_xticklabels([
        f"{int(cilindergetal)} ({int(totaalgetal)})"
        for cilindergetal, totaalgetal in zip(
            data["Cilinderbitgetal"],
            data["Bitgetal"],
        )
    ])

    bovenste_as.set_xlabel(
        "Cilinderbitgetal (inclusief geheugens)",
        labelpad=10,
    )

    bovenste_as.tick_params(
        axis="x",
        labelsize=11,
        pad=6,
    )

    figuur.suptitle("Bewegingsdiagram — zonder tijdschaal")
    figuur.tight_layout()

    return figuur

def maak_ventielaansturing(
    cilinderformules,
    cilinders,
    beginstanden,
    ventielinstellingen,
):
    voorwaarden = {
        (formule["Cilinder"], formule["Beweging"][-1]):
            formule["Formule"].split("=", 1)[1].strip()
        for formule in cilinderformules
    }

    resultaten = []

    for cilinder in cilinders:
        instelling = ventielinstellingen[cilinder]

        plus = voorwaarden[(cilinder, "+")]
        min_ = voorwaarden[(cilinder, "-")]

        if instelling["Type"] == "Bistabiel":
            resultaten.append({
                "Cilinder": cilinder,
                "Type": "Bistabiel",
                "Formules": [
                    rf"Y_{{{cilinder},+}} = {plus}",
                    rf"Y_{{{cilinder},-}} = {min_}",
                ],
                "Beginwaarde": None,
                "Toelichting": (
                    "Twee richtingsuitgangen. Controleer in de "
                    "uiteindelijke besturing dat beide spoelen "
                    "niet gelijktijdig worden aangestuurd."
                ),
            })

            continue

        ruststand = (
            0
            if instelling["Ruststand"] == "Ingeschoven"
            else 1
        )

        spoelvoorwaarde = plus if ruststand == 0 else min_

        resultaten.append({
            "Cilinder": cilinder,
            "Type": "Monostabiel",
            "Formules": [
                rf"Y_{{{cilinder}}} = {spoelvoorwaarde}",
            ],
            "Beginwaarde": None,
            "Toelichting": (
                f"Ruststand: {instelling['Ruststand'].lower()}. "
                "Eén directe spoeluitgang, zonder extra "
                "commandogeheugen. De formule moet waar blijven "
                "zolang de bekrachtigde ventielstand nodig is."
            ),
        })

    return resultaten

# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Bewegingsdiagram",
    layout="wide",
)

st.title("Bewegingsdiagram met automatische fasebesturing")

with st.form("bewegingsvolgorde_formulier"):
    tekst = st.text_input(
        "Bewegingsvolgorde",
        value=VOORBEELD,
        placeholder="Bijvoorbeeld: A-, A+, B+, C+, (C-, B-)",
    )

    st.caption(
        "+ = uitschuiven; - = inschuiven. "
        "Bewegingen tussen haakjes gebeuren in dezelfde stap. "
        "De volgorde moet een gesloten cyclus vormen."
    )

    ventielinstellingen = {}

    try:
        invoerstappen = lees_volgorde(tekst)

        invoercilinders = sorted({
            cilinder
            for bewegingen in invoerstappen
            for cilinder, _ in bewegingen
        })

    except ValueError:
        invoercilinders = []

    if invoercilinders:
        st.subheader("Ventielinstellingen")

        for cilinder in invoercilinders:
            kolom1, kolom2, kolom3 = st.columns([1, 2, 2])

            kolom1.write(f"Cilinder {cilinder}")

            ventieltype = kolom2.selectbox(
                "Ventieltype",
                options=["Bistabiel", "Monostabiel"],
                key=f"ventieltype_{cilinder}",
            )

            if ventieltype == "Monostabiel":
                ruststand = kolom3.selectbox(
                    "Cilinderstand bij onbekrachtigd ventiel",
                    options=["Ingeschoven", "Uitgeschoven"],
                    key=f"ruststand_{cilinder}",
                )
            else:
                ruststand = None
                kolom3.caption("Twee afzonderlijke richtingssignalen")

            ventielinstellingen[cilinder] = {
                "Type": ventieltype,
                "Ruststand": ruststand,
            }

    tekenen = st.form_submit_button(
        "Bereken fasebesturing",
        type="primary",
    )


if tekenen:
    try:
        with st.spinner("Fasen, geheugens en formules berekenen..."):
            stappen = lees_volgorde(tekst)

            basisdata, beginstanden, cilinders = (
                maak_diagramdata(stappen)
            )

            ontwerp = maak_fasebesturing(
                basisdata,
                stappen,
                cilinders,
            )

            data = ontwerp["Data"]
            geheugens = ontwerp["Geheugens"]
            gewichten = ontwerp["Gewichten"]
            schakelingen = ontwerp["Schakelingen"]
            cilinderformules = ontwerp["Cilinderformules"]
            geheugenformules = ontwerp["Geheugenformules"]
            formule_details = ontwerp["Details"]
            fasetabel = ontwerp["Fasetabel"]
            ventielaansturing = maak_ventielaansturing(
                cilinderformules=cilinderformules,
                cilinders=cilinders,
                beginstanden=beginstanden,
                ventielinstellingen=ventielinstellingen,
            )

    except ValueError as fout:
        st.error(str(fout))
        st.stop()

    except RecursionError:
        st.error(
            "Deze volgorde is te ingewikkeld voor "
            "de huidige zoekmethode."
        )
        st.stop()

    # --------------------------------------------------------
    # DIAGRAM
    # --------------------------------------------------------

    st.subheader("Bewegingsdiagram")

    figuur = teken_diagram(
        data,
        stappen,
        cilinders,
        geheugens,
    )

    st.pyplot(figuur)
    plt.close(figuur)

    st.caption(
        "Blauw: cilinderbewegingen. "
        "Paars: fasegeheugens. "
        "Oranje: gelijktijdige bewegingen. "
        "Bovenaan staat het cilinderbitgetal met tussen haakjes "
        "het totale bitgetal inclusief geheugens. "
        "De horizontale as geeft stappen weer, geen tijd."
    )

    # --------------------------------------------------------
    # SAMENVATTING
    # --------------------------------------------------------

    kolom1, kolom2, kolom3, kolom4 = st.columns(4)

    kolom1.metric(
        "Aantal stappen",
        len(stappen),
    )

    kolom2.metric(
        "Aantal fasen",
        len(fasetabel),
    )

    kolom3.metric(
        "Aantal geheugens",
        len(geheugens),
    )

    kolom4.metric(
        "Contactverwijzingen",
        ontwerp["Contacten"],
    )

    st.success(
        "De cilinders en de bedoelde fasegeheugenstanden "
        "eindigen in hun beginstand: de volgorde is cyclisch."
    )

    # --------------------------------------------------------
    # FASE-INDELING
    # --------------------------------------------------------

    st.subheader("Automatische fase-indeling")

    st.dataframe(
        fasetabel,
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "De zoeker kiest eerst het kleinste aantal fasen "
        "waarbij een cilinder maximaal één keer per fase "
        "voorkomt. Bij evenveel fasen kiest hij de indeling "
        "met de minste contactverwijzingen binnen de "
        "gegenereerde formulevorm. "
        "Fase 0 betekent dat alle fasegeheugens uit staan. "
        "De overige fasen gebruiken ieder één geheugen."
    )

    st.caption(
        "Bij een gesloten cyclus kan een fase over het einde "
        "van de ingevoerde volgorde heen lopen. "
        "Een fase met bijvoorbeeld invoerstappen 8 en 1 "
        "bevat de laatste beweging en de eerste beweging "
        "van de volgende cyclus."
    )

    # --------------------------------------------------------
    # NOTATIE
    # --------------------------------------------------------

    st.subheader("Notatie")

    st.latex(
        r"a_0:\ \text{A ingeschoven}"
        r"\qquad"
        r"a_1:\ \text{A uitgeschoven}"
    )

    st.latex(
        r"g_1:\ \text{G1 aan}"
        r"\qquad"
        r"\overline{g_1}:\ \text{G1 uit}"
    )

    st.latex(
        r"\mathrm{st}:\ \text{startsignaal}"
        r"\qquad"
        r"\cdot:\ \text{EN}"
        r"\qquad"
        r"+:\ \text{OF tussen voorwaarden}"
    )

    st.caption(
        "a₀ en a₁ zijn afzonderlijke eindstandsignalen. "
        "Een bovenstreep geeft een ontkend geheugensignaal aan."
    )

    st.subheader("Ventielaansturing")

    st.caption(
        "De bewegingsvoorwaarden beschrijven wanneer een beweging "
        "wordt aangevraagd. Hieronder staat de vertaling naar "
        "de spoeluitgangen van het gekozen ventiel."
    )

    for ventiel in ventielaansturing:
        st.markdown(
            f"### Cilinder {ventiel['Cilinder']} "
            f"— {ventiel['Type'].lower()}"
        )

        for formule in ventiel["Formules"]:
            st.latex(formule)

        if ventiel["Beginwaarde"] is not None:
            st.caption(
                "Vereiste commandogeheugenstand voor de "
                f"getekende beginsituatie: {ventiel['Beginwaarde']}."
            )

        st.caption(ventiel["Toelichting"])

    # --------------------------------------------------------
    # GEHEUGENFORMULES
    # --------------------------------------------------------

    if geheugens:
        st.subheader("SET/RESET-formules van de fasegeheugens")

        st.caption(
            "SET activeert een fasegeheugen. "
            "Een volgende fase reset het voorgaande geheugen. "
            "De laatste fase keert terug naar fase 0. "
            "De bedoelde stabiele toestand heeft maximaal "
            "één actief fasegeheugen."
        )

        for formule in geheugenformules:
            st.markdown(
                f"### Geheugen {formule['Geheugen']}"
            )

            st.latex(formule["SET"])
            st.latex(formule["RESET"])

            st.caption(
                f"Beginstand: {formule['Beginstand']}. "
                f"SET en RESET samen: "
                f"{formule['Contacten']} contactverwijzingen."
            )

        st.subheader("Geheugenschakelingen in het diagram")

        st.dataframe(
            schakelingen,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Deze tabel toont de bedoelde faseovergangen. "
            "Hij is geen simulatie van de verwerkingstijd "
            "of de onderlinge volgorde van SET en RESET."
        )

    # --------------------------------------------------------
    # STARTGEDRAG
    # --------------------------------------------------------

    st.subheader("Startgedrag")

    st.caption(
        "st staat uitsluitend in de formule voor de eerste "
        "beweging van de ingevoerde volgorde. "
        "Als st actief blijft, kan de cyclus opnieuw starten "
        "wanneer de beginvoorwaarden terugkomen. "
        "Voor precies één cyclus per druk op een knop "
        "is aanvullende startpulslogica nodig."
    )

    # --------------------------------------------------------
    # CONTACTENTABEL
    # --------------------------------------------------------

    st.subheader("Contactverwijzingen per onderdeel")

    contactentabel = [
        {
            "Onderdeel": formule["Beweging"],
            "Contactverwijzingen": formule["Contacten"],
        }
        for formule in cilinderformules
    ]

    contactentabel.extend(
        {
            "Onderdeel": (
                f"{formule['Geheugen']} SET + RESET"
            ),
            "Contactverwijzingen": formule["Contacten"],
        }
        for formule in geheugenformules
    )

    st.dataframe(
        pd.DataFrame(contactentabel),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Een contactverwijzing is één signaal in een formule. "
        "Het aantal contactverwijzingen is niet hetzelfde "
        "als het aantal unieke sensoren, PLC-ingangen "
        "of fysieke draden."
    )

    # --------------------------------------------------------
    # STANDENTABEL
    # --------------------------------------------------------

    st.subheader("Standen per stramienlijn")

    st.dataframe(
        data[
            ["Stapgrens", "Fase"]
            + cilinders
            + geheugens
            + [
                "Cilinderbitgetal",
                "Geheugenbitgetal",
                "Bitgetal",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # BITGEWICHTEN
    # --------------------------------------------------------

    st.subheader("Bitgewichten")

    st.dataframe(
        pd.DataFrame([
            {
                "Signaal": signaal,
                "Type": (
                    "Geheugen"
                    if signaal in geheugens
                    else "Cilinder"
                ),
                "Bitgewicht": gewicht,
            }
            for signaal, gewicht in gewichten.items()
        ]),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # BEWEGINGSVOLGORDE
    # --------------------------------------------------------

    st.subheader("Bewegingsvolgorde per stap")

    st.dataframe(
        pd.DataFrame([
            {
                "Stap": nummer,
                "Beweging": staptekst(bewegingen),
                "Gelijktijdig": (
                    "Ja"
                    if len(bewegingen) > 1
                    else "Nee"
                ),
            }
            for nummer, bewegingen in enumerate(
                stappen,
                start=1,
            )
        ]),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # TOEPASSINGSGRENZEN
    # --------------------------------------------------------

    st.warning(
        "Dit is een faseontwerp voor bistabiele "
        "bewegingsaansturing, geen gevalideerde fysieke "
        "schakeling. Het ontwerp veronderstelt dat "
        "geheugenovergangen eerst tot rust komen voordat "
        "uitgangen worden vrijgegeven. "
        "Voor toepassing op dubbelspoelventielen moeten "
        "gelijktijdige spoelsignalen, sensorovergangen "
        "en SET/RESET-prioriteit aanvullend worden gecontroleerd. "
        "Start/stop- en veiligheidslogica zijn niet ontworpen."
    )

    st.caption(
        "De optimalisatie geldt voor de gebruikte fase-indeling "
        "en formulevorm. Dit is geen bewijs van minimale "
        "fysieke bedrading voor alle mogelijke schakelingen. "
        "Het diagram toont de bedoelde standen, niet een "
        "dynamische simulatie van de gegenereerde formules."
    )