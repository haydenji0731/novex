import pytest

from novex.codons import (
    CODON_TABLE,
    START_CODON,
    STOP_CODONS,
    UNKNOWN_AA,
    iter_codons,
    translate,
)


# --- the table ----------------------------------------------------------------

def test_table_is_complete():
    assert len(CODON_TABLE) == 64
    assert set("".join(CODON_TABLE)) == set("ACGT")


def test_stop_and_start_codons():
    assert STOP_CODONS == frozenset({"TAA", "TAG", "TGA"})
    assert CODON_TABLE[START_CODON] == "M"


def test_matches_the_standard_genetic_code():
    # built independently of CODON_TABLE, in TCAG order
    bases = "TCAG"
    aas = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
    standard = {
        a + b + c: aas[i * 16 + j * 4 + k]
        for i, a in enumerate(bases)
        for j, b in enumerate(bases)
        for k, c in enumerate(bases)
    }
    assert CODON_TABLE == standard


# --- iter_codons --------------------------------------------------------------

def test_iter_codons_splits_in_threes():
    assert list(iter_codons("ATGGCCTAA")) == ["ATG", "GCC", "TAA"]


@pytest.mark.parametrize("seq, expected", [("ATGG", ["ATG"]), ("ATGGC", ["ATG"]), ("AT", []), ("", [])])
def test_iter_codons_drops_a_trailing_partial_codon(seq, expected):
    assert list(iter_codons(seq)) == expected


# --- translate ----------------------------------------------------------------

def test_translate_basic():
    assert translate("ATGGCCTAA") == "MA*"


def test_translate_keeps_internal_stops():
    # validate.py needs to see these to detect premature stop codons
    assert translate("ATGTAAGCCTAA") == "M*A*"


def test_translate_maps_ambiguous_codons_to_x():
    assert translate("ATGNNNTAA") == f"M{UNKNOWN_AA}*"
    assert translate("ATGNCCTAA") == f"M{UNKNOWN_AA}*"


def test_translate_ignores_a_trailing_partial_codon():
    assert translate("ATGGC") == "M"


def test_translate_empty():
    assert translate("") == ""


def test_translate_every_codon_round_trips():
    seq = "".join(CODON_TABLE)
    assert translate(seq) == "".join(CODON_TABLE.values())
