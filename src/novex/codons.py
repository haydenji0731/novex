"""The standard genetic code & translates codons.
"""

from collections.abc import Iterator

# canonical translation table (does not apply to chrM)
CODON_TABLE: dict[str, str] = {
    "AAA": "K",
    "AAC": "N",
    "AAG": "K",
    "AAT": "N",
    "ACA": "T",
    "ACC": "T",
    "ACG": "T",
    "ACT": "T",
    "AGA": "R",
    "AGC": "S",
    "AGG": "R",
    "AGT": "S",
    "ATA": "I",
    "ATC": "I",
    "ATG": "M",
    "ATT": "I",
    "CAA": "Q",
    "CAC": "H",
    "CAG": "Q",
    "CAT": "H",
    "CCA": "P",
    "CCC": "P",
    "CCG": "P",
    "CCT": "P",
    "CGA": "R",
    "CGC": "R",
    "CGG": "R",
    "CGT": "R",
    "CTA": "L",
    "CTC": "L",
    "CTG": "L",
    "CTT": "L",
    "GAA": "E",
    "GAC": "D",
    "GAG": "E",
    "GAT": "D",
    "GCA": "A",
    "GCC": "A",
    "GCG": "A",
    "GCT": "A",
    "GGA": "G",
    "GGC": "G",
    "GGG": "G",
    "GGT": "G",
    "GTA": "V",
    "GTC": "V",
    "GTG": "V",
    "GTT": "V",
    "TAA": "*",
    "TAC": "Y",
    "TAG": "*",
    "TAT": "Y",
    "TCA": "S",
    "TCC": "S",
    "TCG": "S",
    "TCT": "S",
    "TGA": "*",
    "TGC": "C",
    "TGG": "W",
    "TGT": "C",
    "TTA": "L",
    "TTC": "F",
    "TTG": "L",
    "TTT": "F"
}

START_CODON = "ATG"
STOP_CODONS = frozenset(codon for codon, aa in CODON_TABLE.items() if aa == "*")

# ambiguous codon translated to:
UNKNOWN_AA = "X"


def iter_codons(seq: str) -> Iterator[str]:
    """Yield successive 3-base codons, ignores trailing 1-2 bases if that happens.
    """
    for i in range(0, len(seq) - len(seq) % 3, 3):
        yield seq[i:i + 3]


def translate(seq: str) -> str:
    """Translate `seq` into amino acids, one letter per codon.
    """
    return "".join(CODON_TABLE.get(codon, UNKNOWN_AA) for codon in iter_codons(seq))
