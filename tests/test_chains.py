import pytest
from pydantic import ValidationError

from novex.chains import head, Chain, GInterval, Strand, splice

TWO = Chain([(10, 19), (30, 39)])
THREE = Chain([(10, 19), (30, 39), (50, 59)])


# --- Chain -------------------------------------------------------------------

def test_chain_coerces_pairs_to_gintervals():
    c = Chain([[10, 19], (30, 39)])
    assert c.intervals == (GInterval(10, 19), GInterval(30, 39))
    assert isinstance(c.intervals, tuple)
    assert all(isinstance(x, GInterval) for x in c.intervals)


def test_chain_len_is_total_bases():
    assert len(TWO) == 20
    assert len(Chain([(5, 5)])) == 1


@pytest.mark.parametrize(
    "pos, expected",
    [(9, False), (10, True), (19, True), (20, False), (29, False), (30, True), (39, True), (40, False)],
)
def test_chain_contains(pos, expected):
    assert (pos in TWO) is expected


@pytest.mark.parametrize(
    "intervals",
    [
        [],                      # empty
        [(19, 10)],              # start > end
        [(30, 39), (10, 19)],    # unsorted
        [(10, 20), (20, 30)],    # overlapping by one base
    ],
)
def test_chain_rejects_invalid(intervals):
    with pytest.raises(ValidationError):
        Chain(intervals)


def test_chain_allows_abutting_intervals():
    assert len(Chain([(10, 19), (20, 29)])) == 20


def test_chain_is_frozen():
    with pytest.raises(ValidationError):
        TWO.intervals = ()


# --- clip_end / clip_start -----------------------------------------------------

@pytest.mark.parametrize(
    "pos, expected",
    [
        (34, [(10, 19), (30, 34)]),             # mid-interval
        (19, [(10, 19)]),                       # last base of an interval keeps it whole
        (30, [(10, 19), (30, 30)]),             # first base of an interval -> single base
        (59, [(10, 19), (30, 39), (50, 59)]),   # chain end -> unchanged
        (10, [(10, 10)]),                       # chain start -> single base
    ],
)
def test_clip_end(pos, expected):
    assert THREE.clip_end(pos) == Chain(expected)


@pytest.mark.parametrize(
    "pos, expected",
    [
        (34, [(34, 39), (50, 59)]),             # mid-interval
        (30, [(30, 39), (50, 59)]),             # first base of an interval keeps it whole
        (39, [(39, 39), (50, 59)]),             # last base of an interval -> single base
        (10, [(10, 19), (30, 39), (50, 59)]),   # chain start -> unchanged
        (59, [(59, 59)]),                       # chain end -> single base
    ],
)
def test_clip_start(pos, expected):
    assert THREE.clip_start(pos) == Chain(expected)


@pytest.mark.parametrize("clip", [Chain.clip_end, Chain.clip_start])
@pytest.mark.parametrize("pos", [5, 25, 45, 60])  # before, in gaps, after
def test_clip_rejects_pos_outside_chain(clip, pos):
    with pytest.raises(ValueError):
        clip(THREE, pos)


def test_clip_does_not_modify_input():
    before = THREE.intervals
    THREE.clip_end(34)
    THREE.clip_start(34)
    assert THREE.intervals == before


# --- splice ------------------------------------------------------------------

LEFT = Chain([(10, 19), (30, 39)])
RIGHT = Chain([(60, 69), (80, 89)])


@pytest.mark.parametrize(
    "intron, expected",
    [
        ((35, 64), [(10, 19), (30, 34), (65, 69), (80, 89)]),  # both boundaries mid-interval
        ((20, 64), [(10, 19), (65, 69), (80, 89)]),            # donor at end of a left interval
        ((35, 79), [(10, 19), (30, 34), (80, 89)]),            # acceptor at start of a right interval
        ((40, 59), [(10, 19), (30, 39), (60, 69), (80, 89)]),  # both at interval edges
    ],
)
def test_splice(intron, expected):
    assert splice(LEFT, RIGHT, GInterval(*intron)) == Chain(expected)


@pytest.mark.parametrize(
    "intron",
    [
        (26, 64),  # donor base 25 falls in left's gap
        (35, 72),  # acceptor base 73 falls in right's gap
        (5, 64),   # donor base 4 is before left
        (35, 95),  # acceptor base 96 is after right
    ],
)
def test_splice_returns_none_when_boundary_not_in_chain(intron):
    assert splice(LEFT, RIGHT, GInterval(*intron)) is None


def test_splice_minus_strand_roles():
    # On '-', the reference CDS is genomically left and the novel chain is right.
    ref_cds = Chain([(100, 120), (150, 170)])
    novel = Chain([(300, 330)])
    result = splice(ref_cds, novel, GInterval(161, 309))
    assert result == Chain([(100, 120), (150, 160), (310, 330)])


# --- genomic_to_tx / tx_to_genomic ----------------------------------------------

# (strand, genomic pos, tx offset) for TWO = (10-19, 30-39)
OFFSETS = [
    (Strand.PLUS, 10, 0),
    (Strand.PLUS, 19, 9),
    (Strand.PLUS, 30, 10),
    (Strand.PLUS, 39, 19),
    (Strand.MINUS, 39, 0),
    (Strand.MINUS, 30, 9),
    (Strand.MINUS, 19, 10),
    (Strand.MINUS, 10, 19),
]


@pytest.mark.parametrize("strand, pos, offset", OFFSETS)
def test_genomic_to_tx(strand, pos, offset):
    assert TWO.genomic_to_tx(strand, pos) == offset


@pytest.mark.parametrize("strand, pos, offset", OFFSETS)
def test_tx_to_genomic(strand, pos, offset):
    assert TWO.tx_to_genomic(strand, offset) == pos


@pytest.mark.parametrize("strand", [Strand.PLUS, Strand.MINUS])
def test_offset_roundtrip(strand):
    for offset in range(len(TWO)):
        assert TWO.genomic_to_tx(strand, TWO.tx_to_genomic(strand, offset)) == offset


@pytest.mark.parametrize("pos", [9, 25, 40])
def test_genomic_to_tx_rejects_pos_outside_chain(pos):
    with pytest.raises(ValueError):
        TWO.genomic_to_tx(Strand.PLUS, pos)


@pytest.mark.parametrize("offset", [-1, 20])
def test_tx_to_genomic_rejects_offset_out_of_range(offset):
    with pytest.raises(IndexError):
        TWO.tx_to_genomic(Strand.PLUS, offset)


def test_unknown_strand_rejected():
    with pytest.raises(ValueError):
        TWO.genomic_to_tx(Strand.UNKNOWN, 10)
    with pytest.raises(ValueError):
        TWO.tx_to_genomic(Strand.UNKNOWN, 0)


# --- head --------------------------------------------------------------------

TWO_UNEVEN = Chain([(11, 19), (40, 51)])   # 9 + 12 = 21 bases


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, [(11, 11)]),                 # first base only
        (9, [(11, 19)]),                 # exactly the first interval
        (10, [(11, 19), (40, 40)]),      # one base into the second
        (12, [(11, 19), (40, 42)]),      # mid-second-interval
        (21, [(11, 19), (40, 51)]),      # the whole chain
    ],
)
def test_head_plus(n, expected):
    assert head(TWO_UNEVEN, Strand.PLUS, n) == Chain(expected)


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, [(51, 51)]),                 # transcript starts at the highest base
        (12, [(40, 51)]),                # exactly the (genomically second) interval
        (13, [(19, 19), (40, 51)]),      # one base into the next one, leftwards
        (21, [(11, 19), (40, 51)]),      # the whole chain
    ],
)
def test_head_minus(n, expected):
    assert head(TWO_UNEVEN, Strand.MINUS, n) == Chain(expected)


@pytest.mark.parametrize("strand", [Strand.PLUS, Strand.MINUS])
def test_head_rejects_out_of_range(strand):
    with pytest.raises(IndexError):
        head(TWO_UNEVEN, strand, len(TWO_UNEVEN) + 1)
    with pytest.raises(IndexError):
        head(TWO_UNEVEN, strand, 0)


def test_head_does_not_modify_input():
    before = TWO_UNEVEN.intervals
    head(TWO_UNEVEN, Strand.PLUS, 12)
    assert TWO_UNEVEN.intervals == before
