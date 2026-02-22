"""
Formal dunk ontology used by the analyzer and contest scorer.
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class DunkOntologyEntry:
    canonical_name: str
    primary_category: str
    difficulty_points: float
    description: str


DUNK_ONTOLOGY: Dict[str, DunkOntologyEntry] = {
    # CATEGORY A — Power Finishes
    "One-Hand Power Dunk": DunkOntologyEntry(
        canonical_name="One-Hand Power Dunk",
        primary_category="CATEGORY A — Power Finishes",
        difficulty_points=0.8,
        description="Single-arm power finish with minimal rotation.",
    ),
    "Two-Hand Power Dunk": DunkOntologyEntry(
        canonical_name="Two-Hand Power Dunk",
        primary_category="CATEGORY A — Power Finishes",
        difficulty_points=1.0,
        description="Two-hand power finish with strong control.",
    ),
    "Two-Hand Reverse Power": DunkOntologyEntry(
        canonical_name="Two-Hand Reverse Power",
        primary_category="CATEGORY A — Power Finishes",
        difficulty_points=1.6,
        description="Two-hand reverse finish with roughly 180-degree body turn.",
    ),
    # CATEGORY B — Cocked and Overhead Variations
    "Tomahawk (Single Arm)": DunkOntologyEntry(
        canonical_name="Tomahawk (Single Arm)",
        primary_category="CATEGORY B — Cocked and Overhead Variations",
        difficulty_points=1.6,
        description="Ball is loaded behind the head then whipped downward.",
    ),
    "Double Tomahawk": DunkOntologyEntry(
        canonical_name="Double Tomahawk",
        primary_category="CATEGORY B — Cocked and Overhead Variations",
        difficulty_points=1.9,
        description="Two-arm overhead load followed by an explosive two-hand finish.",
    ),
    "Statue of Liberty": DunkOntologyEntry(
        canonical_name="Statue of Liberty",
        primary_category="CATEGORY B — Cocked and Overhead Variations",
        difficulty_points=1.7,
        description="Ball is held high overhead with an extended arm before the slam.",
    ),
    # CATEGORY C — Rotation Dunks
    "180 Dunk": DunkOntologyEntry(
        canonical_name="180 Dunk",
        primary_category="CATEGORY C — Rotation Dunks",
        difficulty_points=1.8,
        description="Half-spin dunk with total rotation in the 160-220 degree range.",
    ),
    "360 Dunk": DunkOntologyEntry(
        canonical_name="360 Dunk",
        primary_category="CATEGORY C — Rotation Dunks",
        difficulty_points=2.8,
        description="Full-spin dunk with total rotation in the 300-420 degree range.",
    ),
    "540 Dunk": DunkOntologyEntry(
        canonical_name="540 Dunk",
        primary_category="CATEGORY C — Rotation Dunks",
        difficulty_points=3.6,
        description="One-and-a-half-spin elite dunk with 480+ degrees of rotation.",
    ),
    # CATEGORY D — Windmill Family
    "Standard Windmill": DunkOntologyEntry(
        canonical_name="Standard Windmill",
        primary_category="CATEGORY D — Windmill Family",
        difficulty_points=2.2,
        description="Circular arm path with dip below the waist before finish.",
    ),
    "Reverse Windmill": DunkOntologyEntry(
        canonical_name="Reverse Windmill",
        primary_category="CATEGORY D — Windmill Family",
        difficulty_points=2.8,
        description="Windmill arm path with a reverse-facing finish.",
    ),
    "360 Windmill": DunkOntologyEntry(
        canonical_name="360 Windmill",
        primary_category="CATEGORY D — Windmill Family",
        difficulty_points=3.5,
        description="Full-spin windmill combination dunk.",
    ),
    # CATEGORY E — Leg Thread Dunks
    "Eastbay (Between-the-Legs)": DunkOntologyEntry(
        canonical_name="Eastbay (Between-the-Legs)",
        primary_category="CATEGORY E — Leg Thread Dunks",
        difficulty_points=3.0,
        description="Ball passes beneath the thigh before finish.",
    ),
    "Reverse Eastbay": DunkOntologyEntry(
        canonical_name="Reverse Eastbay",
        primary_category="CATEGORY E — Leg Thread Dunks",
        difficulty_points=3.4,
        description="Between-the-legs transfer with reverse-facing finish.",
    ),
    "Lob Eastbay": DunkOntologyEntry(
        canonical_name="Lob Eastbay",
        primary_category="CATEGORY H — Lob & Assisted Dunks",
        difficulty_points=3.7,
        description="Between-the-legs transfer after catching a lob.",
    ),
    "Double Eastbay": DunkOntologyEntry(
        canonical_name="Double Eastbay",
        primary_category="CATEGORY E — Leg Thread Dunks",
        difficulty_points=4.0,
        description="Two leg-thread actions in the same airborne sequence.",
    ),
    # CATEGORY F — Behind-Back
    "Behind-Back Dunk": DunkOntologyEntry(
        canonical_name="Behind-Back Dunk",
        primary_category="CATEGORY F — Behind-Back",
        difficulty_points=3.0,
        description="Ball transfers behind the back before finish.",
    ),
    "360 Behind-Back": DunkOntologyEntry(
        canonical_name="360 Behind-Back",
        primary_category="CATEGORY F — Behind-Back",
        difficulty_points=3.8,
        description="Behind-back transfer combined with a full body spin.",
    ),
    # CATEGORY G — Double Clutch
    "Double Pump": DunkOntologyEntry(
        canonical_name="Double Pump",
        primary_category="CATEGORY G — Double Clutch",
        difficulty_points=2.2,
        description="Ball is raised, lowered, then raised again before finish.",
    ),
    "Reverse Double Pump": DunkOntologyEntry(
        canonical_name="Reverse Double Pump",
        primary_category="CATEGORY G — Double Clutch",
        difficulty_points=2.8,
        description="Double pump motion with reverse-facing finish.",
    ),
    # CATEGORY H — Lob & Assisted Dunks
    "Alley-Oop Power": DunkOntologyEntry(
        canonical_name="Alley-Oop Power",
        primary_category="CATEGORY H — Lob & Assisted Dunks",
        difficulty_points=2.0,
        description="Catch-and-finish power dunk from a lob pass.",
    ),
    "Alley-Oop Reverse": DunkOntologyEntry(
        canonical_name="Alley-Oop Reverse",
        primary_category="CATEGORY H — Lob & Assisted Dunks",
        difficulty_points=2.6,
        description="Lob catch into a reverse finish.",
    ),
    "Alley-Oop 360": DunkOntologyEntry(
        canonical_name="Alley-Oop 360",
        primary_category="CATEGORY H — Lob & Assisted Dunks",
        difficulty_points=3.4,
        description="Lob catch with a full 360-degree finish.",
    ),
    "Lob Windmill": DunkOntologyEntry(
        canonical_name="Lob Windmill",
        primary_category="CATEGORY H — Lob & Assisted Dunks",
        difficulty_points=3.4,
        description="Lob catch combined with a windmill arm path.",
    ),
    "Off-Bounce Lob": DunkOntologyEntry(
        canonical_name="Off-Bounce Lob",
        primary_category="CATEGORY H — Lob & Assisted Dunks",
        difficulty_points=2.9,
        description="Self-lob off the floor bounce into a dunk finish.",
    ),
    "Off-Glass Lob": DunkOntologyEntry(
        canonical_name="Off-Glass Lob",
        primary_category="CATEGORY H — Lob & Assisted Dunks",
        difficulty_points=3.0,
        description="Self-lob off the backboard before the dunk finish.",
    ),
    # CATEGORY I — Distance & Glide
    "Free Throw Line Dunk": DunkOntologyEntry(
        canonical_name="Free Throw Line Dunk",
        primary_category="CATEGORY I — Distance & Glide",
        difficulty_points=3.2,
        description="Long-distance takeoff with extended glide before finish.",
    ),
    "Baseline Glide": DunkOntologyEntry(
        canonical_name="Baseline Glide",
        primary_category="CATEGORY I — Distance & Glide",
        difficulty_points=2.4,
        description="Long lateral glide approach into the dunk.",
    ),
    # CATEGORY J — Putback & Reaction
    "Putback Dunk": DunkOntologyEntry(
        canonical_name="Putback Dunk",
        primary_category="CATEGORY J — Putback & Reaction",
        difficulty_points=1.6,
        description="Immediate second-jump finish after a rebound action.",
    ),
    "Tip Dunk": DunkOntologyEntry(
        canonical_name="Tip Dunk",
        primary_category="CATEGORY J — Putback & Reaction",
        difficulty_points=1.4,
        description="Controlled tip-style redirect into a dunk finish.",
    ),
}


NON_DUNK_TYPES: List[str] = [
    "Layup",
    "Reverse layup",
    "Finger roll",
    "Floater",
    "Tip-in",
    "Missed dunk",
    "Blocked dunk",
]

