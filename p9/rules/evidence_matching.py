from rules.evidence_types import EVIDENCE_TYPES
from rules.burden_requirements import BURDEN_REQUIREMENTS

def evidence_satisfies_burden(evidence_types, burden):
    requirements = BURDEN_REQUIREMENTS[burden]

    for et in evidence_types:
        if (
            et in requirements["allowed_types"]
            and EVIDENCE_TYPES[et]["strength"] >= requirements["min_strength"]
        ):
            return True

    return False
