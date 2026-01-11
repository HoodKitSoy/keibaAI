# modules/policies/__init__.py

from ._bet_policy import (
    AbstractBetPolicy,
    BetPolicyCoverage25,
    BetPolicyCoverage50,
    BetPolicyCoverage75,
    BetPolicyCoverageBase,
)
from ._hybrid_bet_policy import (
    ExpectedValueBetPolicy,
    HybridBetPolicy,
    TieredCoverageBetPolicy,
)
from ._odds_based_policy import (
    MultiOddsExpectedValuePolicy,
    OddsBasedExpectedValuePolicy,
    OddsRangeFilterPolicy,
    OddsValueTieredPolicy,
)
from ._policy_selector import (
    AVAILABLE_POLICIES,
    PolicySelector,
    create_default_config,
    get_enabled_policies,
)
from ._race_selective_coverage_policy import (
    RaceSelectiveCoverage25Policy,
    RaceSelectiveCoverage50Policy,
    RaceSelectiveCoverage75Policy,
    RaceSelectiveCoveragePolicy,
)
from ._race_selective_policy import (
    RaceSelectiveExpectedValueBetPolicy,
    RaceSelectiveHybridPolicy,
    RaceSelectivePolicy,
    RaceSelectiveTieredCoverageBetPolicy,
)
from ._score_policy import (
    AbstractScorePolicy,
    RelativeProbaScorePolicy,
)
from ._win5_based_policy import (
    WIN5BasedHedgePolicy,
    WIN5BasedMultiPolicy,
)

# RLポリシー（オプション - PyTorchが必要）
try:
    from modules.rl._policy import (
        RLBetPolicy,
        RLWIN5Policy,
    )

    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
