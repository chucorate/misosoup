"""Find one smallest community viable across multiple media."""

import logging
import math
from numbers import Real
from pathlib import Path
from typing import Mapping, Optional, Union

import yaml
from gurobipy import Env
from reframed.solvers.solver import Parameter

from ..reframed.joint_solver import FluxBounds, JointSolver, MultimediaResult
from ..reframed.layered_community import LayeredCommunity

RawBound = Union[float, list[float], tuple[float, float]]
MediaInput = Union[str, Path, Mapping[str, Mapping[str, RawBound]]]


def _number(value: object, label: str) -> float:
    """Reject non-numeric values, booleans, and NaNs."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric.")

    result = float(value)
    if math.isnan(result):
        raise ValueError(f"{label} must not be NaN.")

    return result


def _growth_threshold(value: object, label: str) -> float:
    """Validate a strictly positive finite growth requirement."""
    threshold = _number(value, label)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"{label} must be positive and finite.")

    return threshold


def _bounds(value: object, default_upper: float, label: str) -> FluxBounds:
    """Normalize a scalar lower bound or an explicit lower/upper pair."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{label} must contain exactly two bounds.")

        lower = _number(value[0], label)
        upper = _number(value[1], label)
    else:
        lower = _number(value, label)
        upper = default_upper

    if lower > upper or lower == math.inf or upper == -math.inf:
        raise ValueError(f"{label} has invalid lower/upper bounds.")

    return lower, upper


def _load_media(source: MediaInput) -> Mapping[str, object]:
    """Read one media mapping without reserving a base_medium entry."""
    data: object
    if isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    else:
        data = source

    if not isinstance(data, Mapping) or not data:
        raise ValueError("Provide at least one named medium.")

    if any(not isinstance(key, str) or not key for key in data):
        raise ValueError("Medium IDs must be nonempty strings.")

    return data


class MultimediaMinimizer:
    """Coordinate one joint minimum-cardinality search over named media."""

    def __init__(
        self,
        community: LayeredCommunity,
        media: MediaInput,
        minimal_growth: float = 0.01,
        growth_overrides: Optional[Mapping[str, Mapping[str, float]]] = None,
        env: Optional[Env] = None,
        params: Optional[Mapping[Parameter, float]] = None,
    ) -> None:
        """Load conditions and construct an independent joint solver.

        Parameters
        ----------
        community : LayeredCommunity
            Existing network template, whose model and solver are not modified.
        media : str, Path, or Mapping
            YAML path or mapping of medium IDs to environmental exchange bounds.
            A scalar specifies a lower bound; a pair specifies both bounds.
            Omitted exchanges have uptake closed and retain their upper bounds.
        minimal_growth : float, optional
            Positive finite default for every organism's growth flux; default 0.01.
        growth_overrides : Mapping, optional
            Thresholds keyed by medium, then merged organism biomass reaction ID.
            Missing media or reaction entries use minimal_growth. Every selected
            organism must meet its threshold in every medium.
        env : Env, optional
            Gurobi environment for the new joint solver.
        params : Mapping, optional
            Solver parameters overriding the community's solver parameters.

        Raises
        ------
        ValueError
            If the community is empty, media or growth requirements are invalid,
            or a medium names an existing reaction outside the exchange map.

        Notes
        -----
        Unknown medium reactions are logged and ignored. Every YAML top-level
        key names an independent medium; base_medium has no special meaning.
        """
        self.community = community
        if not community.organisms:
            raise ValueError("The community must contain at least one organism.")

        self.minimal_growth = _growth_threshold(minimal_growth, "minimal_growth")
        self.media = self._normalize_media(_load_media(media))
        self.growth_thresholds = self._normalize_growth(growth_overrides or {})
        self.joint_solver = JointSolver(
            community, self.media, self.growth_thresholds, env=env, params=params  # type: ignore
        ) 

    def _normalize_media(
        self, media: Mapping[str, object]
    ) -> dict[str, dict[str, FluxBounds]]:
        """Validate environmental keys and close uptake for omitted exchanges."""
        model = self.community.merged_model
        exchanges = set(self.community._ext_met_exchange_map.values())

        normalized: dict[str, dict[str, FluxBounds]] = {}
        for medium, composition in media.items():
            if not isinstance(composition, Mapping):
                raise ValueError(f"Medium {medium!r} must map reaction IDs to bounds.")

            bounds = {
                reaction: _bounds(0.0, float(model.reactions[reaction].ub), reaction)
                for reaction in self.community._ext_met_exchange_map.values()
            }

            for reaction, value in composition.items():
                if not isinstance(reaction, str):
                    raise ValueError(
                        f"Reaction IDs in medium {medium!r} must be strings."
                    )

                if reaction not in model.reactions:
                    logging.warning(
                        "Missing reaction %s in model %s for medium %s",
                        reaction,
                        model.id,
                        medium,
                    )
                    continue

                if reaction not in exchanges:
                    raise ValueError(
                        f"Reaction {reaction!r} in medium {medium!r} is not an environmental exchange."
                    )

                bounds[reaction] = _bounds(
                    value, float(model.reactions[reaction].ub), f"{medium}/{reaction}"
                )

            normalized[medium] = bounds

        return normalized

    def _growth_reactions(self) -> set[str]:
        """Identify organism growth through reaction maps."""
        reactions: set[str] = set()
        for organism, model in self.community.organisms.items():
            biomass = model.biomass_reaction
            if (
                biomass is None
                or (organism, biomass) not in self.community.reaction_map
            ):
                raise ValueError(
                    f"Organism {organism!r} has no mapped biomass reaction."
                )
            reactions.add(self.community.reaction_map[(organism, biomass)])

        return reactions

    def _normalize_growth(
        self, overrides: Mapping[str, Mapping[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Resolve partial medium/reaction overrides against the common default."""
        unknown_media = set(overrides) - set(self.media)
        if unknown_media:
            raise ValueError(
                f"Unknown media in growth_overrides: {sorted(unknown_media)}"
            )

        growth_reactions = self._growth_reactions()
        normalized: dict[str, dict[str, float]] = {}

        # values not specified get mapped to the minimal_growth value, otherwise overrides are applied
        for medium in self.media:
            thresholds = dict.fromkeys(sorted(growth_reactions), self.minimal_growth)

            entries = overrides.get(medium, {})
            if not isinstance(entries, Mapping):
                raise ValueError(f"Growth overrides for {medium!r} must be a mapping.")

            for reaction, value in entries.items():
                if reaction not in growth_reactions:
                    raise ValueError(
                        f"{reaction!r} is not a merged organism growth reaction."
                    )
                thresholds[reaction] = _growth_threshold(value, f"{medium}/{reaction}")

            normalized[medium] = thresholds

        return normalized

    def minimize(self) -> MultimediaResult:
        """Minimize community size subject to growth in every medium.

        Returns
        -------
        MultimediaResult
            One shared species selection, per-medium fluxes, solver termination
            status, and objective lower bound. A feasible incumbent may be
            SUBOPTIMAL if optimization stops before proving minimum size.

        Raises
        ------
        RuntimeError
            If the minimizer's joint solver has already been closed.
        """
        return self.joint_solver.solve()

    def close(self) -> None:
        """Release the joint solver without closing the supplied community."""
        self.joint_solver.close()
