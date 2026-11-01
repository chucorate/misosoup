"""Joint community optimization with shared species selection across media."""

import math
from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
from gurobipy import GRB, Env, GurobiError, MVar, Model, Var
from numpy.typing import NDArray
from reframed import CBModel
from reframed.solvers.gurobi_solver import status_mapping
from reframed.solvers.solution import Status
from reframed.solvers.solver import Parameter
from scipy.sparse import coo_matrix, csr_matrix

from .gurobi_env_solver import GurobiEnvSolver
from .layered_community import LayeredCommunity

FluxBounds = tuple[float, float]
MediaBounds = dict[str, dict[str, FluxBounds]]
GrowthThresholds = dict[str, dict[str, float]]


@dataclass(frozen=True)
class MultimediaResult:
    """Result of selecting a community viable in every supplied medium.

    Attributes
    ----------
    status : Status
        Optimization outcome; an incumbent without proven optimality is SUBOPTIMAL.
    solver_status : int
        Original Gurobi termination code, including limits and interruptions.
    community : tuple of str
        Selected organism IDs, empty when no feasible solution was found.
    cardinality : int or None
        Number of selected organisms, or None when no solution was found.
    fluxes : dict
        Fluxes keyed by medium ID, then original merged reaction ID.
    objective_bound : float or None
        Solver's lower bound on the minimum community size, when available.
    """

    status: Status
    solver_status: int
    community: tuple[str, ...]
    cardinality: Optional[int]
    fluxes: dict[str, dict[str, float]]
    objective_bound: Optional[float]


def _sparse_stoichiometry_matrix(model: CBModel, reaction_ids: tuple[str, ...]) -> csr_matrix:
    """Build CSR stoichiometry with model metabolite order and requested columns."""
    metabolite_index: dict[str, int] = {
        metabolite_id: index for index, metabolite_id in enumerate(model.metabolites)
    }
    met_rows: list[int] = []
    rxn_columns: list[int] = []
    coefficients: list[float] = []

    for reaction_column, reaction_id in enumerate(reaction_ids):
        for metabolite_id, coefficient in model.reactions[
            reaction_id
        ].stoichiometry.items():
            # only non-zero entries are considered
            if coefficient:
                met_rows.append(metabolite_index[metabolite_id])
                rxn_columns.append(reaction_column)
                coefficients.append(float(coefficient))

    return coo_matrix(
        (coefficients, (met_rows, rxn_columns)),
        shape=(len(metabolite_index), len(reaction_ids)),
    ).tocsr()  # type: ignore


class JointSolver:
    """Own a sparse MILP with shared binaries and independent medium fluxes."""

    def __init__(
        self,
        community: LayeredCommunity,
        media: MediaBounds,
        growth_thresholds: GrowthThresholds,
        env: Optional[Env] = None,
        params: Optional[dict[Parameter, float]] = None,
    ) -> None:
        """
        Initialize and build the joint optimization problem.

        Parameters
        ----------
        community : LayeredCommunity
            Community model, including reaction ownership and exchange maps.
        media : dict of dict
            Complete environmental exchange bounds by medium, then reaction ID.
            Each bound is a normalized (lower, upper) pair.
        growth_thresholds : dict of dict
            Positive finite thresholds for every medium and merged biomass ID.
        env : Env, optional
            Gurobi environment; defaults to LayeredCommunity's shared environment.
        params : dict, optional
            Overrides for the community's solver parameters. Unspecified absolute
            and relative optimality gaps default to zero when not inherited.
        """
        self.community = community

        self._model: CBModel = community.merged_model
        self._media = media
        self._growth_thresholds = growth_thresholds
        self._reaction_ids: tuple[str, ...] = tuple(self._model.reactions)
        self._reaction_index = {
            reaction: index for index, reaction in enumerate(self._reaction_ids)
        }
        self._organism_ids: tuple[str, ...] = tuple(community.organisms)

        # the stoichiometry matrix is passed as a sparse matrix
        # to avoid copying the matrix for each medium
        self._rxn_idx_org_idx = self._build_reaction_idx_org_idx_mapping()
        self._sparse_stoichiometry = _sparse_stoichiometry_matrix(self._model, self._reaction_ids)

        parameters = dict(community.solver.params or {})
        parameters.setdefault(Parameter.MIP_REL_GAP, 0.0)
        parameters.setdefault(Parameter.MIP_ABS_GAP, 0.0)
        parameters.update(params or {})

        self._solver = GurobiEnvSolver(
            env=env if env is not None else community.default_environment,
            params=parameters,
        )
        self.problem: Model = self._solver.problem

        self._closed = False
        self._flux_variables: dict[str, MVar] = {}

        try:
            self._build_problem()
        except Exception:
            self.close()
            raise

    def _build_reaction_idx_org_idx_mapping(self) -> dict[int, int]:
        """Map organism-owned reaction columns to shared binary variable indexes."""
        organism_index = {
            organism: index for index, organism in enumerate(self._organism_ids)
        }

        return {
            self._reaction_index[merged_id]: organism_index[organism]
            for (organism, _), merged_id in self.community.reaction_map.items()
        }

    def _build_problem(self) -> None:
        """Assemble one optimization problem without modifying the community."""
        self._setup_binary_variables()
        self._setup_flux_variables()
        self._setup_mass_balances()
        self._setup_flux_bounds()
        self._setup_environmental_bounds()
        self._setup_growth_requirements()

        self.problem.addConstr(self._binary_variables.sum() >= 1, name="nonempty")
        self.problem.setObjective(self._binary_variables.sum(), GRB.MINIMIZE)
        self.problem.update()

    def _setup_binary_variables(self) -> None:
        """Create the organism-selection variables shared across all media."""
        self._binary_variables: MVar = self.problem.addMVar(
            len(self._organism_ids), vtype=GRB.BINARY, name="selection"
        )

    def _setup_flux_variables(self) -> None:
        """Create per-medium fluxes, allowing zero for absent organisms' reactions."""
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []

        for index, reaction_id in enumerate(self._reaction_ids):
            reaction = self._model.reactions[reaction_id]
            lower = float(reaction.lb)
            upper = float(reaction.ub)

            if index in self._rxn_idx_org_idx:
                lower = min(lower, 0.0)
                upper = max(upper, 0.0)
    
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        for index, medium in enumerate(self._media):
            self._flux_variables[medium] = self.problem.addMVar(
                len(self._reaction_ids),
                lb=np.asarray(lower_bounds),
                ub=np.asarray(upper_bounds),
                name=f"flux_{index}",
            )

    def _setup_mass_balances(self) -> None:
        """Reuse one sparse stoichiometric matrix for each medium's balance block."""
        right_hand_side = np.zeros(len(self._model.metabolites))

        for index, variables in enumerate(self._flux_variables.values()):
            self.problem.addMConstr(
                cast(np.ndarray, self._sparse_stoichiometry),
                variables,
                GRB.EQUAL,
                right_hand_side,
                name=f"balance_{index}",
            )

    def _prepare_flux_constraints(
        self,
    ) -> tuple[csr_matrix, NDArray[np.str_], list[int]]:
        """Prepare flux-selection constraints shared by every medium.

        Returns
        -------
        csr_matrix
            Coefficients for finite nonzero bounds, with flux columns followed
            by organism-selection columns. Each row expresses v - bound*y.
        ndarray of str
            Lower-bound (>=) or upper-bound (<=) sense for each row.
        list of int
            Local reaction indexes with an infinite bound, requiring an
            indicator to force zero flux when their organism is absent.
        """
        rows: list[int] = []
        columns: list[int] = []
        coefficients: list[float] = []
        inequality_senses: list[str] = []
        unbounded_rxn_indices: list[int] = []

        for reaction_index, reaction_owner in self._rxn_idx_org_idx.items():
            reaction = self._model.reactions[self._reaction_ids[reaction_index]]

            for bound, inequality_sense in [
                (reaction.lb, GRB.GREATER_EQUAL),
                (reaction.ub, GRB.LESS_EQUAL),
            ]:
                if math.isfinite(bound) and bound != 0:
                    row = len(inequality_senses)
                    rows.extend([row, row])
                    columns.extend([reaction_index, len(self._reaction_ids) + reaction_owner])
                    coefficients.extend([1.0, -float(bound)])
                    inequality_senses.append(inequality_sense)

            if math.isinf(reaction.lb) or math.isinf(reaction.ub):
                unbounded_rxn_indices.append(reaction_index)
    
        flux_selection_matrix = cast(csr_matrix, coo_matrix(
            (coefficients, (rows, columns)),
            shape=(len(inequality_senses), len(self._reaction_ids) + len(self._organism_ids)),
        ).tocsr())

        return flux_selection_matrix, np.asarray(inequality_senses), unbounded_rxn_indices

    def _setup_flux_bounds(self) -> None:
        """Apply finite coupling rows and inactivity indicators in every medium.

        Zero bounds remain variable bounds. Infinite bounds use indicators,
        avoiding an artificial finite cap on active organisms' fluxes.
        """
        flux_selection_matrix, inequality_senses, unbounded_rxn_indices = (
            self._prepare_flux_constraints()
        )

        # bounded fluxes
        selection_variables: list[Var] = self._binary_variables.tolist()
        constraint_count = len(inequality_senses)
        right_hand_side = np.zeros(constraint_count, dtype=np.float64)

        # unbounded fluxes
        unbounded_organisms: NDArray[np.intp] = np.asarray(
            [self._rxn_idx_org_idx[index] for index in unbounded_rxn_indices], dtype=np.intp
        )
        unbounded_flux_indices: NDArray[np.intp] = np.asarray(unbounded_rxn_indices, dtype=np.intp)

        for index, variables in enumerate(self._flux_variables.values()):
            if constraint_count:
                self.problem.addMConstr(
                    cast(np.ndarray, flux_selection_matrix),
                    variables.tolist() + selection_variables,
                    inequality_senses,
                    right_hand_side,
                    name=f"bounds_{index}",
                )
            if unbounded_rxn_indices:
                self.problem.addGenConstrIndicator(
                    self._binary_variables[unbounded_organisms],  # type: ignore
                    False,
                    variables[unbounded_flux_indices],  # type: ignore
                    GRB.EQUAL,
                    0.0,
                    name=f"inactive_{index}",
                )

    def _setup_environmental_bounds(self) -> None:
        """Assign each medium's lower and upper bounds using the exchange map."""
        exchange_indices = [
            self._reaction_index[reaction_id]
            for reaction_id in self.community._ext_met_exchange_map.values()
        ]
        reaction_ids = tuple(self.community._ext_met_exchange_map.values())

        for medium, variables in self._flux_variables.items():
            if exchange_indices:
                variables[exchange_indices].LB = [
                    self._media[medium][reaction][0] for reaction in reaction_ids
                ]  # type: ignore
                variables[exchange_indices].UB = [
                    self._media[medium][reaction][1] for reaction in reaction_ids
                ]  # type: ignore

    def _setup_growth_requirements(self) -> None:
        """Require every selected organism to grow in each medium."""
        for medium_index, (medium, thresholds) in enumerate(
            self._growth_thresholds.items()
        ):
            reaction_indices = [
                self._reaction_index[reaction] for reaction in thresholds
            ]
            reaction_owners = [self._rxn_idx_org_idx[index] for index in reaction_indices]

            self.problem.addConstr(
                self._flux_variables[medium][reaction_indices]
                >= np.asarray(list(thresholds.values()))
                * self._binary_variables[reaction_owners],
                name=f"growth_{medium_index}",
            )

    def solve(self) -> MultimediaResult:
        """
        Minimize shared community size and return any available incumbent.

        Returns
        -------
        MultimediaResult
            Selected organisms, per-medium fluxes, termination status, and the
            objective lower bound. An incumbent without proven optimality is
            SUBOPTIMAL; no incumbent yields an empty community tuple and fluxes.

        Raises
        ------
        RuntimeError
            If the joint solver has already been closed.
        """
        if self._closed:
            raise RuntimeError("The joint solver is closed.")

        self.problem.optimize()
        solver_status = int(self.problem.Status)

        try:
            objective_bound = float(self.problem.ObjBound)
        except (AttributeError, GurobiError):
            objective_bound = None

        if not self.problem.SolCount:
            return MultimediaResult(
                status=status_mapping.get(solver_status, Status.UNKNOWN),
                solver_status=solver_status,
                community=(),
                cardinality=None,
                fluxes={},
                objective_bound=objective_bound,
            )

        selected_organisms = tuple(
            organism
            for organism, value in zip(self._organism_ids, self._binary_variables.X)  # type: ignore
            if value > 0.5
        )

        status = Status.OPTIMAL if solver_status == GRB.OPTIMAL else Status.SUBOPTIMAL

        # check if there could be smaller communities
        if objective_bound is not None and math.isfinite(objective_bound):
            if math.ceil(objective_bound - 1e-6) < len(selected_organisms):
                status = Status.SUBOPTIMAL

        fluxes = {
            medium: dict(zip(self._reaction_ids, map(float, variables.X)))  # type: ignore
            for medium, variables in self._flux_variables.items()
        }

        return MultimediaResult(
            status=status,
            solver_status=solver_status,
            community=selected_organisms,
            cardinality=len(selected_organisms),
            fluxes=fluxes,
            objective_bound=objective_bound,
        )

    def close(self) -> None:
        """Release the owned optimization model, leaving the environment open."""
        if not self._closed:
            self.problem.dispose()
            self._closed = True
