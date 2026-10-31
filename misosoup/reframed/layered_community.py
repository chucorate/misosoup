"""Layered community class implementation."""

import logging
import math
from collections import Counter
from typing import Mapping, Optional

from gurobipy import Env
from reframed import (
    CBModel,
    CBReaction,
    Community,
    Compartment,
    Gene,
    GPRAssociation,
    Metabolite,
    Protein,
    ReactionType,
)
from reframed.solvers.solver import Parameter, VarType

from ..reframed.gurobi_env_solver import GurobiEnvSolver

BOUND_INF = 1000


class LayeredCommunity(Community):
    """Community model with additional layer of exchange reactions for each member."""

    default_environment = Env(params={"LogToConsole": 0, "Method": 1})

    def __init__(
        self,
        community_id: str,
        models: list[CBModel],
        env: Optional[Env] = None,
        copy_models: bool = False,
        suffix: str = "_i",
        params: Optional[dict[Parameter, float]] = None,
        _exchange_map: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Initialize a layered community.

        Parameters
        ----------
        community_id : str
            Identifier of the merged model.
        models : list[CBModel]
            Organism models to merge.
        env : Env, optional
            Gurobi environment; defaults to the shared environment.
        copy_models : bool, optional
            Whether to copy the organism models.
        suffix : str, optional
            Suffix for organism-local exchange reactions.
        params : dict[Parameter, float], optional
            Solver parameters.
        exchange_map : dict[str, str], optional
            Environmental reaction IDs keyed by shared metabolite ID. Pass the
            parent community's map when rebuilding a subset.
        """
        super().__init__(
            community_id=community_id,
            models=models,
            copy_models=copy_models,
        )

        if env is None:
            env = self.default_environment

        self.suffix = suffix
        self._ext_met_exchange_map: dict[str, str] = (
            _exchange_map if _exchange_map is not None else {}
        )

        self.solver = GurobiEnvSolver(model=self.merged_model, env=env, params=params)
        self.has_binary_variables = False

    def merge_models(self) -> CBModel:
        """Merge organism models into a community model."""
        comm_model = CBModel(self.id)
        old_ext_comps = []
        ext_mets_ids: set[str] = set()
        original_exchange_ids_by_met: dict[str, set[str]] = {}

        self.reaction_map = {}
        self.metabolite_map = {}

        # default IDs
        ext_comp_id = "ext"
        comm_biomass_id = "community_biomass"
        comm_growth = "community_growth"

        # create external compartment

        ext_comp = Compartment(ext_comp_id, "extracellular environment", external=True)
        comm_model.add_compartment(ext_comp)

        # community biomass

        comm_biomass_met = Metabolite(comm_biomass_id, "Total community biomass", ext_comp_id)
        comm_model.add_metabolite(comm_biomass_met)

        comm_biomass_rxn = CBReaction(
            comm_growth,
            name="Community growth rate",
            reversible=False,
            stoichiometry={comm_biomass_id: -1},
            lb=0,
            ub=math.inf,
            objective=1,
        )

        comm_model.add_reaction(comm_biomass_rxn)

        # add each organism
        for org_id, model in self.organisms.items():

            def rename(old_id):
                return f"{old_id}_{org_id}"

            # add internal compartments
            for c_id, comp in model.compartments.items():
                if comp.external:
                    old_ext_comps.append(c_id)
                new_comp = Compartment(rename(c_id), comp.name)
                comm_model.add_compartment(new_comp)

            # add metabolites
            for m_id, met in model.metabolites.items():
                # always create a local metabolite
                new_met_id = rename(m_id)
                new_met = Metabolite(new_met_id, met.name, rename(met.compartment))
                new_met.metadata = met.metadata.copy()
                comm_model.add_metabolite(new_met)
                self.metabolite_map[(org_id, m_id)] = new_met_id

                # if the metabolite is external and not yet added to the
                # community model, add it
                if (
                    met.compartment in old_ext_comps
                    and m_id not in comm_model.metabolites
                ):
                    new_met = Metabolite(m_id, met.name, ext_comp_id)
                    new_met.metadata = met.metadata.copy()
                    comm_model.add_metabolite(new_met)
                    ext_mets_ids.add(new_met.id)

            # add genes
            for g_id, gene in model.genes.items():
                new_gene_id = rename(g_id)
                new_gene = Gene(new_gene_id, gene.name)
                new_gene.metadata = gene.metadata.copy()
                comm_model.add_gene(new_gene)

            # add internal reactions
            for r_id, rxn in model.reactions.items():
                new_rxn_id = rename(r_id)

                is_external_exchange: bool = (
                    rxn.reaction_type == ReactionType.EXCHANGE
                    and r_id.startswith("R_EX")
                    # check if the single metabolite is in an external compartment
                    # this avoids treating internal demand reactions as exchange ones
                    and r_id != model.biomass_reaction
                    and model.compartments[
                        model.metabolites[list(rxn.stoichiometry.keys())[0]].compartment
                    ].external
                )

                if is_external_exchange:
                    if len(rxn.stoichiometry) != 1:
                        logging.warning(
                            "External exchange %s in organism %s "
                            "must contain exactly one metabolite", r_id, org_id
                        )

                    new_rxn_id = new_rxn_id + self.suffix

                    # assumes the exchange reaction is of the form {m_id: -1}
                    ext_m_id = list(rxn.stoichiometry.keys())[0]
                    int_m_id = rename(ext_m_id)

                    if ext_m_id in ext_mets_ids:
                        original_exchange_ids_by_met.setdefault(ext_m_id, set()).add(r_id)

                    new_stoichiometry = {
                        ext_m_id: 1,
                        int_m_id: -1,
                    }
                    new_rxn = CBReaction(
                        new_rxn_id,
                        reversible=True,
                        stoichiometry=new_stoichiometry,
                        reaction_type=ReactionType.EXCHANGE,
                    )
                else:
                    new_stoichiometry = {
                        rename(m_id): coeff for m_id, coeff in rxn.stoichiometry.items()
                    }

                    if r_id == model.biomass_reaction:
                        new_stoichiometry[comm_biomass_id] = 1

                    if rxn.gpr is None:
                        new_gpr = None
                    else:
                        new_gpr = GPRAssociation()
                        new_gpr.metadata = rxn.gpr.metadata.copy()

                        for protein in rxn.gpr.proteins:
                            new_protein = Protein()
                            new_protein.genes = [rename(g_id) for g_id in protein.genes]
                            new_protein.metadata = protein.metadata.copy()
                            new_gpr.proteins.append(new_protein)

                    new_rxn = CBReaction(
                        new_rxn_id,
                        name=rxn.name,
                        reversible=rxn.reversible,
                        stoichiometry=new_stoichiometry,
                        reaction_type=rxn.reaction_type,
                        lb=rxn.lb,
                        ub=rxn.ub,
                        gpr_association=new_gpr,
                    )

                comm_model.add_reaction(new_rxn)
                new_rxn.metadata = rxn.metadata.copy()
                self.reaction_map[(org_id, r_id)] = new_rxn_id

        if not self._ext_met_exchange_map:
            self._build_exchange_map(comm_model, ext_mets_ids, original_exchange_ids_by_met)

        # parent maps can contain metabolites belonging only to removed organisms,
        # so only metabolites from included organisms are kept  
        self._ext_met_exchange_map = {
            m_id: self._ext_met_exchange_map[m_id] for m_id in sorted(ext_mets_ids)
        }

        for m_id, r_id in self._ext_met_exchange_map.items():
            # exchange selection and solution validation requires this naming convention
            if not r_id.startswith("R_EX_") or r_id.endswith("_i"):
                raise ValueError(
                    f"Environmental exchange {r_id!r} must start with 'R_EX_' and must not end with '_i'."
                )

            rxn = CBReaction(
                r_id,
                reversible=True,
                stoichiometry={m_id: -1},
                reaction_type=ReactionType.EXCHANGE,
            )
            comm_model.add_reaction(rxn)

        return comm_model

    def _build_exchange_map(
        self,
        comm_model: CBModel,
        ext_mets_ids: set[str],
        original_exchange_ids_by_met: dict[str, set[str]],
    ) -> None:
        """Allocate external exchange IDs for the initial community, avoiding collisions."""
        # the actual names of the exchange reactions are used as candidates,
        # the exchange reaction is only reconstructed when no source exchange exists
        candidates = {
            metabolite_id: original_exchange_ids_by_met.get(
                metabolite_id,
                {
                    f"R_EX_{metabolite_id[2:] if metabolite_id.startswith('M_') else metabolite_id}"
                },
            )
            for metabolite_id in ext_mets_ids
        }

        candidate_counts = Counter(
            reaction_id
            for reaction_ids in candidates.values()
            for reaction_id in reaction_ids
        )

        # reserve merged reactions and candidate IDs so suffixes cannot overwrite them
        assigned_ids: set[str] = set(comm_model.reactions)
        reserved_ids: set[str] = assigned_ids | set(candidate_counts)

        exchange_map: dict[str, str] = {}

        # sorting keeps suffix allocation independent of organism order
        for m_id in sorted(candidates):
            r_id = min(candidates[m_id])

            # if there are collisions or the ids have already been assigned,
            # avoid overwriting the reaction by giving them an index
            if candidate_counts[r_id] > 1 or r_id in assigned_ids:
                source_id = r_id
                index = 1
                r_id = f"{source_id}__{index}"

                while r_id in reserved_ids:
                    index += 1
                    r_id = f"{source_id}__{index}"

            exchange_map[m_id] = r_id
            assigned_ids.add(r_id)
            reserved_ids.add(r_id)

        self._ext_met_exchange_map = exchange_map

    def setup_binary_variables(self, minimal_growth):
        """Setup binary variables for each organism."""
        for org_id in self.organisms.keys():
            self.solver.add_variable(
                f"y_{org_id}",
                0,
                1,
                vartype=VarType.BINARY,
            )

        self.solver.update()

        for org_id, org_model in self.organisms.items():
            org_var = f"y_{org_id}"
            for r_id, reaction in org_model.reactions.items():
                if (
                    not r_id.startswith("R_EX")
                    and r_id != org_model.biomass_reaction
                    and reaction.lb * reaction.ub <= 0
                ):
                    continue

                merged_id = self.reaction_map[(org_id, r_id)]
                ubound = BOUND_INF
                lbound = -BOUND_INF

                if r_id == org_model.biomass_reaction:
                    lbound = minimal_growth

                if reaction.lb * reaction.ub > 0:
                    lbound = -BOUND_INF if math.isinf(reaction.lb) else reaction.lb
                    ubound = BOUND_INF if math.isinf(reaction.ub) else reaction.ub
                    self.solver.reset_bounds({merged_id: (-BOUND_INF, BOUND_INF)})

                self.solver.add_constraint(
                    f"c_{merged_id}_lb",
                    {merged_id: 1, org_var: -lbound},
                    ">",
                    0,
                )
                self.solver.add_constraint(
                    f"c_{merged_id}_ub",
                    {merged_id: 1, org_var: -ubound},
                    "<",
                    0,
                )

        self.solver.update()
        self.has_binary_variables = True

    def setup_growth_requirement(self, minimal_growth):
        for org_id, org_model in self.organisms.items():
            merged_id = self.reaction_map[(org_id, org_model.biomass_reaction)]
            self.solver.add_constraint(
                f"c_{merged_id}_lb",
                {merged_id: 1},
                ">",
                minimal_growth,
            )
        self.solver.update()

    def setup_medium(self, medium: Mapping[str, float]) -> None:
        """Setup the medium for model on solver."""
        missing_reactions = set(medium.keys()) - set(self.merged_model.reactions.keys())
        for r_id in sorted(missing_reactions):
            logging.warning(
                "Missing reaction %s in model %s", r_id, self.merged_model.id
            )
        for r_id in self._ext_met_exchange_map.values():
            bound = medium.get(r_id, 0)
            self.solver.add_constraint(
                f"c_{r_id}_lb",
                {r_id: 1},
                ">",
                bound,  # type: ignore
            )

    def setup_parsimony(self):
        # add absolute variables for each reaction
        for rid in self.merged_model.reactions:
            self.solver.add_variable(f"abs_{rid}_pos", 0, 1000)
            self.solver.add_variable(f"abs_{rid}_neg", 0, 1000)

        self.solver.update()

        # add absolute constraints for each reaction
        for rid in self.merged_model.reactions:
            self.solver.add_constraint(
                f"c_{rid}_abs",
                {f"abs_{rid}_pos": 1, f"abs_{rid}_neg": -1, rid: -1},
                "=",
                0,
            )

    def check_feasibility(self, values: list):
        existing_values = set(values) & set(self.merged_model.reactions.keys())
        logging.debug("Gathering variables: %s", existing_values)
        return self.solver.solve(get_values=existing_values)

    def objective_optimization(self, objective: dict, values: list):
        existing_values = set(values) & set(self.merged_model.reactions.keys())
        return self.solver.solve(
            objective=objective,
            get_values=existing_values,
            minimize=False,
        )

    def parsimony_optimization(self, objective: dict, value: float, values: list):
        if value > 0:
            self.solver.add_constraint(
                "c_growth",
                objective,
                ">",
                value,
            )
            self.solver.update()

        existing_values = set(values) & set(self.merged_model.reactions.keys())
        parsimony_solution = self.solver.solve(
            objective={
                f"abs_{rid}_{sense}": 1
                for rid in self.merged_model.reactions
                for sense in ["pos", "neg"]
            },
            get_values=existing_values,
            minimize=True,
        )

        self.solver.remove_constraint("c_growth")

        return parsimony_solution
