from _base import GeneralPolicyIterationComponent
from mdp import ClosedFormMDP
from mdp._trial_interface import TrialInterface
import numpy as np
import pandas as pd
from abc import abstractmethod


class TrialBasedPolicyEvaluator(GeneralPolicyIterationComponent):
    def __init__(
        self,
        trial_interface,
        gamma: float,
        exploring_starts: bool,
        max_trial_length: int = np.inf,
        random_state: np.random.RandomState = None,
    ):
        super().__init__()
        self.trial_interface = trial_interface
        self.gamma = gamma
        self.max_trial_length = max_trial_length
        self.exploring_starts = exploring_starts
        if random_state is None:
            random_state = np.random.RandomState()
        self.random_state = random_state

    def step(self):
        """
        Creates and processes a trial to update state-values and q-values.
        """
        pi = self.workspace.policy
        s0 = None
        pi_used = pi

        # --- Exploring starts handling ---
        if self.exploring_starts:
            if hasattr(self.trial_interface, "init_states"):
                init_states = list(self.trial_interface.init_states)
            else:
                init_states = getattr(self.trial_interface, "states", [])
            if len(init_states) > 0:
                s0 = self.random_state.choice(init_states)
            actions = []
            # try to get available actions
            if hasattr(self.trial_interface, "get_actions_in_state"):
                actions = self.trial_interface.get_actions_in_state(s0)
            elif hasattr(self.trial_interface, "mdp"):
                actions = self.trial_interface.mdp.get_actions_in_state(s0)
            if len(actions) > 0:
                a0 = self.random_state.choice(actions)
            else:
                a0 = None

            first = {"used": False}

            def pi_es(s):
                if not first["used"] and s == s0:
                    first["used"] = True
                    return a0
                return pi(s)

            pi_used = pi_es

        # --- Generate trial (support different interfaces) ---
        if hasattr(self.trial_interface, "generate_trial"):
            df = self.trial_interface.generate_trial(pi_used, s=s0)
        elif hasattr(self.trial_interface, "run_trial"):
            df = self.trial_interface.run_trial(pi_used)
        elif hasattr(self.trial_interface, "record_trial"):
            df = self.trial_interface.record_trial(pi_used)
        else:
            raise AttributeError(
                "Trial interface has no method to generate or run a trial."
            )

        # --- Truncate trial ---
        if np.isfinite(self.max_trial_length):
            df = df.iloc[: int(self.max_trial_length) + 1].reset_index(drop=True)

        # normalize column names if possible
        cols = list(df.columns)
        lower_cols = [c.lower() for c in cols]
        rename_map = {}
        for i, c in enumerate(lower_cols):
            if "state" in c:
                rename_map[cols[i]] = "state"
            elif "action" in c:
                rename_map[cols[i]] = "action"
            elif "reward" in c or "return" in c:
                rename_map[cols[i]] = "reward"
        df = df.rename(columns=rename_map)

        if "state" not in df.columns or "action" not in df.columns:
            raise ValueError("Trial dataframe missing state/action columns")

        # --- Process trial ---
        report = self.process_trial_for_policy(df, pi_used)

        # --- Update V from Q ---
        q = getattr(self.workspace, "_q", None)
        if q is not None:
            v = {}
            for s in getattr(self.trial_interface, "states", []):
                try:
                    a = pi(s)
                    v[s] = q.get(s, {}).get(a, 0.0)
                except Exception:
                    v[s] = 0.0
            self.workspace._v = v

        return {"trial_length": len(df), **(report or {})}

    @abstractmethod
    def process_trial_for_policy(self, trial, policy):
        raise NotImplementedError