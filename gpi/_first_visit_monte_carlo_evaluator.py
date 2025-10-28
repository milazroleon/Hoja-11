import numpy as np
import pandas as pd

from mdp._trial_interface import TrialInterface
from gpi._trial_based_policy_evaluator import TrialBasedPolicyEvaluator


class FirstVisitMonteCarloEvaluator(TrialBasedPolicyEvaluator):

    def __init__(
            self,
            trial_interface: TrialInterface,
            gamma: float,
            exploring_starts: bool,
            max_trial_length: int = np.inf,
            random_state: np.random.RandomState = None
        ):
        super().__init__(
            trial_interface=trial_interface,
            gamma=gamma,
            exploring_starts=exploring_starts,
            max_trial_length=max_trial_length,
            random_state=random_state
        )
        self.N = {} 

        if getattr(self.workspace, "_q", None) is None:
            try:
                self.workspace._q = {}
            except Exception:
                pass

    def process_trial_for_policy(self, df_trial, policy):

        if not {"state", "action", "reward"}.issubset(set(df_trial.columns)):
            raise ValueError("df_trial must contain state, action and reward columns")

        Gs = [0.0] * len(df_trial)
        G = 0.0
        for t in range(len(df_trial)-1, -1, -1):
            r = df_trial.loc[t, "reward"]
            G = self.gamma * G + (0.0 if pd.isna(r) else float(r))
            Gs[t] = G

        first_visits = set()
        q = getattr(self.workspace, "_q", {})

        for t in range(len(df_trial)-1):  
            s = df_trial.loc[t, "state"]
            a = df_trial.loc[t, "action"]
            if (s,a) in first_visits:
                continue
            first_visits.add((s,a))
            # initialize
            if s not in q:
                q[s] = {}
            if a not in q[s]:
                q[s][a] = 0.0
            k = self.N.get((s,a), 0) + 1
            self.N[(s,a)] = k
            q[s][a] = q[s][a] + (Gs[t] - q[s][a]) / k

        try:
            self.workspace._q = q
        except Exception:
            self.workspace._q = q

        return {"updated_pairs": len(first_visits)}
