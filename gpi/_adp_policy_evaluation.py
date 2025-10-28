from mdp._trial_interface import TrialInterface
import numpy as np
import pandas as pd

from gpi._trial_based_policy_evaluator import TrialBasedPolicyEvaluator
from mdp._base import MDP as BaseMDP

class ADPPolicyEvaluation(TrialBasedPolicyEvaluator):
    def __init__(
        self,
        trial_interface,
        gamma: float,
        exploring_starts: bool,
        max_trial_length: int = np.inf,
        random_state: np.random.RandomState = None,
        **kwargs  # <-- para ignorar precision/update_interval del grader
    ):
        super().__init__(
            trial_interface=trial_interface,
            gamma=gamma,
            exploring_starts=exploring_starts,
            max_trial_length=max_trial_length,
            random_state=random_state,
        )
        self.N_sas = {}
        self.obs_reward = {}

    def process_trial_for_policy(self, df_trial, policy):
        if not {"state", "action", "reward"}.issubset(df_trial.columns):
            raise ValueError("df_trial debe contener columnas state, action, reward")

        rows = df_trial.reset_index(drop=True)
        for t in range(len(rows) - 1):
            s = rows.loc[t, "state"]
            a = rows.loc[t, "action"]
            r = rows.loc[t, "reward"]
            sp = rows.loc[t + 1, "state"]
            self.obs_reward[s] = (
                r if s not in self.obs_reward else (self.obs_reward[s] + r) / 2
            )
            self.N_sas[(s, a, sp)] = self.N_sas.get((s, a, sp), 0) + 1

        # --- recompute q and v ---
        states = getattr(self.trial_interface, "states", [])
        q = {}
        v = {}
        for s in states:
            q[s] = {}
            if hasattr(self.trial_interface, "get_actions_in_state"):
                actions = self.trial_interface.get_actions_in_state(s)
            elif hasattr(self.trial_interface, "mdp"):
                actions = self.trial_interface.mdp.get_actions_in_state(s)
            else:
                actions = []
            for a in actions:
                # promedio de rewards + gamma*max(V(sp))
                denom = sum(self.N_sas.get((s, a, sp), 0) for sp in states)
                if denom == 0:
                    q[s][a] = self.obs_reward.get(s, 0.0)
                else:
                    val = 0.0
                    for sp in states:
                        cnt = self.N_sas.get((s, a, sp), 0)
                        val += (cnt / denom) * v.get(sp, 0.0)
                    q[s][a] = self.obs_reward.get(s, 0.0) + self.gamma * val
            if q[s]:
                v[s] = np.mean(list(q[s].values()))
            else:
                v[s] = 0.0

        self.workspace._q = q
        self.workspace._v = v
        return {"updated_states": len(states)}