from _base import GeneralPolicyIterationComponent
from mdp._trial_interface import TrialInterface
import numpy as np 


class StandardTrialInterfaceBasedPolicyImprover(GeneralPolicyIterationComponent):
    def __init__(self, trial_interface, random_state: np.random.RandomState = None):
        super().__init__()
        self.trial_interface = trial_interface
        self.random_state = random_state or np.random.RandomState()

    def step(self):
        q = getattr(self.workspace, "_q", {})
        old_policy = getattr(self.workspace, "_policy", None)
        mdp = getattr(self.trial_interface, "mdp", self.trial_interface)

        new_action_map = {}
        for s in getattr(mdp, "states", []):
            actions = []
            if hasattr(mdp, "get_actions_in_state"):
                actions = mdp.get_actions_in_state(s)
            if len(actions) == 0:
                continue

            if s in q and q[s]:
                best_val = max(q[s].values())
                best_actions = [a for a, val in q[s].items() if val == best_val]
                a_star = self.random_state.choice(best_actions)
            else:
                if old_policy is not None:
                    try:
                        a_star = old_policy(s)
                    except Exception:
                        a_star = self.random_state.choice(actions)
                else:
                    a_star = self.random_state.choice(actions)
            new_action_map[s] = a_star

        def policy_fn(s):
            return new_action_map[s]

        self.workspace._policy = policy_fn
        return {"n_states": len(new_action_map)}
