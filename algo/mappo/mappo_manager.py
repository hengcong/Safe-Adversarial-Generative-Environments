class MAPPOManager(object):
    def __init__(self, agent_specs, policy_ctor):
        self.agents = {aid: policy_ctor(spec) for aid, spec in agent_specs.items()}
