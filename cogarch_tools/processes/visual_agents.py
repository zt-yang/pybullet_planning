from world_builder.world import Agent


class VisualAgent(Agent):
    requires_rgb = requires_depth = True
    def __init__(self, world, **kwargs):
        super(VisualAgent, self).__init__(world, **kwargs)
    def policy(self, observation):
        # TODO: deep learning people
        return None


class SegmentAgent(Agent):
    requires_rgb = requires_depth = requires_segment = True
    def __init__(self, world, **kwargs):
        super(SegmentAgent, self).__init__(world, **kwargs)
    def policy(self, observation):
        # TODO: deep learning people
        return None