from phase313_multi_capability_campaign import MultiCapabilityCampaign

def run_frontier_integration(steps:int=4):
    campaign=MultiCapabilityCampaign()
    events=campaign.run(steps)
    return campaign,events
