"""Research architectures; shared training lives in mfp_nova.research."""

LEGACY_MODELS = ("trirelay", "geofactor", "protoroute", "helixset", "theraset")
HH_MODELS = ("hh_setreread", "hh_setreread_v12", "hh_setreread_v13", "hh_setreread_v14")
RESEARCH_MODELS = (*LEGACY_MODELS, "termiscale", "motifspan", "privilegedbridge", "setreread", *HH_MODELS)
