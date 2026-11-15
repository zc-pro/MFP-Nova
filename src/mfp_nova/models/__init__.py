"""Research architectures; shared training lives in mfp_nova.research."""

LEGACY_MODELS = ("trirelay", "geofactor", "protoroute", "helixset", "theraset")
HH_MODELS = ("hh_setreread", "hh_setreread_v12", "hh_setreread_v13", "hh_setreread_v14")
HH_V15_RANKER = "hh_setreread_v15_ranker"
HH_V16 = "hh_setreread_v16"
SEQUENCE_ONLY_MODELS = ("termiscale", "motifspan", HH_V15_RANKER)
RESEARCH_MODELS = (*LEGACY_MODELS, *SEQUENCE_ONLY_MODELS, "privilegedbridge", "setreread", *HH_MODELS, HH_V16)
