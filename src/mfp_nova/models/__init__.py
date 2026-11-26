"""Research architectures; shared training lives in mfp_nova.research."""

LEGACY_MODELS = ("trirelay", "geofactor", "protoroute", "helixset", "theraset")
HH_MODELS = ("hh_setreread", "hh_setreread_v12", "hh_setreread_v13", "hh_setreread_v14")
HH_V15_RANKER = "hh_setreread_v15_ranker"
HH_V16 = "hh_setreread_v16"
HH_V17 = "hh_setreread_v17"
HH_V20 = "hh_setreread_v20"
HH_V20_CHEMICAL_MODES = ("chemical_cnn", "chemical_cnn_fusion", "chemical_cnn_local")
PEPTIDE_V3 = "peptide_v3"
PEPTIDE_V6 = "peptide_v6"
PEPTIDE_V7 = "peptide_v7"
XBRIDGE = "xbridge"
THERASET_BRIDGE = "theraset_bridge"
UNIFIED_EVIDENCE = "unified_evidence"
DUALPATH_MFP = "dualpath_mfp"
V3_MODES = ("cnn", "additive", "product", "centered")
V7_HEADS = ("flat", "structured")
SEQUENCE_ONLY_MODELS = ("termiscale", "motifspan", HH_V15_RANKER, HH_V20, PEPTIDE_V3, PEPTIDE_V6, PEPTIDE_V7)
RESEARCH_MODELS = (*LEGACY_MODELS, *SEQUENCE_ONLY_MODELS, "privilegedbridge", "setreread",
                   *HH_MODELS, HH_V16, HH_V17, XBRIDGE, THERASET_BRIDGE, UNIFIED_EVIDENCE, DUALPATH_MFP)
