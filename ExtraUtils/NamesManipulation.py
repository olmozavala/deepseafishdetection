from constants.AI_params import AiModels

def getCompleteNameTxt(name):
    _run_name = F'RUN_NETWORK_IN_No-STD_OUTPUT_160x160'
    sections = name.split('_')
    id = sections[0]
    netname = getNetworkTypeTxt(name)
    in_names = getInputFieldsTxt(name)
    out_names = getOutputFieldsTxt(name)

    final_name = _run_name.replace("RUN", id)
    final_name = final_name.replace("NETWORK", netname)
    final_name = final_name.replace("IN", F"IN_{in_names}")
    final_name = final_name.replace("OUTPUT", F"OUT_{out_names}")
    return final_name

def getNetworkTypeTxt(name):
    sections = name[name.find("NET"):].split("_")
    return F"{sections[1]}"

def getNeworkArchitectureAndTypeFromName(networkName):
    if "Simple" in networkName:
        if "2" in networkName:
            return "SimpleCNN", NetworkTypes.SimpleCNN_2
        if "4" in networkName:
            return "SimpleCNN", NetworkTypes.SimpleCNN_4
        if "8" in networkName:
            return "SimpleCNN", NetworkTypes.SimpleCNN_8
        if "16" in networkName:
            return "SimpleCNN", NetworkTypes.SimpleCNN_16
    else:
        return AiModels.UNET_2D_SINGLE, NetworkTypes.UNET

def getInputFields(name):
    sections = name[name.find("IN"):].split("_")
    model_fields = ['temp', 'srfhgt', 'salin', 'u-vel.', 'v-vel.']
    obs_fields = ['sst', 'sss']
    # obs_fields = ['sst', 'sss', 'ssh']

    if "No-STD" in sections[1]:
        var_fields = []
    else:
        var_fields = ['tem', 'sal', 'ssh', 'mdt']
    return model_fields, obs_fields, var_fields

def getInputVarFields(name):
    sections = name[name.find("IN"):].split("_")
    if "No-STD" in sections[1]:
        var_fields = []
    else:
        var_fields = ['tem', 'sal', 'ssh', 'mdt']
    return var_fields

def getInputFieldsTxt(name):
    sections = name[name.find("IN"):].split("_")
    return sections[1]

def getOutputFields(name):
    # ProjTrainingParams.output_fields: ['temp', 'srfhgt', 'salin', 'u-vel.', 'v-vel.']
    sections = name[name.find("OUT"):].split("_")
    return [sections[1].lower()]

def getOutputFieldsTxt(name):
    sections = name[name.find("OUT"):].split("_")
    return sections[1]

def getBBOX(name):
    if "264x400" in name:
        return "264x400"
    return "Unknown"

def landperc(name):
    if "no_land" in name.lower():
        return "No Land"
    return "Unknown"