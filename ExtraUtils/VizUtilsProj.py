import cmocean

def chooseCMAP(fields):
    cmaps_fields = []
    for c_field in fields:
        if c_field == "srfhgt" or c_field == "ssh":
            cmaps_fields.append(cmocean.cm.deep_r)
        elif c_field == "temp" or c_field == "sst" or c_field == "temp":
            cmaps_fields.append(cmocean.cm.thermal)
        elif c_field == "salin" or c_field == "sss" or c_field == "sal":
            cmaps_fields.append(cmocean.cm.haline)
        elif c_field == "u-vel.":
            cmaps_fields.append(cmocean.cm.delta)
        elif c_field == "v-vel.":
            cmaps_fields.append(cmocean.cm.delta)
        else:
            cmaps_fields.append(cmocean.cm.thermal)
    return cmaps_fields


def getMinMaxPlot(fields):
    minmax = []
    for c_field in fields:
        if c_field == "srfhgt" or c_field == "ssh":
            minmax.append([-4, 4])
        elif c_field == "temp" or c_field == "sst" or c_field == "temp":
            minmax.append([-2, 2])
        elif c_field == "salin" or c_field == "sss" or c_field == "sal":
            minmax.append([-.2, .2])
        elif c_field == "u-vel":
            minmax.append([-1, 1])
        elif c_field == "v-vel":
            minmax.append([-1, 1])
        else:
            minmax.append(cmocean.cm.thermal)
    return minmax