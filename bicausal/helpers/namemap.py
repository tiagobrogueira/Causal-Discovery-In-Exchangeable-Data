name_map = { "anm": "ANM", "cgnn": "CGNN", "emd": "EMD", "lingam": "LiNGAM", "pnl": "PNL", "qcd_function": "bQCD", "SlopeR": "Slope","lcube": "LCube", "igci": "IGCI", "loci": "LOCI"}

def get_method_name(func):
    return name_map.get(func.__name__, func.__name__)