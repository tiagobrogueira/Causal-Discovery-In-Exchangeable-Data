name_map = { "anm": "ANM", "cgnn": "CGNN", "emd": "EMD", "lingam": "LiNGAM", "pnl": "PNL", "qcd_function": "bQCD", "slope": "Slope","lcube": "LCube", "igci": "IGCI", "loci": "LOCI", "nncl": "NNCL", "heci": "HECI", "grci":"GRCI", "roche":"ROCHE","fom":"FOM","sloppy":"Sloppy","reci":"RECI","rcc":"RCC","cds":"CDS","cdci":"CDCI","bqcd":"bQCD","cam": "CAM", "slope": "Slope"}

def get_method_name(func):
    return name_map.get(func.__name__, func.__name__)