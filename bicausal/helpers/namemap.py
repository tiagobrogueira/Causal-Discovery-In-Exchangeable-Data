name_map = { "anm": "ANM", "cgnn": "CGNN",  "slope": "SLOPE","lcube": "LCUBE", "igci": "IGCI", "loci": "LOCI", "nncl": "NNCL", "heci": "HECI", "grci":"GRCI", "roche":"ROCHE","fom":"FOM","sloppy":"SLOPPY","reci":"RECI","rcc":"RCC","cds":"CDS","cdci":"CDCI","bqcd":"bQCD","cam": "CAM", "rdmdl": "RDMDL"}

def get_method_name(func):
    return name_map.get(func.__name__, func.__name__)