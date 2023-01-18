import xtgeo
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from plotly.colors import n_colors
import pathlib
import sys
import argparse
import pandas as pd
from dataclasses import dataclass
from typing import List, Union, Tuple, Literal, Optional
import dataclasses
from dateutil.relativedelta import relativedelta
#gram/mol
DEFAULT_CO2_MOLAR_MASS = 44
DEFAULT_WATER_MOLAR_MASS = 18

@dataclass
class SourceData:
    EFF_VOL: List[np.ndarray]
    source: str
    SWAT: List[np.ndarray]
    SGAS: List[np.ndarray]
    AMFG: List[np.ndarray]
    YMFG: List[np.ndarray]
    DWAT: Optional[np.ndarray] = None
    DGAS: Optional[np.ndarray] = None
    BWAT: Optional[np.ndarray] = None
    BGAS: Optional[np.ndarray] = None


def main(arguments):
    arguments = process_args(arguments)
    props = ["SWAT", "DWAT", "BWAT", "SGAS", "DGAS", "BGAS",  "AMFG", "YMFG"]
    out = extract_source_data(
        arguments.grid,
        arguments.unrst,
        arguments.init, props)
    return out

def process_args(arguments):
    args = make_parser().parse_args(arguments)
    if args.unrst is None:
        args.unrst = args.grid.replace(".EGRID", ".UNRST")
    if args.init is None:
        args.init = args.grid.replace(".EGRID", ".INIT")
    return args

def make_parser():
    pn = pathlib.Path(__file__).name
    parser = argparse.ArgumentParser(pn)
    parser.add_argument("grid", help="Grid (.EGRID) from which maps are generated")
    parser.add_argument("--unrst", help="Path to UNRST file. Will assume same base name as grid if not provided",
                        default=None)
    parser.add_argument("--init", help="Path to INIT file. Will assume same base name as grid if not provided",
                        default=None)
    return parser

def rm_all_zeros(pdf):
    pdf_sum = pdf.sum(axis=1)
    allzero_idx = np.where(pdf_sum == 0.0)
    return allzero_idx

def try_prop(unrst_file,grid,names):
    try:
        prop = xtgeo.gridproperties_from_file(unrst_file, grid=grid, names=names, dates="all")
    except ValueError:
        prop = None
    return prop

def _effective_densities(props_dfs_dict,
                         co2_molar_mass=DEFAULT_CO2_MOLAR_MASS,
                         water_molar_mass=DEFAULT_WATER_MOLAR_MASS):
    dwat = props_dfs_dict.dwat
    dgas = props_dfs_dict.dgas
    amfg = props_dfs_dict.amfg
    ymfg = props_dfs_dict.ymfg
    gas_mass_frac = _mole_to_mass_fraction(ymfg, co2_molar_mass, water_molar_mass)
    aqu_mass_frac = _mole_to_mass_fraction(amfg, co2_molar_mass, water_molar_mass)
    w_gas = (dgas) * gas_mass_frac.values
    w_aqu = (dwat) * aqu_mass_frac.values
    return w_gas, w_aqu

def _mole_to_mass_fraction(x, m_co2, m_h20):
    return x * m_co2 / (m_h20 + (m_co2 - m_h20) * x)


def extract_source_data(
        grid_file:str,
        unrst_file:str,
        init_file:str,
        props:List) -> SourceData:
    grid = xtgeo.grid_from_file(grid_file)
    properties = [try_prop(unrst_file, grid=grid, names=[p]) for p in props]
    ## Checking the available properties
    props_f = []
    properties_f = []
    for k in range(0,len(props)):
        if properties[k] is not None:
            props_f.append(props[k])
            properties_f.append(properties[k])
    props = props_f
    properties = properties_f
    ## Checking if it is not enough for computations
    if set(['SGAS','SWAT','DGAS','DWAT','AMFG','YMFG']).issubset(set(props_f)):
        source = 'PFlotran'
        print('Data Source is '+source)
    else:
        if set(['SGAS','SWAT','BGAS','BWAT','AMFG','YMFG']).issubset(set(props_f)):
            source = 'Eclipse'
            print('Data Source is ' + source)
        else:
            print('Information is not enough to compute CO2 mass')

    initf = xtgeo.gridproperties_from_file(init_file, grid=grid, names=["PORO"], dates="all")
    act_idx = np.where(grid.get_actnum().values1d == 1)

    props_dfs = [p.dataframe().loc[act_idx[0], :] for p in properties]
    prop_rm_idx = np.unique(list(np.concatenate([rm_all_zeros(p) for p in props_dfs], axis=1)[0]))
    props_dfs_final = [p.drop(p.index[prop_rm_idx], axis=0) for p in props_dfs]

    vols = pd.DataFrame(data={'VOL': grid.get_bulk_volume().get_npvalues1d()[act_idx[0]]})
    poro = initf.dataframe().loc[act_idx[0], :]
    vols = vols.set_index(poro.index)
    eff_vols = vols * poro.values
    eff_vols.columns = ["EFF_VOL"]
    eff_vols = eff_vols.drop(eff_vols.index[prop_rm_idx], axis=0)
    props_dfs_dict = {}
    props_dfs_dict['EFF_VOL'] = eff_vols
    for k in range(0, len(props)):
        props_dfs_dict[props[k]] = props_dfs_final[k]
    sd = SourceData(
        source= source,
        **{p: v for p,v in props_dfs_dict.items() }
    )
    return sd



def calculate_co2mass(source_data: SourceData,
                      co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,
                      water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS
                      ):
    eff_vols = source_data.EFF_VOL
    source = source_data.source
    sgas = source_data.SGAS
    swat = source_data.SWAT
    state_vol = [sgas*eff_vols.values,swat*eff_vols.values]
    if source == 'PFlotran':
        eff_dens = _effective_densities(source_data,co2_molar_mass,water_molar_mass)
    else:
        if source == 'Eclipse':
            bgas = source_data['BGAS']
            bwat = source_data['BWAT']
            eff_dens = [bgas*conv_gas,bwat*conv_wat]

    weights0 = [a * (b.values) for a,b in zip(eff_dens,state_vol)]
    for p in weights0: p.columns = [x.split("_")[1] for x in p.columns]
    pdf = sum(weights0)
    pdf.columns = ["CO2MASS_" + p for p in pdf.columns]
    return pdf

main(sys.argv[1:])





