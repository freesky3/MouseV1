import numpy as np

import typer
app = typer.Typer()

from src.v1model.experimental_data import ExperimentalData
from src.v1model.geometry import L4, L2_3
from src.v1model.SpatialConnectMatrix import SpatialConnectMatrix
from src.v1model.input import L4VisualInput
from src.v1model.NeuronTransfer import tabulate_response
from src.v1model.WilsonCowanModel import WCModel, solve_dynamical_system, do_dynamics
from src.v1model.default_config import Config

from src.tests import test_geometry
from src.tests import test_input
from src.tests import test_transfer
from src.tests import test_scm
from src.tests import test_WC
from src.tests import test_orientation


@app.command()
def main(
    check_geometry: bool = typer.Option(False, "--check_geometry", "-cg", help="plot the network of L4 and L2/3"), 
    check_input: bool = typer.Option(False, "--check_input", "-ci", help="plot the input of L4"),
    check_transfer: bool = typer.Option(False, "--check_transfer", "-ct", help="plot the transfer function"),
    check_connect: bool = typer.Option(False, "--check_connect", "-cc", help="plot the connection of L4 and L2/3"), 
    check_WC: bool = typer.Option(False, "--check_WC", "-cw", help="test the WC model"),
    check_orientation: bool = typer.Option(False, "--check_orientation", "-co", help="test the orientation"),
): 
    np.random.seed(42)
    if check_geometry:
        test_geometry()
        return

    if check_input:
        test_input()
        return
    
    if check_transfer:
        test_transfer()
        return
    
    if check_connect:
        test_scm()
        return
    
    if check_WC:
        test_WC()
        return
    
    if check_orientation:
        test_orientation()
        return
    

if __name__ == "__main__":
    app()