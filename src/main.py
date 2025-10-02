#!/usr/bin/env python3

import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation import Simulation

def main():
    # Create and run the simulation
    simulation = Simulation()
    simulation.run()

if __name__ == "__main__":
    main()
