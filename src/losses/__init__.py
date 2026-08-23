# -*- coding: utf-8 -*-
"""Loss package for the milestone-1 conduction PINN."""

from .conduction import ConductionLoss
from .interface import InterfaceLoss
from .boundary import BoundaryLoss
from .energy_conservation import EnergyConservationLoss

__all__ = ["ConductionLoss", "InterfaceLoss", "BoundaryLoss",
           "EnergyConservationLoss"]
