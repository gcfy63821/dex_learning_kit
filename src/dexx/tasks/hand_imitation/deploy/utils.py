# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for deploy environment."""

import torch


def dof_isaaclab2sharpa(dof_pos):
    """Convert joint order from IsaacLab to Sharpa.
    
    Args:
        dof_pos: Joint positions in IsaacLab order (22 joints)
        
    Returns:
        Joint positions in Sharpa order (22 joints)
    """
    return dof_pos[[4, 9, 14, 19, 21, 0, 5, 10, 15, 1, 6, 11, 16, 3, 8, 13, 18, 2, 7, 12, 17, 20]]


def dof_sharpa2isaaclab(dof_pos):
    """Convert joint order from Sharpa to IsaacLab.
    
    Args:
        dof_pos: Joint positions in Sharpa order (22 joints)
        
    Returns:
        Joint positions in IsaacLab order (22 joints)
    """
    return dof_pos[[5, 9, 17, 13, 0, 6, 10, 18, 14, 1, 7, 11, 19, 15, 2, 8, 12, 20, 16, 3, 21, 4]]

