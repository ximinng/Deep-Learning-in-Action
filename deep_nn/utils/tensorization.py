# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import torch.utils.data


def gather2featmap(consts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """ Gather consts for $t$ and reshape to feature map shape
    Args:
        consts:
        t: index
    Returns: feature map
    """
    # The index of the consts: (-1, t_0), (-1, t_1), ..., (-1, t_n)
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)
