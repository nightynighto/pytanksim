# -*- coding: utf-8 -*-
"""Initializes the logger for pytanksim."""
# This file is a part of the python package pytanksim.
#
# Copyright (c) 2024 Muhammad Irfan Maulana Kusdhany, Kyushu University
#
# pytanksim is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import logging
import sys

logger = logging.getLogger('pytanksim')
logger.setLevel(logging.DEBUG)

# create console handler
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.INFO)

# add the handlers to the logger
logger.addHandler(ch)
