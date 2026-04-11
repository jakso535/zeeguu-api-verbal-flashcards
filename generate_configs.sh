#!/bin/bash

# Script that takes all the configuration info from .env and create the api.cfg and fmd.cfg
# configuration files that are required for the API

# Remove comments (lines starting with #) and export variables from .env
set -a  # Automatically export all variables
source <(grep -v '^#' .env)  # Only read lines that do not start with #
set +a

# Generate the config files using envsubst
envsubst < default.api.cfg > api.cfg
envsubst < default.fmd.cfg > fmd.cfg