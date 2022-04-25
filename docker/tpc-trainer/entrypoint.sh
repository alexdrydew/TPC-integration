#!/bin/bash

source build/config.sh
export DISPLAY=localhost:0
exec "$@"