#!/usr/bin/env bash

# Run the standard ESPnet ASR pipeline from the SAP ASR3 recipe directory.
# Keeping this small wrapper separate from asr3.sh makes the online E2E path
# independent of the existing offline token-extraction path.
set -e
set -u
set -o pipefail

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${recipe_dir}"

exec ../../TEMPLATE/asr1/asr.sh "$@"
