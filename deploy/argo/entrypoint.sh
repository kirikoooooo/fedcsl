#!/usr/bin/env bash
# Container entrypoint for Argo stages.
#
# The image keeps code at /opt/fedcsl while mutable state lives on the PVC at
# /data. FedCSL's training code uses relative paths, so expose the PVC
# directories as symlinks under the code root and then exec the requested
# command (usually `python -m observability.stage ...`).
set -euo pipefail

CODE_DIR="${FEDCSL_CODE_DIR:-/opt/fedcsl}"
DATA_DIR="${FEDCSL_DATA_DIR:-/data}"

cd "${CODE_DIR}"

mkdir -p \
  "${DATA_DIR}/Epilepsy" \
  "${DATA_DIR}/Multivariate_ts" \
  "${DATA_DIR}/HAR" \
  "${DATA_DIR}/UCI HAR Dataset" \
  "${DATA_DIR}/algoutils" \
  "${DATA_DIR}/checkpoint" \
  "${DATA_DIR}/result" \
  "${DATA_DIR}/data" \
  "${DATA_DIR}/observability/runs"

for dir in Epilepsy Multivariate_ts HAR "UCI HAR Dataset" algoutils checkpoint result data; do
  if [ -e "${CODE_DIR}/${dir}" ] && [ ! -L "${CODE_DIR}/${dir}" ]; then
    rm -rf "${CODE_DIR:?}/${dir}"
  fi
  ln -sfn "${DATA_DIR}/${dir}" "${CODE_DIR}/${dir}"
done

mkdir -p "${CODE_DIR}/observability"
if [ -e "${CODE_DIR}/observability/runs" ] && [ ! -L "${CODE_DIR}/observability/runs" ]; then
  rm -rf "${CODE_DIR}/observability/runs"
fi
ln -sfn "${DATA_DIR}/observability/runs" "${CODE_DIR}/observability/runs"

exec "$@"
