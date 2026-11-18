#!/usr/bin/env bash
# launcher_angle_per_vertex.sh  —  single-pair wrapper for angle_per_vertex.py
#
# Usage:
#   bash launcher_angle_per_vertex.sh              # default set: sub-02, ses 01-06 except 05
#   bash launcher_angle_per_vertex.sh -s 02,03     # one sub,ses pair
#   bash launcher_angle_per_vertex.sh -f list.tsv  # batch from a sub<TAB>ses file

set -euo pipefail

# ── configuration ────────────────────────────────────────────────────────────
REPO_DIR="/bcbl/home/home_n-z/tlei/soft/fmri-angle-metric"
PY_CMD="python ${REPO_DIR}/src/fmri_angle_metric/angle_per_vertex.py"
# If the package is pip-installed, this works instead:
# PY_CMD="angle-per-vertex"

GLM_DIR="/bcbl/home/public/Gari/VOTCLOC/main_exp/derivatives/l1_surface/analysis-final_v3_basic_MC"
OUT_DIR="/bcbl/home/public/Gari/VOTCLOC/main_exp/derivatives/angle_calculation/analysis-not_disgard_both_neg"
LOG_DIR="${OUT_DIR}/logs"

TASK="fLoc"
SPACE="fsnative"
PAIRS="RW:CS,RW:FF,RW:Faces,RW:Limbs,Faces:Limbs"
VS_SUFFIX="vsNull"
MIN_MAGNITUDE="0.1"
HEMIS="L,R"

# Default target set: sub-02, sessions 01-06
DEFAULT_SUB="02"
DEFAULT_SESSIONS=(01 02 03 04 05 06)

mkdir -p "${LOG_DIR}"

# ── runner ───────────────────────────────────────────────────────────────────
run_one() {
    local SUB=$1 SES=$2
    local LOG_OUT="${LOG_DIR}/sub-${SUB}_ses-${SES}.out"
    local LOG_ERR="${LOG_DIR}/sub-${SUB}_ses-${SES}.err"
    local EXIT_CODE=0

    echo "→ sub-${SUB} ses-${SES}"
    {
        echo "Start : $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        time ${PY_CMD} \
            --glm-dir "${GLM_DIR}" \
            --sub "${SUB}" \
            --ses "${SES}" \
            --task "${TASK}" \
            --space "${SPACE}" \
            --pairs "${PAIRS}" \
            --vs-suffix "${VS_SUFFIX}" \
            --min-magnitude "${MIN_MAGNITUDE}" \
            --hemis "${HEMIS}" \
            --out-dir "${OUT_DIR}" || EXIT_CODE=$?
        echo ""
        echo "End   : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Exit  : ${EXIT_CODE}"
    } >"${LOG_OUT}" 2>"${LOG_ERR}"

    if [[ ${EXIT_CODE} -ne 0 ]]; then
        echo "  FAILED (exit ${EXIT_CODE}) — see ${LOG_ERR}" >&2
    fi
    return 0
}

# ── argument parsing ─────────────────────────────────────────────────────────
MODE="default"
SUB=""
SES=""
SUBSES_FILE=""

while getopts "s:f:" opt; do
    case $opt in
        s) MODE="single"; IFS=',' read -r SUB SES <<< "${OPTARG}" ;;
        f) MODE="batch";  SUBSES_FILE="${OPTARG}" ;;
        *) echo "Usage: $0 [-s <sub>,<ses>] [-f <subses_file.tsv>]" >&2; exit 1 ;;
    esac
done

case "${MODE}" in
    single)
        run_one "${SUB}" "${SES}"
        ;;
    batch)
        while IFS=$'\t' read -r SUB SES _rest; do
            [[ -z "${SUB:-}" ]] && continue
            [[ "${SUB}" == sub* ]] && SUB="${SUB#sub-}"
            [[ "${SES}" == ses* ]] && SES="${SES#ses-}"
            run_one "${SUB}" "${SES}"
        done < "${SUBSES_FILE}"
        ;;
    default)
        for SES in "${DEFAULT_SESSIONS[@]}"; do
            run_one "${DEFAULT_SUB}" "${SES}"
        done
        ;;
esac

echo "All done. Logs in ${LOG_DIR}"
