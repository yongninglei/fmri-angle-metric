#!/usr/bin/env bash
# launcher_all.sh  —  run the whole chain for every subject in a subseslist
#
#   stage 1  angle_per_vertex        (one call per subject x session)
#   stage 2  pca_response_profile    (one call per subject x session)
#   stage 3  the three cross-session summaries (one call per subject)
#
# The subject/session grid comes from a subseslist CSV (sub,ses,RUN); only
# rows with RUN=True are run, so sessions that were never acquired are skipped
# rather than failing later.
#
# Usage:
#   bash launcher_all.sh                          # every RUN=True row, minus -e
#   bash launcher_all.sh -e 02,03                 # exclude subjects
#   bash launcher_all.sh -s 04                    # only these subjects
#   bash launcher_all.sh -S 01,02,03              # force this session list
#   bash launcher_all.sh -x summary               # only re-run the summaries
#   bash launcher_all.sh -l /path/to/other.txt    # a different subseslist
#   bash launcher_all.sh -n                       # dry run: print the grid, do nothing
#
# Requires the env that has nibabel/nilearn/matplotlib (e.g. micromamba lc).
# Override the interpreter with:  PY_BIN=/path/to/python bash launcher_all.sh

set -euo pipefail

# ── configuration ────────────────────────────────────────────────────────────
REPO_DIR="/bcbl/home/home_n-z/tlei/soft/fmri-angle-metric"
SRC="${REPO_DIR}/src/fmri_angle_metric"
PY_BIN="${PY_BIN:-python}"

MAIN="/bcbl/home/public/Gari/VOTCLOC/main_exp"
GLM_DIR="${MAIN}/derivatives/l1_surface/analysis-final_v3_basic_MC"
ANGLE_OUT="${MAIN}/derivatives/angle_calculation/analysis-not_disgard_both_neg"
PCA_OUT="${MAIN}/derivatives/angle_calculation/analysis-pca"
FS_DIR="${MAIN}/derivatives/freesurfer-with_t2"
LABEL_SUBDIR="label/manual_label_clusters_analysis_27_v3"

SUBSES_FILE="/export/home/tlei/tlei/soft/launchcontainers/launchcontainers/tests/ref_107ses.txt"

TASK="fLoc"
SPACE="fsnative"
HEMIS="L,R"
PAIRS="RW:CS,RW:FF,RW:Faces,RW:Limbs,Faces:Limbs"
CONDITIONS="RW,FF,CS,PER,Faces,Limbs"
VS_SUFFIX="vsNull"
MIN_MAGNITUDE="0.1"
N_PCS="5"
NORMALIZE="zscore"
N_WORKERS="4"

# sub-02 is already done, so it is excluded by default.
EXCLUDE="02"
ONLY_SUBS=""
FORCE_SESSIONS=""
STAGES="angle,pca,summary"
DRY_RUN=0

# ── argument parsing ─────────────────────────────────────────────────────────
while getopts "l:e:s:S:x:n" opt; do
    case $opt in
        l) SUBSES_FILE="${OPTARG}" ;;
        e) EXCLUDE="${OPTARG}" ;;
        s) ONLY_SUBS="${OPTARG}" ;;
        S) FORCE_SESSIONS="${OPTARG}" ;;
        x) STAGES="${OPTARG}" ;;
        n) DRY_RUN=1 ;;
        *) echo "Usage: $0 [-l subseslist] [-e excl_subs] [-s only_subs]" \
                "[-S sessions] [-x stages] [-n]" >&2; exit 1 ;;
    esac
done

has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }
in_csv()    { [[ ",$2," == *",$1,"* ]]; }

if [[ ! -f "${SUBSES_FILE}" ]]; then
    echo "ERROR: subseslist not found: ${SUBSES_FILE}" >&2
    exit 1
fi

# ── build the subject -> sessions grid ───────────────────────────────────────
# Only RUN=True rows. Order is preserved as it appears in the file.
declare -A SES_OF
SUBS=()

while IFS=, read -r sub ses run _rest; do
    sub="${sub//[[:space:]]/}"; ses="${ses//[[:space:]]/}"; run="${run//[[:space:]]/}"
    [[ "${sub}" == "sub" || -z "${sub}" ]] && continue          # header / blank
    [[ "${run}" =~ ^([Tt]rue|1)$ ]] || continue                 # RUN=False
    sub="${sub#sub-}"; ses="${ses#ses-}"

    in_csv "${sub}" "${EXCLUDE}" && continue
    [[ -n "${ONLY_SUBS}" ]] && ! in_csv "${sub}" "${ONLY_SUBS}" && continue

    if [[ -z "${SES_OF[${sub}]:-}" ]]; then
        SUBS+=("${sub}")
        SES_OF[${sub}]="${ses}"
    else
        SES_OF[${sub}]="${SES_OF[${sub}]},${ses}"
    fi
done < "${SUBSES_FILE}"

if [[ ${#SUBS[@]} -eq 0 ]]; then
    echo "ERROR: no subjects left after filtering ${SUBSES_FILE}" >&2
    exit 1
fi

# An explicit -S overrides whatever the list said, for every subject.
if [[ -n "${FORCE_SESSIONS}" ]]; then
    for s in "${SUBS[@]}"; do SES_OF[${s}]="${FORCE_SESSIONS}"; done
fi

# Keep only sessions that actually have PSC maps on disk, so a listed-but-
# missing session is reported here instead of failing mid-run.
TOTAL=0
for sub in "${SUBS[@]}"; do
    kept=""; dropped=""
    IFS=',' read -r -a want <<< "${SES_OF[${sub}]}"
    for ses in "${want[@]}"; do
        d="${GLM_DIR}/sub-${sub}/ses-${ses}"
        if compgen -G "${d}/*stat-PSC*.func.gii" >/dev/null 2>&1; then
            kept="${kept:+${kept},}${ses}"
        else
            dropped="${dropped:+${dropped},}${ses}"
        fi
    done
    SES_OF[${sub}]="${kept}"
    [[ -n "${dropped}" ]] && echo "  note: sub-${sub} has no PSC for ses-${dropped}"
    n=0; [[ -n "${kept}" ]] && n=$(awk -F, '{print NF}' <<< "${kept}")
    TOTAL=$((TOTAL + n))
done

echo "=============================================================="
echo " subseslist : ${SUBSES_FILE}"
echo " subjects   : ${#SUBS[@]}  (${SUBS[*]})"
echo " excluded   : ${EXCLUDE:-none}"
echo " stages     : ${STAGES}"
echo " python     : ${PY_BIN}"
echo ""
for sub in "${SUBS[@]}"; do
    printf " sub-%-4s %s\n" "${sub}" "${SES_OF[${sub}]:-<none>}"
done
echo ""
echo " total subject x session runs per stage: ${TOTAL}"
echo "=============================================================="

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "dry run — nothing executed"
    exit 0
fi

FAILED=0
declare -A FAIL_OF

# run_step <logdir> <logname> <command...>
run_step() {
    local logdir=$1 name=$2; shift 2
    local out="${logdir}/${name}.out"
    local err="${logdir}/${name}.err"
    local code=0

    printf '  %-34s ' "${name}"
    {
        echo "Start : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Cmd   : $*"
        echo ""
        time "$@" || code=$?
        echo ""
        echo "End   : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Exit  : ${code}"
    } >"${out}" 2>"${err}"

    if [[ ${code} -eq 0 ]]; then
        echo "ok"
    else
        echo "FAILED (exit ${code}) -> ${err}"
        FAILED=$((FAILED + 1))
        FAIL_OF[${CUR_SUB}]=$(( ${FAIL_OF[${CUR_SUB}]:-0} + 1 ))
    fi
    return 0
}

# ── main loop ────────────────────────────────────────────────────────────────
for CUR_SUB in "${SUBS[@]}"; do
    SES_CSV="${SES_OF[${CUR_SUB}]}"
    if [[ -z "${SES_CSV}" ]]; then
        echo ""
        echo "### sub-${CUR_SUB} — no usable sessions, skipping"
        continue
    fi
    IFS=',' read -r -a SES_LIST <<< "${SES_CSV}"

    LOG_DIR="${MAIN}/derivatives/angle_calculation/logs/sub-${CUR_SUB}"
    mkdir -p "${LOG_DIR}"

    echo ""
    echo "### sub-${CUR_SUB}  (${#SES_LIST[@]} sessions: ${SES_CSV})"

    if has_stage angle; then
        for SES in "${SES_LIST[@]}"; do
            run_step "${LOG_DIR}" "angle_sub-${CUR_SUB}_ses-${SES}" \
                "${PY_BIN}" "${SRC}/angle_per_vertex.py" \
                    --glm-dir "${GLM_DIR}" \
                    --sub "${CUR_SUB}" --ses "${SES}" \
                    --task "${TASK}" --space "${SPACE}" \
                    --pairs "${PAIRS}" --vs-suffix "${VS_SUFFIX}" \
                    --min-magnitude "${MIN_MAGNITUDE}" --hemis "${HEMIS}" \
                    --out-dir "${ANGLE_OUT}"
        done
    fi

    if has_stage pca; then
        for SES in "${SES_LIST[@]}"; do
            run_step "${LOG_DIR}" "pca_sub-${CUR_SUB}_ses-${SES}" \
                "${PY_BIN}" "${SRC}/pca_response_profile.py" \
                    --glm-dir "${GLM_DIR}" \
                    --sub "${CUR_SUB}" --ses "${SES}" \
                    --task "${TASK}" --space "${SPACE}" \
                    --conditions "${CONDITIONS}" --vs-suffix "${VS_SUFFIX}" \
                    --n-pcs "${N_PCS}" --normalize "${NORMALIZE}" \
                    --min-magnitude "${MIN_MAGNITUDE}" --hemis "${HEMIS}" \
                    --out-dir "${PCA_OUT}"
        done
    fi

    if has_stage summary; then
        run_step "${LOG_DIR}" "summary_angle_sub-${CUR_SUB}" \
            "${PY_BIN}" "${SRC}/angle_cross_session.py" \
                --angle-dir "${ANGLE_OUT}" \
                --sub "${CUR_SUB}" --sessions "${SES_CSV}" \
                --pairs "${PAIRS}" --task "${TASK}" --space "${SPACE}" \
                --hemis "${HEMIS}" --n-workers "${N_WORKERS}" \
                --out-dir "${ANGLE_OUT}/summary"

        run_step "${LOG_DIR}" "summary_pca_sub-${CUR_SUB}" \
            "${PY_BIN}" "${SRC}/pca_cross_session.py" \
                --pca-dir "${PCA_OUT}" \
                --sub "${CUR_SUB}" --sessions "${SES_CSV}" \
                --task "${TASK}" --space "${SPACE}" --normalize "${NORMALIZE}" \
                --n-pcs 2 --n-workers "${N_WORKERS}" \
                --out-dir "${PCA_OUT}/summary"

        LABEL_DIR="${FS_DIR}/sub-${CUR_SUB}/${LABEL_SUBDIR}"
        if [[ -d "${LABEL_DIR}" ]]; then
            run_step "${LOG_DIR}" "summary_roi_sub-${CUR_SUB}" \
                "${PY_BIN}" "${SRC}/angle_roi_cross_session.py" \
                    --angle-dir "${ANGLE_OUT}" --label-dir "${LABEL_DIR}" \
                    --sub "${CUR_SUB}" --sessions "${SES_CSV}" \
                    --pairs "${PAIRS}" --task "${TASK}" --space "${SPACE}" \
                    --hemi L --n-workers "${N_WORKERS}" \
                    --out-dir "${ANGLE_OUT}/roi_summary"
        else
            echo "  ROI summary skipped: no ${LABEL_DIR}"
        fi
    fi
done

# ── report ───────────────────────────────────────────────────────────────────
echo ""
echo "=============================================================="
if [[ ${FAILED} -eq 0 ]]; then
    echo " all steps ok across ${#SUBS[@]} subjects"
else
    echo " ${FAILED} step(s) FAILED:"
    for sub in "${SUBS[@]}"; do
        [[ -n "${FAIL_OF[${sub}]:-}" ]] && \
            echo "   sub-${sub}: ${FAIL_OF[${sub}]} — see ${MAIN}/derivatives/angle_calculation/logs/sub-${sub}/*.err"
    done
fi
echo ""
echo " figures under:"
echo "   ${ANGLE_OUT}/summary"
echo "   ${PCA_OUT}/summary"
echo "   ${ANGLE_OUT}/roi_summary"
echo "=============================================================="

exit $(( FAILED > 0 ? 1 : 0 ))
