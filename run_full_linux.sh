PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$SCRIPT_DIR"
cd "$PKG_ROOT"

export PYTHONPATH="${PKG_ROOT}/src:${PKG_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_ROOT="${RUN_ROOT:-runs}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-1}"
FORCE_CHINESE_DATA="${FORCE_CHINESE_DATA:-0}"
PAPER_SAMPLES_PER_CLASS="${PAPER_SAMPLES_PER_CLASS:-5}"
PAPER_MAX_SAMPLES=$((2 * PAPER_SAMPLES_PER_CLASS))

export PYTHONUNBUFFERED=1

if [[ ! -d experiments || ! -d src ]]; then
    printf 'ERROR: incomplete repository layout under %s\n' "$PKG_ROOT" >&2
    exit 2
fi

printf 'Repository: %s\n' "$PKG_ROOT"
printf 'Run root: %s\n' "$RUN_ROOT"
printf 'Python: %s | device: %s | workers: %s\n' \
    "$PYTHON_BIN" "$DEVICE" "$WORKERS"
printf 'PySB paper validation: %s samples/class, %s samples/run\n' \
    "$PAPER_SAMPLES_PER_CLASS" "$PAPER_MAX_SAMPLES"

# 1. Prepare datasets.
"$PYTHON_BIN" -m experiments.single_layer.prepare_st003390

if [[ "$FORCE_CHINESE_DATA" == "1" ]]; then
    "$PYTHON_BIN" -m experiments.chinese_mnist.prepare_data \
        --data-root data/chinese_mnist \
        --image-size 28 \
        --force
else
    "$PYTHON_BIN" -m experiments.chinese_mnist.prepare_data \
        --data-root data/chinese_mnist \
        --image-size 28
fi

# 2. Run the complete formal and supplementary experiment grids (425 runs).
"$PYTHON_BIN" -m experiments.single_layer.run \
    --profile circles_main \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --output-root "$RUN_ROOT/single/circles" \
    --resume \
    --fail-fast

"$PYTHON_BIN" -m experiments.single_layer.run \
    --profile st003390_main \
    --st003390-csv data/st003390/st003390_m1m2.csv \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --output-root "$RUN_ROOT/single/st003390" \
    --resume \
    --fail-fast

"$PYTHON_BIN" -m experiments.single_layer.run \
    --profile supplement_main \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --output-root "$RUN_ROOT/single/supplement" \
    --resume \
    --fail-fast

"$PYTHON_BIN" -m experiments.chebyshev.run \
    --profile main \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --output-root "$RUN_ROOT/chebyshev/main" \
    --resume \
    --fail-fast

"$PYTHON_BIN" -m experiments.chebyshev.run \
    --profile d1 \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --output-root "$RUN_ROOT/chebyshev/d1" \
    --resume \
    --fail-fast

"$PYTHON_BIN" -m experiments.chinese_mnist.run \
    --profile full \
    --data-root data/chinese_mnist \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --output-root "$RUN_ROOT/chinese_mnist/full" \
    --resume \
    --fail-fast

# 3. Resolve the two fixed checkpoints used for the paper's PySB validation.
find_one_run() {
    local search_root="$1"
    local pattern="$2"
    local task_name="$3"
    local -a matches=()

    mapfile -t matches < <(
        find "$search_root" -mindepth 1 -maxdepth 1 -type d \
            -name "$pattern" -print | sort
    )

    if (( ${#matches[@]} != 1 )); then
        printf 'Expected exactly one %s run, found %s.\n' \
            "$task_name" "${#matches[@]}" >&2
        return 3
    fi

    printf '%s\n' "${matches[0]}"
}

ST_RUN="$(find_one_run \
    "$RUN_ROOT/single/st003390" \
    'st003390_m1m2_raw_d1_r00f00s000__cfg_*' \
    'ST003390')"

CHEB_M3_RUN="$(find_one_run \
    "$RUN_ROOT/chebyshev/main" \
    'm3_w16_d9_I1_C1_continuation_s000__cfg_*' \
    'Chebyshev m3')"

# 4. Reproduce the paper's PySB validation: five samples from each target
# class for ST003390 and Chebyshev m=3 (20 samples in total).
"$PYTHON_BIN" -m experiments.crn_validation.run \
    --run-dir "$ST_RUN" \
    --run-dir "$CHEB_M3_RUN" \
    --st003390-csv data/st003390/st003390_m1m2.csv \
    --output-root "$RUN_ROOT/crn_validation/paper_representatives" \
    --device cpu \
    --max-samples-per-run "$PAPER_MAX_SAMPLES" \
    --samples-per-class "$PAPER_SAMPLES_PER_CLASS" \
    --initial-t-end 20 \
    --max-t-end 320 \
    --n-timepoints 201 \
    --atol 1e-4 \
    --rtol 1e-4 \
    --tail-atol 1e-6 \
    --tail-rtol 1e-6 \
    --fail-fast

printf 'All paper experiments and validations completed: %s\n' "$RUN_ROOT"