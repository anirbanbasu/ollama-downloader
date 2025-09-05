# Work-in-progress script to filter pstats data to only include relevant functions.
import os
import pstats
import re
import subprocess
from rich import print as print


def filter_pstats_and_save_svg(
    input_file: str, output_file: str, keep_regexes: list[str]
):
    """
    Loads pstats data, filters it, and saves a new pstats file.
    `keep_regexes` is a list of regex patterns to match against function names.
    """
    stats = pstats.Stats(input_file)
    # FIXME: Get the raw dictionary of stats -- this is very hacky because pstats doesn't expose this directly.
    original_stats = stats.stats  # type: ignore[attr-defined]

    filtered_stats = {}

    # Compile the regexes for efficiency
    compiled_regexes = [re.compile(r) for r in keep_regexes]
    print(f"Compiled regexes: {len(compiled_regexes)}")

    # Find all functions that match your regexes
    to_keep = set()
    for func, (cc, nc, tt, ct, callers) in original_stats.items():
        # func is a tuple like ('my_package/my_module.py', 10, 'my_function')
        func_name = str(func)
        if any(regex.search(func_name) for regex in compiled_regexes):
            to_keep.add(func)

    # Include all dependencies of the kept functions
    all_needed = set(to_keep)
    queue = list(to_keep)
    print("Matching functions:")
    print(queue)

    while queue:
        func = queue.pop(0)
        (cc, nc, tt, ct, callers) = original_stats.get(func, (0, 0, 0, 0, {}))
        for caller_func in callers.keys():
            if caller_func not in all_needed:
                all_needed.add(caller_func)
                queue.append(caller_func)

    # Build the new stats dictionary, including only the functions we need
    for func in all_needed:
        if func in original_stats:
            filtered_stats[func] = original_stats[func]

    stats.stats = filtered_stats  # type: ignore[attr-defined]
    stats.dump_stats(output_file)


def main():
    # Define the patterns to keep. This should capture your tests and application code.
    patterns_to_keep = [
        # Your specific test files or modules
        "tests/",
        "src/ollama_downloader",
        # Any other modules you want to include
        # "src/other_module"
    ]

    # Filter and save the new profile data
    PROF_DIR = "prof"
    TARGET_PROF = "combined.prof"
    OUTPUT_PROF = "filtered.prof"
    OUTPUT_SVG = "filtered_profile.svg"
    target_path = os.path.join(PROF_DIR, TARGET_PROF)
    try:
        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"Profile data file {target_path} not found. Maybe run the tests with profiling first?"
            )
        output_path = os.path.join(PROF_DIR, OUTPUT_PROF)
        output_svg_path = os.path.join(PROF_DIR, OUTPUT_SVG)
        filter_pstats_and_save_svg(target_path, output_path, patterns_to_keep)
        print(f"Filtered profile data saved to {output_path}.")

        conversion_command = (
            f"gprof2dot -f pstats {output_path} | dot -Tsvg -o {output_svg_path}"
        )
        result = subprocess.run(
            conversion_command, capture_output=True, shell=True, check=True
        )
        print(f"{result.args} completed with return code {result.returncode}.")
        if result.returncode != 0:
            print(f"stderr: {result.stderr.decode()}")
            print(f"stdout: {result.stdout.decode()}")
    except Exception as e:
        print(f"Error during filtering and conversion of profile data. {e}")


if __name__ == "__main__":
    main()
