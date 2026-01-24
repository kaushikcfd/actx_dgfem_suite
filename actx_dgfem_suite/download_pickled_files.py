import tempfile
import subprocess
import os


box_link = "https://uofi.box.com/shared/static/l0velh8co5yo57g099v16lm60m466777.zip"

equations = ["wave", "euler", "cns_without_chem"]
orders = [1, 2, 3, 4]
dims = [3]


def download_from_box(target_loc: str) -> None:
    subprocess.call(["wget", "-L", box_link, "--output-document", target_loc])


def unzip(zipfile_loc: str, target_dir: str) -> None:
    subprocess.call(["unzip", zipfile_loc, "-d", target_dir])


def remove_file(loc: str) -> None:
    subprocess.call(["rm", loc])


def remove_empty_dir(loc: str) -> None:
    subprocess.call(["rm", "-r", loc])


def move_file(*, src_loc: str, dst_loc: str) -> None:
    subprocess.call(["mv", src_loc, dst_loc])


def main() -> None:
    tmpdir = tempfile.mkdtemp()
    zipfile_loc = os.path.join(tmpdir, "pickled_data.zip")
    print(f"Downloading zip file to '{zipfile_loc}'...")
    download_from_box(zipfile_loc)

    unzip(zipfile_loc, tmpdir)
    assert os.path.isdir("suite")

    for eqn in equations:
        for dim in dims:
            for order in orders:
                case_dir = f"{eqn}_{dim}D_P{order}"
                literals_file_src = os.path.join(
                    tmpdir, "actx_dgfem_suite_data", case_dir, "literals.npz")
                literals_file_dst = os.path.join(
                    "suite", case_dir, "literals.npz")
                move_file(src_loc=literals_file_src, dst_loc=literals_file_dst)

                ref_outputs_file_src = os.path.join(
                    tmpdir, "actx_dgfem_suite_data", case_dir, "ref_outputs.pkl")
                ref_outputs_file_dst = os.path.join(
                    "suite", case_dir, "ref_outputs.pkl")
                move_file(src_loc=ref_outputs_file_src, dst_loc=ref_outputs_file_dst)

                ref_input_file_src = os.path.join(
                    tmpdir, "actx_dgfem_suite_data", case_dir, "ref_input_args.pkl")
                ref_input_file_dst = os.path.join(
                    "suite", case_dir, "ref_input_args.pkl")
                move_file(src_loc=ref_input_file_src, dst_loc=ref_input_file_dst)

    remove_file(zipfile_loc)
    remove_empty_dir(tmpdir)


if __name__ == "__main__":
    main()

# vim: fdm=marker
