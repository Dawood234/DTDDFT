import subprocess

method = "tddft"
# xc_functional = '''xcamb88 1.00 lyp 0.81 vwn_5 0.19 hfexch 1.00  
#  cam 0.33 cam_alpha 0.19 cam_beta 0.46'''''
# xc_functional=xc_functional[0:7]
xc_functional = "pbe96"
basis_set = "def2-SVP"
folder_path = f"{method}_{basis_set}_{xc_functional}"
geometry_file='geom.txt'
num_atoms=10
unit = 'angstrom'
run_calc= True

def run_script(script_name, **kwargs):
    cmd = ["python3", script_name]
    for key, value in kwargs.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))
    subprocess.run(cmd)


if run_calc==True:
    run_script("gen_inputFiles.py", geometry_file=geometry_file,num_atoms=num_atoms,
            folder_path=folder_path, method=method,xc_functional=xc_functional,
            basis_set=basis_set, unit=unit)

    run_script("run_nwchem.py", folder_path=folder_path, method=method)

    run_script("clean.py")

    run_script("move_files.py", method=method, basis_set=basis_set, xc_functional=xc_functional)

run_script("gs.py", method=method, folder_path=folder_path)
run_script("mo_energies.py", method=method, folder_path=folder_path)
run_script("mo17.py", method=method, folder_path=folder_path)
run_script("states.py", method=method, folder_path=folder_path)
run_script("apbv.py", folder_path=folder_path)
run_script("sqrtambv.py", folder_path=folder_path, method=method)

run_script("hq1d.py", method=method, basis_set=basis_set, xc_functional=xc_functional)
run_script("hq2d.py", method=method, basis_set=basis_set, xc_functional=xc_functional)
