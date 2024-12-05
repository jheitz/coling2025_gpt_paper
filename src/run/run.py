import os, argparse, time, random, string, sys, yaml, subprocess, shutil, datetime, pytest
sys.path.append(sys.path[0] + '/..')  # to make the import from parent dir util work

from util.helpers import create_directory
from config.constants import Constants

constants = Constants()

arg_parser = argparse.ArgumentParser(description="Read in configuration")
arg_parser.add_argument("--config", help="config file", required=True)
arg_parser.add_argument("--results_base_dir", help="base directory to write results to")
arg_parser.add_argument("--temporal", help="temporal [1,5]")
arg_parser.add_argument("--runname", help="optional runname argument")

args = arg_parser.parse_args()


# run any pytests here
pytest.main(["-x", "../test"])

# RUN_PARAMETERS: dictionary containing flags, can later be used to add further run information

# back to /git directory, because the config file are in git/configs/
f = open(os.path.join(os.getcwd(), "..", args.config), 'r')
RUN_PARAMETERS = yaml.load(f, Loader=yaml.FullLoader)

try:
    MODEL = RUN_PARAMETERS['model']
except KeyError:
    MODEL = None

try:
    CONFIG_NAME = RUN_PARAMETERS['name']
except KeyError:
    CONFIG_NAME = None

if args.runname is not None:
    RUN_NAME = args.runname
    print("Using RUN_NAME " + RUN_NAME)
else:
    RUN_NAME = ""

if args.results_base_dir is not None:
    RESULTS_BASE_DIR = args.results_base_dir  # + "_" + time.strftime("%Y-%m-%d_%H%M")
else:
    RESULTS_BASE_DIR = ""

results_subdir = time.strftime("%Y%m%d_%H%M") + "_" + RUN_NAME + '_' + ''.join(random.choices(string.ascii_lowercase, k=4))

# RESULTS_DIR = os.path.join(os.getcwd(), "../../results", RESULTS_BASE_DIR, results_subdir)
assert os.path.exists(constants.RESULTS_ROOT), \
    f"Root directory for results ({constants.RESULTS_ROOT}) does not exist. Are you connected to methlab?"
year = datetime.datetime.today().isocalendar().year
calendar_week = datetime.datetime.today().isocalendar().week
year_week = f"{year}_kw{calendar_week}"
RESULTS_DIR = os.path.join(constants.RESULTS_ROOT, "runs", year_week, RESULTS_BASE_DIR, results_subdir)
create_directory(RESULTS_DIR)
stdout_file = os.path.join(RESULTS_DIR, "stdout.txt")
errorlog_file = os.path.join(RESULTS_DIR, "errorlog.txt")

def export_conda():
    """
    Export current conda environment to file, so it can be persisted in git
    """
    os.system("conda env export > '../../conda/environment_science_cloud.yaml'")

export_conda()

def persist_current_code():
    """
    commit and push current state of code to constants.EXPERIMENT_BRANCH in git
    """
    def git_command(command):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error executing Git command: {result.stderr}")
            raise Exception()
        return result.stdout.strip()

    # Change to EXPERIMENT_BRANCH branch, commit, and push
    git_command(["git", "checkout", constants.EXPERIMENT_BRANCH])
    git_command(["git", "add", "../.."])

    # check if changes to commit
    status = git_command(["git", "status", "--porcelain"])
    if status:
        # Commit changes and push
        git_command(["git", "commit", "-m", f"Run {os.path.join(RESULTS_BASE_DIR, results_subdir)}"])
        git_command(["git", "push"])

    # Get the commit hash of the latest commit
    commit_hash = git_command(["git", "rev-parse", "HEAD"])

    # Write the commit hash to a file
    with open(os.path.join(RESULTS_DIR, "git_commit_hash.txt"), "w") as file:
        file.write(commit_hash)

try:
    persist_current_code()
except Exception as e:
    print(f"An exception occurred persisting the current code to git:\n\nException message: {str(e)}")
    print("Continuing without pushing to git...")


def copy_config_file():
    config_file = os.path.join("..", args.config)
    assert os.path.isfile(config_file), f"Config file does not exist: {config_file}"
    config_file_name = os.path.basename(config_file)
    destination_path = os.path.join(RESULTS_DIR, config_file_name)
    shutil.copy2(config_file, destination_path)

copy_config_file()

# prepare command line arguments to pass to python script
arguments = args.__dict__
python_commandline_arguments = ' '.join(["--{} {}".format(k, arguments[k]) for k in arguments if arguments[k] is not None])

command = f"python -u main.py --results_dir {RESULTS_DIR} {python_commandline_arguments} 2> {errorlog_file} > {stdout_file}"
print(f"Running command {command}")
os.system('cd .. && ' + command)