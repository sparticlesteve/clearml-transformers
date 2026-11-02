import argparse
import subprocess

from clearml import Task, PipelineController
from clearml.backend_interface.util import get_or_create_project

PIPELINE_NAME = "Sentiment Analysis Pipeline"
TAGS = ["sc26-test", "llm-pipeline"]

# Step templates track a branch; each launch pins the steps to the branch's
# current head commit (see get_head_commit), so templates only need to be
# created once and code changes on the branch are picked up automatically.
# The train step runs as plain python: the Slurm agent ignores other binaries
# (e.g. "accelerate launch"). srun starts one process per GPU and
# CONTAINER_SETUP_SCRIPT maps the Slurm ranks to torch.distributed variables.
REPO_URL = "https://github.com/sparticlesteve/clearml-transformers.git"

# clearml is pinned: newer releases upgrade numpy to 2.x, which breaks the
# image's torch build.
CONTAINER_SETUP_SCRIPT = """
pip install clearml==2.0.2
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
"""
DOCKER_IMAGE = "nersc/pytorch:24.08.01"

parser = argparse.ArgumentParser()
parser.add_argument("--project", default="SC26 demo-dev/LLM pipeline")
parser.add_argument("--queue", default="muller")
parser.add_argument("--branch", default="sc26-demo")
parser.add_argument("--raw-dataset-id", default="cf56cd3cb38a47f0b91d68cf5f034161",
                    help="ClearML Dataset ID of the raw data (default: IMDB)")
parser.add_argument("--base-model", default="distilbert-base-uncased")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--learning-rate", type=float, default=1e-5)
parser.add_argument("--train-batch-size", type=int, default=16,
                    help="Per-device training batch size")
args = parser.parse_args()


def get_head_commit(repo, branch):
    """Resolve the branch head now, so each step task records the exact
    commit it ran."""
    out = subprocess.run(["git", "ls-remote", repo, f"refs/heads/{branch}"],
                         check=True, capture_output=True, text=True).stdout
    if not out:
        raise RuntimeError(f"Branch {branch} not found in {repo}")
    return out.split()[0]


def get_or_create_template(name, **kwargs):
    """Reuse an existing step template so each launch only creates the
    controller and step tasks. Steps inherit the template's tags."""
    found = Task.get_tasks(
        project_name=template_project,
        task_name=f"^{name}$",
        task_filter={"status": ["created"], "search_hidden": True, "_allow_extra_fields_": True},
    )
    if found:
        return found[0]
    task = Task.create(
        project_name=template_project, task_name=name,
        repo=REPO_URL, branch=args.branch,
        docker=DOCKER_IMAGE, docker_bash_setup_script=CONTAINER_SETUP_SCRIPT,
        add_task_init_call=False, **kwargs
    )
    task.add_tags(TAGS)
    return task


# Templates go in a hidden subproject, like the pipeline's .pipelines and
# .datasets; a leading dot alone doesn't hide a project, the system tag does.
template_project = f"{args.project}/.templates"
get_or_create_project(Task._get_default_session(), template_project, system_tags=["hidden"])

commit = get_head_commit(REPO_URL, args.branch)
print(f"Pinning steps to {args.branch} @ {commit[:8]}")
step_overrides = {"script.branch": args.branch, "script.version_num": commit}

pipe = PipelineController(
    name=PIPELINE_NAME,
    project=args.project,
    version="1.3",
    # Steps go in the project itself, not the pipeline's hidden subproject
    target_project=args.project,
)
pipe.add_tags(TAGS)

pipe.set_default_execution_queue(args.queue)

pipe.add_parameter(
    name="raw_dataset_id",
    description="ClearML Dataset ID of the raw data (train.csv/test.csv with text and label columns).",
    default=args.raw_dataset_id,
)
pipe.add_parameter(
    name="base_model",
    description="The base Hugging Face model to use for tokenizing and training.",
    default=args.base_model,
)
pipe.add_parameter(name="epochs", description="Training epochs.", default=args.epochs)
pipe.add_parameter(name="learning_rate", description="Training learning rate.", default=args.learning_rate)
pipe.add_parameter(name="train_batch_size", description="Per-device training batch size.", default=args.train_batch_size)

base_preprocess_task = get_or_create_template(
    "Template: preprocess", script="preprocess.py")
base_train_task = get_or_create_template(
    "Template: train", script="train.py")
base_train_task.set_user_properties(
    num_nodes=2,
    ntasks_per_node=4,
    cpus_per_task=32,
)
base_eval_task = get_or_create_template(
    "Template: evaluate", script="eval.py")

preprocess_step_name = "step_preprocess"
pipe.add_step(
    name=preprocess_step_name,
    base_task_id=base_preprocess_task.id,
    parameter_override={
        "Args/raw_dataset_id": "${pipeline.raw_dataset_id}",
        "Args/base_model": "${pipeline.base_model}"
    },
    task_overrides=step_overrides,
    cache_executed_step=True,
)

train_step_name = "step_train"
pipe.add_step(
    name=train_step_name,
    parents=[preprocess_step_name],
    base_task_id=base_train_task.id,
    parameter_override={
        "Args/processed_dataset_id": "${step_preprocess.parameters.General/processed_dataset_id}",
        "Args/base_model": "${pipeline.base_model}",
        "Args/epochs": "${pipeline.epochs}",
        "Args/learning_rate": "${pipeline.learning_rate}",
        "Args/train_batch_size": "${pipeline.train_batch_size}",
    },
    task_overrides=step_overrides,
)


eval_step_name = "step_evaluate"
pipe.add_step(
    name=eval_step_name,
    parents=[train_step_name],
    base_task_id=base_eval_task.id,
    parameter_override={
        "Args/processed_dataset_id": "${step_preprocess.parameters.General/processed_dataset_id}",
        "Args/model_task_id": "${step_train.id}"
    },
    task_overrides=step_overrides,
)


pipe.start_locally()
