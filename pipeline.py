import argparse

from clearml import Task, PipelineController

PROJECT_NAME = "LLM pipeline test"
PIPELINE_NAME = "IMDB Sentiment Analysis Pipeline"

# Step templates track the branch head (no pinned commit), so they only need
# to be created once; code changes on the branch are picked up automatically.
REPO_URL = "https://github.com/ccasert/clearml-transformers.git"
REPO_BRANCH = "main"
TEMPLATE_TAG = "pipeline-template"

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
parser.add_argument("--queue", default="muller")
args = parser.parse_args()


def get_or_create_template(name, **kwargs):
    """Reuse an existing step template so each launch only creates the
    controller and step tasks."""
    found = Task.get_tasks(
        project_name=PROJECT_NAME,
        task_name=f"^{name}$",
        tags=[TEMPLATE_TAG],
        task_filter={"status": ["created"]},
    )
    if found:
        return found[0]
    task = Task.create(
        project_name=PROJECT_NAME, task_name=name,
        repo=REPO_URL, branch=REPO_BRANCH,
        docker=DOCKER_IMAGE, docker_bash_setup_script=CONTAINER_SETUP_SCRIPT,
        add_task_init_call=False, **kwargs
    )
    task.add_tags([TEMPLATE_TAG])
    return task


pipe = PipelineController(
    name=PIPELINE_NAME,
    project=PROJECT_NAME,
    version="1.2",
)

pipe.set_default_execution_queue(args.queue)

pipe.add_parameter(
    name="raw_dataset_id",
    description="The ClearML Dataset ID of the raw IMDB data.",
    default="f38bce9da2994eeb91e37b850a4049ab"
)
pipe.add_parameter(
    name="base_model",
    description="The base Hugging Face model to use for tokenizing and training.",
    default="distilbert-base-uncased"
)

base_preprocess_task = get_or_create_template(
    "Template: preprocess", script="preprocess.py")
base_train_task = get_or_create_template(
    "Template: train", script="train.py", binary="accelerate launch")
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
    cache_executed_step=True,
)

train_step_name = "step_train"
pipe.add_step(
    name=train_step_name,
    parents=[preprocess_step_name],
    base_task_id=base_train_task.id,
    parameter_override={
        "Args/processed_dataset_id": "${step_preprocess.parameters.General/processed_dataset_id}",
        "Args/base_model": "${pipeline.base_model}"
    },
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
)


pipe.start_locally()
