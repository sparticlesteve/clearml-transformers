import argparse
import os
import torch
import torch.distributed as dist

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from clearml import Task, Dataset

def setup_ddp():
    """Initializes the DDP process group."""
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)

    print(f"[Process {rank}] World size: {world_size}, Local rank: {local_rank}, Device: {torch.cuda.current_device()}")
    return rank, local_rank, world_size

def train_model(processed_dataset_id: str, base_model: str, epochs: int,
                learning_rate: float, train_batch_size: int, eval_batch_size: int):
    rank, local_rank, world_size = setup_ddp()

    if rank == 0:
        task = Task.init(project_name="LLM pipeline test", task_name="Model Training")
        logger = task.get_logger()
        logger.report_text(f"Step 2: Training model using processed dataset ID: {processed_dataset_id}")

    if rank != 0:
        dist.barrier()

    processed_data_path = Dataset.get(dataset_id=processed_dataset_id).get_local_copy()
    tokenized_datasets = load_from_disk(processed_data_path)

    if rank == 0:
        dist.barrier()


    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)

    training_args = TrainingArguments(
        output_dir=f"./results_rank_{rank}",
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        logging_dir=f'./logs_rank_{rank}',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="clearml",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    trainer.train()

    if rank == 0:
        logger.report_text("Training complete. Uploading final best model as 'model' artifact...")

        final_model_path = "./final_model"
        trainer.save_model(final_model_path)

        task.upload_artifact(name="model", artifact_object=final_model_path)

        logger.report_text("Artifact named 'model' uploaded successfully.")

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dataset_id", type=str, required=True, help="ID of the processed ClearML Dataset")
    parser.add_argument("--base_model", type=str, default="distilbert-base-uncased")
    # Hyperparameters are argparse args, not task.connect() values: ClearML
    # applies Args/* overrides on every rank, but connect() only on rank 0.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    args = parser.parse_args()

    train_model(args.processed_dataset_id, args.base_model, args.epochs,
                args.learning_rate, args.train_batch_size, args.eval_batch_size)