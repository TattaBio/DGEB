"""
Main command to run diverse genomic embedding benchmarks (DGEB) on a model.
example command to run DGEB:
python run_dgeb.py -m facebook/esm2_t6_8M_UR50D
"""

import argparse
import logging
import os

import dgeb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_TASK_NAMES = dgeb.get_all_task_names()
ALL_MODEL_NAMES = dgeb.get_all_model_names()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help=f"Model to evaluate. Choose from {ALL_MODEL_NAMES}",
    )
    parser.add_argument(
        "-t",
        "--tasks",
        type=lambda s: [item for item in s.split(",")],
        default=None,
        help=f"Comma separated tasks to evaluate on. Choose from {ALL_TASK_NAMES} or do not specify to evaluate on all tasks",
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=str,
        default=None,
        help="Layer to evaluate. Comma separated list of integers or 'mid' and 'last'. Default is 'mid,last'",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0",
        help="Comma separated list of GPU device ids to use. Default is 0 (if GPUs are detected).",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Output directory for results. Will default to results/model_name if not set.",
    )
    parser.add_argument(
        "-v", "--verbosity", type=int, default=2, help="Verbosity level"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Maximum sequence length for model, default is 1024.",
    )
    parser.add_argument(
        "--pool_type",
        type=str,
        default="mean",
        help="Pooling type for model, choose from mean, max, cls, last. Default is mean.",
    )

    args = parser.parse_args()

    # set logging based on verbosity level
    if args.verbosity == 0:
        logging.getLogger("geb").setLevel(logging.CRITICAL)
    elif args.verbosity == 1:
        logging.getLogger("geb").setLevel(logging.WARNING)
    elif args.verbosity == 2:
        logging.getLogger("geb").setLevel(logging.INFO)
    elif args.verbosity == 3:
        logging.getLogger("geb").setLevel(logging.DEBUG)

    if args.model is None:
        raise ValueError("Please specify a model using the -m or --model argument")

    # make sure that devices are comma separated list of integers
    try:
        devices = [int(device) for device in args.devices.split(",")]
    except ValueError:
        raise ValueError("Devices must be comma separated list of integers")

    layers = args.layers
    if layers:
        if layers not in ["mid", "last"]:
            # Layers should be list of integers.
            try:
                layers = [int(layer) for layer in layers.split(",")]
            except ValueError:
                raise ValueError("Layers must be a list of integers.")

    model_name = args.model.split("/")[-1]
    output_folder = args.output_folder
    if output_folder is None:
        output_folder = os.path.join("results", model_name)
        # create output folder if it does not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        logger.info(f"Results will be saved to {output_folder}")

    # Load the model by name.
    model = dgeb.get_model(
        model_name=args.model,
        layers=layers,
        devices=devices,
        max_seq_length=args.max_seq_len,
        batch_size=args.batch_size,
        pool_type=args.pool_type,
    )

    all_tasks_for_modality = dgeb.get_tasks_by_modality(model.modality)

    if args.tasks:
        task_list = dgeb.get_tasks_by_name(args.tasks)
        if not all([task.metadata.modality == model.modality for task in task_list]):
            raise ValueError(f"Tasks must be one of {all_tasks_for_modality}")
    else:
        task_list = all_tasks_for_modality
    evaluation = dgeb.DGEB(tasks=task_list)
    _ = evaluation.run(model)


if __name__ == "__main__":
    main()
