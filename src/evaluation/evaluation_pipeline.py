from inference import main as inference
from evaluation_metrics import main as get_metrics
from gather_results import main as gather_results
import argparse


def main(dataset_handler, model_id, device, output_path, bootstrap, end_index):
    if device == "None" or device == "none":
        device = None
    if "100" in dataset_handler:
        end_index = min(100, end_index)
    elif "500" in dataset_handler:
        end_index = min(500, end_index)

    inference_results, _ = inference(
        dataset_handler=dataset_handler,
        model_id=model_id,
        device=device,
        dtype="bf16",
        save_interval=5,  # Save progress every 100 images
        resume=True,  # Whether to resume from the last saved state
        end_index=end_index,
    )
    metric_per_instance = get_metrics(
        model_id=model_id,
        eval_path=None,
        output_path=output_path,
        json_filename=None,
        dataset_handler=dataset_handler,
        inference_results=inference_results,
        end_index=end_index,
    )
    model_name = model_id.replace("/", "-")
    gather_results(
        None, bootstrap, metric_per_instance, end=end_index, model_name=model_name
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation pipeline of VCR.")
    parser.add_argument(
        "--model_id",
        type=str,
        help="model name of from huggingface model hub",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device name",
        default="cuda",
    )
    parser.add_argument("--output_path", type=str, help="folder path", default=".")
    parser.add_argument(
        "--dataset_handler",
        type=str,
        help="dataset handler",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="also calculate the bootstrap confidence interval",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        help="end index of the evaluation",
        default=5000,
    )
    args = parser.parse_args()

    main(
        args.dataset_handler,
        args.model_id,
        args.device,
        args.output_path,
        args.bootstrap,
        args.end_index,
    )
    # main(
    #     "vcr-org/VCR-wiki-en-easy-test",
    #     "openbmb/MiniCPM-Llama3-V-2_5",
    #     ".",
    #     False,
    #     5000,
    # )
