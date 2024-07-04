import json
import glob
import numpy as np
import argparse
import csv




# get all json under a folder and return a dict with key as filename and value as json content
def read_json_files(folder_path):
    json_files = glob.glob(f"{folder_path}/*_evaluation_result.json")
    json_dict = {}
    for json_file in json_files:
        with open(json_file, "r") as f:
            json_dict[json_file] = json.load(f)
    return json_dict


def dicts2csv(list_of_dicts, name_of_dicts, output_csv_path):
    filenames = name_of_dicts
    all_data = list_of_dicts
    # Open the CSV file for writing
    with open(output_csv_path, "w", newline="") as file:
        writer = csv.writer(file)

        # First row: filenames starting from the second column
        writer.writerow([""] + filenames)

        # Extract all keys from the first JSON object (assuming all are identical)
        keys = all_data[0].keys()

        # For each key, write a row with the key and the corresponding values from each JSON file
        for key in keys:
            row = [key] + [data[key] for data in all_data]
            writer.writerow(row)
    # close writer
    file.close()
    # print the first 2 lines
    print(f"Results saved to {output_csv_path}. Below are the first 2 lines:")
    with open(output_csv_path, "r") as file:
        print(file.readline())
        print(file.readline())


def get_score(data, target_domain, target_metric_name, start, end, bootstrap=False):
    if start is None:
        start = 0
    if end is None:
        end = 1e10
    output_result = {}
    dictionary_list_detailed_results = {}
    for k, v in data.items():
        model_task_name = k.split("/")[-1].split("_evaluation_result.json")[0]
        tmp = []
        for image_id, inner_dict in v.items():
            if int(image_id) >= start and int(image_id) < end:
                for inner_key, inner_value in inner_dict.items():
                    if inner_key == target_domain:
                        for blank_id, blank_metrics in inner_value.items():
                            for metric_name, metric_value in blank_metrics.items():
                                if metric_name == target_metric_name:
                                    tmp.append(metric_value)
        output_result[model_task_name] = sum(tmp) / len(tmp)
        dictionary_list_detailed_results[model_task_name] = tmp
    if bootstrap:
        std_dict = {}
        percentile_25_dict = {}
        percentile_975_dict = {}
        for k, v in dictionary_list_detailed_results.items():
            std_dict[k], percentile_25_dict[k], percentile_975_dict[k] = bootstrap_std(
                v
            )
        return output_result, std_dict, percentile_25_dict, percentile_975_dict
    return output_result, None, None, None


def bootstrap_std(data, n_bootstrap=1000, ci=0.95):
    """
    Args:
        data: a list of values
        n_bootstrap: number of bootstrap samples
        ci: confidence interval
    Returns:
        a tuple of mean, lower bound, upper bound
    """
    n = len(data)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, n, replace=True)
        means.append(np.mean(sample))
    means = np.array(means)
    lower_bound = np.percentile(means, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(means, (1 + ci) / 2 * 100)
    std = np.std(means)
    return std, lower_bound, upper_bound


def main(
    jsons_path,
    bootstrap,
    metric_per_instance=None,
    start=None,
    end=None,
    model_name=None,
):
    if metric_per_instance is None:
        data = read_json_files(jsons_path)
    else:
        if model_name is None:
            model_name = "tmp"
        data = {f"{model_name}_evaluation_result.json": metric_per_instance}

    col_names = [
        "res_stacked_image",
        "res_only_it_image",
        # "res_only_it_image_small",
    ]
    # metric_names = ["exact_match", "precision", "recall", "f1", "jaccard", "rouge1"]
    metric_names = ["exact_match", "jaccard"]

    for metric_name in metric_names:
        if bootstrap:
            std_list = []
            lower_bound_list = []
            upper_bound_list = []

        mean_list = []
        for target_domain in col_names:
            mean, std, lower_bound, upper_bound = get_score(
                data, target_domain, metric_name, start, end, bootstrap=bootstrap
            )
            mean_list.append(mean)

            if bootstrap:
                std_list.append(std)
                lower_bound_list.append(lower_bound)
                upper_bound_list.append(upper_bound)

        dicts2csv(
            mean_list,
            col_names,
            f"{metric_name}_{start}_{end}_mean_score.csv",
        )
        if bootstrap:
            dicts2csv(
                std_list,
                col_names,
                f"{metric_name}_{start}_{end}_std.csv",
            )
            dicts2csv(
                lower_bound_list,
                col_names,
                f"{metric_name}_{start}_{end}_lower_bound.csv",
            )
            dicts2csv(
                upper_bound_list,
                col_names,
                f"{metric_name}_{start}_{end}_upper_bound.csv",
            )


# print(get_score(data, target_domain, target_metric_name))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation pipeline of VCR.")
    parser.add_argument(
        "--jsons_path",
        type=str,
        help="path to the folder containing all the json files",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="also calculate the bootstrap confidence interval",
    )
    args = parser.parse_args()
    main(args.jsons_path, args.bootstrap)
    # main("/home/work/VCR/eval_metrics", True)
