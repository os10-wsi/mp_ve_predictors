# -*- coding: utf-8 -*-
"""Train DeepSequence on an alignment and score a variant list.

Runs inside the Python 2.7 container built from the Dockerfile beside this
file. The mutation effect is ELBO(mutant) - ELBO(wild type), so higher means
more tolerated, the convention in Riesselman et al. 2018.
"""

from __future__ import print_function

import argparse
import csv
import sys

sys.path.append("/opt/DeepSequence")

import helper     # noqa: E402
import model      # noqa: E402
import train      # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--mutations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=300000)
    args = parser.parse_args()

    data = helper.DataHelper(
        alignment_file=args.alignment, working_dir="/data",
        calc_weights=True, theta=0.2,
    )
    vae = model.VariationalAutoencoder(
        data, batch_size=100, encoder_architecture=[1500, 1500],
        decoder_architecture=[100, 500], n_latent=30, logit_p=0.001,
        sparsity="logit", encode_nonlinearity_type="relu",
        decode_nonlinearity_type="relu", final_decode_nonlinearity="sigmoid",
        final_pwm_scale=True, conv_decoder_size=40, convolve_patterns=True,
        n_patterns=4, random_seed=42,
    )
    train.train(data, vae, num_updates=args.iterations, save_progress=False,
                save_parameters=False)

    with open(args.mutations) as handle:
        mutants = [row["mutant"].strip() for row in csv.DictReader(handle)]

    with open(args.output, "w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mutant", "elbo_difference"])
        for mutant in mutants:
            try:
                value = data.custom_mutant_matrix_from_list(
                    [mutant], vae, N_pred_iterations=args.samples
                )
            except Exception as error:                 # noqa: BLE001
                print("  %s: %s" % (mutant, error), file=sys.stderr)
                continue
            writer.writerow([mutant, float(value[0])])
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
