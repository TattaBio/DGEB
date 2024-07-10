"""Script to replicate results from the DGEB paper."""

import torch
import dgeb
from functools import partial


ALL_DEVICES = list(range(torch.cuda.device_count()))
DEFAULT_BATCH_SIZE = 64
DEFAULT_SEQ_LEN = 1024


get_model = partial(
    dgeb.get_model,
    devices=ALL_DEVICES,
    batch_size=DEFAULT_BATCH_SIZE,
    max_seq_length=DEFAULT_SEQ_LEN,
)


def main():
    ######################### Protein Models #########################
    protein_tasks = dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN)
    protein_evaluation = dgeb.DGEB(tasks=protein_tasks)

    # ESM models.
    protein_evaluation.run(get_model("facebook/esm2_t6_8M_UR50D"))
    protein_evaluation.run(get_model("facebook/esm2_t12_35M_UR50D"))
    protein_evaluation.run(get_model("facebook/esm2_t30_150M_UR50D"))
    protein_evaluation.run(get_model("facebook/esm2_t33_650M_UR50D", batch_size=32))
    protein_evaluation.run(get_model("facebook/esm2_t36_3B_UR50D", batch_size=1))

    # ESM3 models.
    protein_evaluation.run(get_model("esm3_sm_open_v1", batch_size=1, devices=[0]))

    # ProtT5 models.
    protein_evaluation.run(get_model("Rostlab/prot_t5_xl_uniref50", batch_size=32))
    protein_evaluation.run(get_model("Rostlab/prot_t5_xl_bfd", batch_size=32))

    # ProGen2 models.
    protein_evaluation.run(get_model("hugohrban/progen2-small"))
    protein_evaluation.run(get_model("hugohrban/progen2-medium", batch_size=32))
    protein_evaluation.run(get_model("hugohrban/progen2-large", batch_size=1))
    protein_evaluation.run(get_model("hugohrban/progen2-xlarge", batch_size=1))

    ######################### DNA Models #########################
    dna_tasks = dgeb.get_tasks_by_modality(dgeb.Modality.DNA)
    dna_evaluation = dgeb.DGEB(tasks=dna_tasks)

    # Evo models
    dna_evaluation.run(
        get_model(
            "togethercomputer/evo-1-8k-base", batch_size=1, seq_len=8192, devices=[0]
        )
    )
    # 131k will OOM so we use half this length.
    evo_131k_max_seq_len = int(131072 / 2)
    dna_evaluation.run(
        get_model(
            "togethercomputer/evo-1-131k-base",
            batch_size=1,
            seq_len=evo_131k_max_seq_len,
            devices=[0],
        )
    )

    # Nucleotide Transformer models.
    dna_evaluation.run(
        get_model("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    )
    dna_evaluation.run(
        get_model("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species")
    )
    dna_evaluation.run(
        get_model("InstaDeepAI/nucleotide-transformer-v2-250m-multi-species")
    )
    dna_evaluation.run(
        get_model("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    )
    dna_evaluation.run(
        get_model("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", batch_size=1)
    )


if __name__ == "__main__":
    main()
