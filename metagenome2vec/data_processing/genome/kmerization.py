import os
from metagenome2vec.utils import file_manager
from metagenome2vec.utils import parser_creator
from metagenome2vec.utils import transformation_ADN

from tqdm import tqdm
import re


if __name__ == "__main__":

    args = parser_creator.ParserCreator().parser_genome_kmerization()
    file_manager.create_dir(os.path.dirname(args.path_save), "local")
    k = args.k_mer_size
    s = args.step

    line_to_write = ""
    n_lines = sum(1 for line in open(args.path_data, 'r'))
    with open(args.path_data, 'r') as genome:
        with open(args.path_save, 'w') as f_res:
            for line in tqdm(genome, total=n_lines):
                if line[0] == ">" or line == "\n":
                    continue
                line = re.sub("NN+", "", line).replace("\n", "")
                transformation_ADN.cut_and_write_read(f_res, line, k, s)
