#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:00:57 2017

@author: maxence
"""

import re
import numpy as np
cimport numpy as np
from Bio.Seq import Seq


cdef dict dico_nucleotide_to_int = {"A": 0, "T": 3, "C": 1, "G": 2}
cdef dict dico_int_to_nucleotide = {0: "A", 3: "T", 1: "C", 2: "G"}

cdef dict pt = {'match': 1, 'mismatch': -1, 'gap': -1}


cpdef int kmer_to_int(str kmer):
    cdef:
        int res = 0
        int i
    for i in range(len(kmer)):
        res += 4**i * dico_nucleotide_to_int[kmer[i]]
    return res

cpdef str int_to_kmer(int num, int nb_nucleotide):
    cdef:
        str res = ""
        int i
    for i in range(nb_nucleotide, 0, -1):
        res += dico_int_to_nucleotide[num % 4]
        num /= 4
    return res


cpdef str complement(str s, int not_AC=False):
    """
    Complement of a DNA sequence

    :param s:  String, Chain of nucleotides
    :return: s inversed

    Example :
        complement("GTT") => "CAA"
        complement("TGA") => "ACT"
    """
    if not_AC:
        if s[0] in "AC":
            return s
        return Seq(s).complement()._data
    return Seq(s).complement()._data


cpdef str reverse_complement(str s, int not_AC=False):
    """
    Reverse complement of a DNA sequence

    :param s:  String, Chain of nucleotides
    :return: s inversed

    Example :
        reverse_complement("GTT") => "AAC"
        reverse_complement("TGA") => "TCA"
    """
    if not_AC:
        if s[0] in "AC":
            return s
        return Seq(s).reverse_complement()._data
    return Seq(s).reverse_complement()._data


def cut_word(str x, int k, int s=1, int remove_unk=0):
    """
    Convert a sequence of nucleotides into a list of k nucleotides spaced by s nucleotides

    :param x: String, nucleotide chain
    :param k: int, number of nucleotides by chain
    :param s: int (default=1), The step between each sequences
    :param remove_unk, int (default=0), if 1 remove unk kmer
    :return: l : List, list of String where one element corresponds to k nucleotides.
    """
    return cut_word_in_c(x, k, s, remove_unk)


cdef np.ndarray[np.str, ndim=1] cut_word_in_c(str x, int k, int s=1, int remove_unk=0):
    """
    Convert a sequence of nucleotides into a list of k nucleotides spaced by s nucleotides

    :param x: String, nucleotide chain
    :param k: int, number of nucleotides by chain
    :param s: int (default=1), The step between each sequences
    :param remove_unk, int (default=0), if 1 remove unk kmer
    :return: l : List, list of String where one element corresponds to k nucleotides.
    """
    cdef:
        int i
        np.ndarray[np.str, ndim=1] res = np.zeros(np.int(np.ceil((len(x) - k + 1) / s)), dtype=object)
    for i in range(0, len(res), s):
        res[i] = reverse_complement(x[i:i + k], not_AC=True)
    if remove_unk:
        return remove_unk_kmer(res)
    return res


cpdef str preprocess_read_str(str read, int k, int max_length=-1):
    """
    Create all 'sentences' of feasible kmers with read
    All sentences are represented as a list of k-mers
    """
    cdef:
        int i=0
    if max_length > 0:
        read = read[:max_length]
    cdef str res = ""
    for i in range(len(read) - k + 1):
        res += " %s" % reverse_complement(read[i:i + k], not_AC=True)
    return res.strip()


cpdef np.ndarray[np.str, ndim=1] preprocess_several_reads_str(list L_read, int k, int max_length=-1):
    """
    Create all 'sentences' of feasible kmer with the list of reads L_read
    All sentences are represented as a list of k-mers
    ps : The first column is an index to retrieve the line from a same read
    """
    return np.array([preprocess_read_str(read, k, max_length) for read in L_read])


cpdef np.ndarray[np.int_t, ndim=1] preprocess_read(str read, int k, dict dico_index,
                                                   int index_unk=-1, int index_pad=-2,
                                                   int max_length=-1, int min_length=-1):
    """
    Create all 'sentences' of feasible kmers with read
    All sentences are represented as a list of k-mers
    """
    cdef:
        int i=0, read_length, unk_seen=0, j=0, cpt_remove=0
    if max_length > 0:
        read = read[:max_length]
    #cdef np.ndarray[np.str, ndim=1] read_cut = cut_word(read, k, s=1)
    read_length = len(read)
    min_length = read_length if min_length <= 0 else min_length
    cdef np.ndarray[np.int_t, ndim=1] res = np.zeros(max(read_length, min_length) - k + 1, dtype=np.int)
    for i in range(len(res)):
        kmer = complement(read[i:i + k], not_AC=True)
        if i < read_length:
            try:
                res[j] = dico_index[kmer]
                unk_seen = 0
                j += 1
            except KeyError:
                if not unk_seen:
                    res[j] = index_unk
                    j += 1
                    unk_seen = 1
                else:
                    cpt_remove += 1  # To remove additional cases in numpy array
        else:
            res[j] = index_pad
            j += 1
    if cpt_remove > 0:
        return res[:-cpt_remove]
    return res


cpdef np.ndarray[np.int_t, ndim=2] preprocess_several_reads(list L_read, int k, dict dico_index,
                                                            int index_unk=-1, int index_pad=-2,
                                                            int max_length=-1, int min_length=-1):
    """
    Create all 'sentences' of feasible kmer with the list of reads L_read
    All sentences are represented as a list of k-mers
    ps : The first column is an index to retrieve the line from a same read
    """

    cdef:
        int maxlen
        str read
        list reads_prepro, leans
        np.ndarray A_kmer, read_prepro
    reads_prepro = [preprocess_read(read, k, dico_index, index_unk, index_pad, max_length, min_length) for read in L_read]
    lens = [len(read_prepro) for read_prepro in reads_prepro]
    maxlen = max(lens)
    cdef:
        np.ndarray[np.int_t, ndim = 2] arr = np.zeros((len(reads_prepro), maxlen), dtype=np.int) + index_pad
        np.ndarray mask = np.arange(maxlen) < np.array(lens, dtype=np.int)[:, None]
    arr[mask] = np.concatenate(reads_prepro)
    return arr


cpdef np.ndarray[np.str, ndim=1] remove_unk_kmer(np.ndarray[np.str, ndim=1] kmer_array):
    cdef:
        int cpt_remove=0, i=0
        str kmer
        np.ndarray[np.str, ndim=1] res = np.zeros(len(kmer_array), dtype=object)
    for kmer in kmer_array:
        if re.match("^.*[URYKMSWBDHVNX]+.*$", kmer) is None:
            res[i] = kmer
            i += 1
        else:
            cpt_remove += 1
    return np.array(res[:-cpt_remove], dtype=str)


cpdef cut_and_write_read(f, str read, int k, int s, str mode="c", int remove_unk=0):
    cdef:
        int i, unk_seen=0, cpt_remove=0
        str kmer
    if mode == "b":
        read = read[:(len(read) / k) * k]
    elif mode == "e":
        read = read[len(read) % k:]
    else:
        pass
    cdef np.ndarray[np.str, ndim=1] res = cut_word(read, k, s, remove_unk)
    cdef np.ndarray[np.str, ndim=1] res2 = np.zeros(len(res), dtype=object)
    if remove_unk:
        f.write(' '.join(res).strip()+'\n')
    else:
        i = 0
        for kmer in res:
            kmer = re.sub("^.*[URYKMSWBDHVNX]+.*$", "<unk>", kmer)
            if kmer != "<unk>":
                res2[i] = kmer
                i += 1
                unk_seen = 0
            else:
                 if not unk_seen:
                     res2[i] = kmer
                     i += 1
                     unk_seen = 1
                 else:
                     cpt_remove += 1
        f.write(' '.join(res2[:-cpt_remove]).strip()+'\n')


cpdef cut_and_write_reads(list L_reads, f, int k, int s=1, str mode="c", remove_unk=False):
    cdef:
        int i
    for i in range(len(L_reads)):
        cut_and_write_read(f, L_reads[i], k, s, mode, remove_unk)


cpdef int mch(str alpha, str beta):
    if alpha == beta:
        return pt['match']
    elif alpha == '-' or beta == '-':
        return pt['gap']
    else:
        return pt['mismatch']


cpdef int needle(str s1, str s2):
    cdef:
        int m, n, i, j, seqN, seq_score = 0, ident = 0
        float diag, delete, insert, score_current, score_diag, score_left, score_up
        str align1, align2, a1, a2, sym = ''
        np.ndarray[np.float_t, ndim=2] score

    m, n = len(s1), len(s2)
    score = np.zeros((m + 1, n + 1))

    # Initialization
    for i in range(m + 1):
        score[i][0] = pt['gap'] * i
    for j in range(n + 1):
        score[0][j] = pt['gap'] * j

    # Fill
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = score[i - 1][j - 1] + mch(s1[i - 1], s2[j - 1])
            delete = score[i - 1][j] + pt['gap']
            insert = score[i][j - 1] + pt['gap']
            score[i][j] = max(diag, delete, insert)

    align1, align2 = '', ''
    i, j = m, n

    while i > 0 and j > 0:
        score_current = score[i][j]
        score_diag = score[i - 1][j - 1]
        score_left = score[i][j - 1]
        score_up = score[i - 1][j]

        if score_current == score_diag + mch(s1[i - 1], s2[j - 1]):
            a1, a2 = s1[i - 1], s2[j - 1]
            i, j = i - 1, j - 1
        elif score_current == score_up + pt['gap']:
            a1, a2 = s1[i - 1], '-'
            i -= 1
        elif score_current == score_left + pt['gap']:
            a1, a2 = '-', s2[j - 1]
            j -= 1
        align1 += a1
        align2 += a2

    while i > 0:
        a1, a2 = s1[i - 1], '-'
        align1 += a1
        align2 += a2
        i -= 1

    while j > 0:
        a1, a2 = '-', s2[j - 1]
        align1 += a1
        align2 += a2
        j -= 1

    align1 = align1[::-1]
    align2 = align2[::-1]
    seqN = len(align1)
    for i in range(seqN):
        a1 = align1[i]
        a2 = align2[i]
        if a1 == a2:
            sym += a1
            ident += 1
            seq_score += mch(a1, a2)
        else:
            seq_score += mch(a1, a2)
            sym += ' '
    return seq_score


cpdef int ED(str s, str t):
    cdef:
        int res = 0
        int i
    for i in range(len(s)):
        res += (s[i] != t[i])
    return res


cpdef dict create_distance(similarities, algo, reverse_index):
    D_distance = {}  # Key is the distance value, value is the list of all mean the cosine similarity
    for i in range(len(similarities)):
        for j in range(i + 1, len(similarities)):
            kmer_1, kmer_2 = reverse_index[i], reverse_index[j]
            if kmer_1 == "UNK" or kmer_2 == "UNK" or kmer_1 == kmer_2:
                continue
            dist = algo(kmer_1, kmer_2)
            sim = similarities[i][j]
            if dist in D_distance:
                D_distance[dist].append(sim)
            else:
                D_distance[dist] = [sim]
    return D_distance

