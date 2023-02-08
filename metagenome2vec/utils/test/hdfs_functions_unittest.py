#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 19:36:47 2018

@author: maxence
"""

import unittest
import numpy as np
from Pipeline.utils.hdfs_functions import *
from pyspark import SparkContext


class TestHDFSFunctions(unittest.TestCase):
    def test_rdd_cleaning(self):
        rdd = self.sc.parallelize(
            [
                "('ERR479504.9 D3FCO8P1:1:1101:1910:2175/1', 'CGAGTAAAAATTTACCAAAGAATGCAATCGTATGCAGACAATGTCCGCGAGCTGGGCAAGGGCCGTCCACGAGCCAACCATCGCCTGTC', 'HJJJGHJJJJJJJJJJJJJIJJIJJJJJJJJJJJIJJIJJJJJJIIIJJEDDDEEDDDDDDDDD@BDDDDDDDDDDDDDDDDDDDDDDD', 'CGCGCGAGGAGCGTCTGAAGGTCGGCATCACCGACGAGCTGGTACGTCTCGCGGTCGGCGTCGAAGACAAGCAGGACCTCATCGAGG', 'HJJJJJJIJIJJJJJJJJJJJCGHIHHFFFFEDDDDDDDDDD<CDDDDDDDBDD8BBDBDBDDDDDDDDDBBBCD?DDDDDCDDCBD')",
                "('ERR479504.21 D3FCO8P1:1:1101:3580:2128/1', 'GAATGTATTCTTCGATGATAGAGTCCTGTAAGGCAACGCTGGAGGTAGAAATCACCACCGGCTGAGAACCTGCAGGTCCATGCGGGGCA', 'GIJJJIIJJJJJJJJJJJIJIJJFGIJJIJJJIJIJJIJJJIGIJFGHJJJJJJHHGHFFDCBBDDDDDDDDDDDDBDD>CCDDDDDDD', 'GGTTTGTCGGTTCGGGAAGAACAAATCACCCTCTGTCACGCCATGCTGGATACCCTGCTCAAAAACAATATCGCTCTGTGCGACGCAGG', 'HJHIIJIJJJHIJJJJJJGIJGHIJFHIJJJIJIJIIIHHJJBEHHHHFFFFFEECCDCCDDDDDD?<:A@CBDDDDDCDCBDDDD@BB')",
            ]
        )
        rdd2 = self.sc.parallelize(
            [
                "('ERR479504.9 D3FCO8P1:1:1101:1910:2175/1', 'CGAGTAAAAATTTACCAAAGAATGCAATCGTATGCAGACAATGTCCGCGAGCTGGGCAAGGGCCGTCCACGAGCCAACCATCGCCTGTC', 'HJJJGHJJJJJJJJJJJJJIJJIJJJJJJJJJJJIJJIJJJJJJIIIJJEDDDEEDDDDDDDDD@BDDDDDDDDDDDDDDDDDDDDDDD')",
                "('ERR479504.21 D3FCO8P1:1:1101:3580:2128/1', 'GAATGTATTCTTCGATGATAGAGTCCTGTAAGGCAACGCTGGAGGTAGAAATCACCACCGGCTGAGAACCTGCAGGTCCATGCGGGGCA', 'GIJJJIIJJJJJJJJJJJIJIJJFGIJJIJJJIJIJJIJJJIGIJFGHJJJJJJHHGHFFDCBBDDDDDDDDDDDDBDD>CCDDDDDDD')",
            ]
        )
        rdd3 = self.sc.parallelize(
            [
                "('ERR479504.9 D3FCO8P1:1:1101:1910:2175/1', 'CGTC', 'HJJJGHJJJJJJJJJJJJJIJJIJJJJJJJJJJJIJJIJJJJJJIIIJJEDDDEEDDDDDDDDD@BDDDDDDDDDDDDDDDDDDDDDDD', 'TGGA', 'HJJJJJJIJIJJJJJJJJJJJCGHIHHFFFFEDDDDDDDDDD<CDDDDDDDBDD8BBDBDBDDDDDDDDDBBBCD?DDDDDCDDCBD')",
                "('ERR479504.21 D3FCO8P1:1:1101:3580:2128/1', 'GAATGTATTCTTCGATGATAGAGTCCTGTAAGGCAACGCTGGAGGTAGAAATCACCACCGGCTGAGAACCTGCAGGTCCATGCGGGGCA', 'GIJJJIIJJJJJJJJJJJIJIJJFGIJJIJJJIJIJJIJJJIGIJFGHJJJJJJHHGHFFDCBBDDDDDDDDDDDDBDD>CCDDDDDDD', 'GGTTTGTCGGTTCGGGAAGAACAAATCACCCTCTGTCACGCCATGCTGGATACCCTGCTCAAAAACAATATCGCTCTGTGCGACGCAGG', 'HJHIIJIJJJHIJJJJJJGIJGHIJFHIJJJIJIJIIIHHJJBEHHHHFFFFFEECCDCCDDDDDD?<:A@CBDDDDDCDCBDDDD@BB')",
            ]
        )

        rdd_res = rdd_cleaning(rdd, 3, 3, True)
        rdd_res2 = rdd_cleaning(rdd2, 3, 3, True)
        rdd_res3 = rdd_cleaning(rdd3, 3, 3, True)

        self.assertTrue(len(rdd_res.collect()) == 4)
        self.assertTrue(len(rdd_res2.collect()) == 2)
        self.assertTrue(len(rdd_res3.collect()) == 2)

    """
    def test_rdd_to_rdd_embedding(self):
        embeddings = np.array([[3.4, 0.3, -4.5], [1.4, 0.5, -1.5], [0.5, 0.1, 1.4]])
        dico_index = {"AAA": 0, "CCC": 1, "TTT": 2}
        k = 3
        s = 1
        rdd = self.sc.parallelize(["AAAA", "TTT", "CCC", "AAA"])
        rdd_res = rdd_to_rdd_embeddings(rdd, embeddings, dico_index, k, s, "normal")
        rdd_res_mean = rdd_to_rdd_embeddings(rdd, embeddings, dico_index, k, s, "mean")
        np.testing.assert_array_equal(np.array([[[3.4, 0.3, -4.5], [3.4, 0.3, -4.5]], [[0.5, 0.1, 1.4]],
                                                 [[1.4, 0.5, -1.5]], [[3.4, 0.3, -4.5]]]),
                                      np.array(rdd_res.collect()))
        np.testing.assert_array_equal((np.mean([np.mean([[3.4, 0.3, -4.5], [3.4, 0.3, -4.5]], axis=0).tolist()] +
                                               [[1.4, 0.5, -1.5], [0.5, 0.1, 1.4], [3.4, 0.3, -4.5]], axis=0)),
                                      np.mean(rdd_res_mean.collect(), axis=0))
    """

    def test_rdd_to_rdd_context(self):
        rdd = self.sc.parallelize(["ATGCGGT", "GGATTC"])
        rdd_res = rdd_to_rdd_context(rdd, k=3, kc=3, w=1, s=1)
        rdd_true = self.sc.parallelize(
            [(("ACC", "GCA"), 1), (("ATG", "CGG"), 1), (("GGA", "TTC"), 1)]
        )
        self.assertEqual(rdd_res.collect(), rdd_true.collect())

    def test_rdd_context_to_rdd_context_proba(self):
        rdd = self.sc.parallelize([(("ATG", "TTT"), 2), (("CCC", "TGA"), 4)])
        rdd_res = rdd_context_to_rdd_context_proba(rdd)
        rdd_true = self.sc.parallelize(
            [(("CCC", "TGA"), 4.0 / 6), (("ATG", "TTT"), 2.0 / 6)]
        )
        self.assertEqual(rdd_res.collect(), rdd_true.collect())

    def test_rdd_to_rdd_word_count(self):
        rdd = self.sc.parallelize(["ATGCGGT", "ATGCGG"])
        rdd_res = rdd_to_rdd_word_count(rdd, k=6, s=1, reverse=False, one=True)
        rdd_true = self.sc.parallelize([("ATGCGG", 2), ("TGCGGT", 1)])
        self.assertEqual(rdd_res.collect(), rdd_true.collect())

    def test_rdd_word_count_to_rdd_index(self):
        rdd = self.sc.parallelize([("ATGCGG", 4), ("TGCGGT", 1), ("GGTAGG", 3)])
        rdd_res = rdd_word_count_to_rdd_index(rdd)
        rdd_true = self.sc.parallelize([("ATGCGG", 1), ("GGTAGG", 2), ("TGCGGT", 3)])
        self.assertEqual(rdd_res.collect(), rdd_true.collect())

    def test_merge_rdd_count(self):
        rdd_1 = self.sc.parallelize([("ATGCGG", 4), ("TGCGGT", 1), ("GGTAGG", 3)])
        rdd_2 = self.sc.parallelize([("ATGCGG", 4), ("AAAAAA", 2)])
        rdd_res = merge_rdd_count(rdd_1, rdd_2)
        rdd_true = self.sc.parallelize(
            [("ATGCGG", 8), ("TGCGGT", 1), ("GGTAGG", 3), ("AAAAAA", 2)]
        )
        l = rdd_res.collect()
        l2 = rdd_true.collect()
        for elem in l:
            self.assertTrue(elem in l2)

    def test_get_rdd_GC_rate(self):
        s1 = "GAGTCGGCGGCCGCCTAGGTTCCC"
        s2 = "CTAGGCTT"
        rdd = self.sc.parallelize([s1, s2])
        gc_rate = get_rdd_GC_rate(rdd)
        self.assertEqual(gc_rate, (3.0 / 4 + 1.0 / 2) / 2)
        t1 = ("xaonfeaoifnaoz", "GAGTCGGCGGCCGCCTAGGTTCCC", "aindoizandza")
        t2 = ("pzlpdzapdl", "CTAGGCTT", "dzodzadad")
        rdd = self.sc.parallelize([t1, t2])
        gc_rate = get_rdd_GC_rate(rdd, 1)
        self.assertEqual(gc_rate, (3.0 / 4 + 1.0 / 2) / 2)
        t1 = "('xaonfeaoifnaoz', 'GAGTCGGCGGCCGCCTAGGTTCCC', 'aindoizandza')"
        t2 = "('pzlpdzapdl', 'CTAGGCTT', 'dzodzadad')"
        rdd = self.sc.parallelize([t1, t2])
        gc_rate = get_rdd_GC_rate(rdd, 1, eval)
        self.assertEqual(gc_rate, (3.0 / 4 + 1.0 / 2) / 2)

    def setUp(self):
        self.sc = SparkContext()

    def tearDown(self):
        self.sc.stop()


if __name__ == "__main__":
    unittest.main()
