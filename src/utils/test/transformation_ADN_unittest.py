#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 17:30:47 2017

@author: maxence
"""

import unittest
from Pipeline.utils.transformation_ADN import *
import numpy as np


class TestStructurationSequence(unittest.TestCase):
    def test_create_context(self):
        s1 = "AACTCG"
        s2 = "AACTCGG"
        s3 = "AACTCGGA"
        s4 = "TAGTCG"
        real_res1 = [('AA','CT'), ('AA','CG'), ('AC','TC'),
                    ('CT','AA'), ('CT','CG'), ('TC','AC'),
                    ('CG','CT'), ('CG','AA')]
        real_res2 = [('AA','CT'), ('AA','CG'), ('TC','AC'),
                    ('TC','GG')]
        real_res3 = [('AA','CTC'), ('AA','GGA'), ('AC','TCG'),
                    ('CT','CGG'),('TC','AAC'), ('TC','GGA'), 
                    ('CG','ACT'),('GG','CTC'), ('GA','TCG'),
                    ('GA','AAC')]
        real_res3_bis = [(x, 1) for x in real_res3]
        real_res4 = [(("AC", "TA"), 1), (("CG", "TA"), 1), (("AG", "TC"), 1),
                     (("GT", "TA"), 1), (("CG", "AC"), 1), (("CT", "GA"), 1),
                     (("CG", "GT"), 1), (("CG", "TA"), 1)]
        real_res5 = [("AC", "TA"), ("CG", "TA"), ("AG", "TC"),
                     ("GT", "TA"), ("CG", "AC"), ("CT", "GA"),
                     ("CG", "GT"), ("CG", "TA")]
        self.assertListEqual(sorted(create_context(s1, k=2, kc=2, w=2)), sorted(real_res1))
        self.assertListEqual(sorted(create_context(s2, k=2, kc=2, w=2, s=3)), sorted(real_res2))
        self.assertListEqual(sorted(create_context(s3, k=2, kc=3, w=2)), sorted(real_res3))
        self.assertListEqual(sorted(create_context_one(s3, k=2, kc=3, w=2)), sorted(real_res3_bis))
        self.assertListEqual(sorted(create_context_reverse_one(s4, k=2, kc=2, w=2)), sorted(real_res4))
        self.assertListEqual(sorted(create_context_reverse(s4, k=2, kc=2, w=2)), sorted(real_res5))

    def test_cut_word(self):
        s1 = "GTATAG"
        s2 = "AATTGCGCCA"
        self.assertEqual(cut_word(s1, 2), ["GT", "TA", "AT", "TA", "AG"])
        self.assertEqual(cut_word(s2, 3, 2), ["AAT", "TTG", "GCG", "GCC"])
        self.assertEqual(cut_word_one(s1, 2), [("GT", 1), ("TA", 1), ("AT", 1), ("TA", 1), ("AG", 1)])
        self.assertEqual(cut_word_one(s2, 3, 2), [("AAT", 1), ("TTG", 1), ("GCG", 1), ("GCC", 1)])

    def test_reverse(self):
        self.assertEqual(reverse_one_chain_not_AC("GATACA"),"TGTATC")
        self.assertEqual(reverse_one_chain_not_AC("TAGGA"),"TCCTA")
        self.assertEqual(reverse_one_chain_not_AC("AGATACCA"),"AGATACCA")
        self.assertEqual(reverse_one_chain_not_AC("CGAGCT"),"CGAGCT")
        self.assertEqual(reverse_one_chain("CGAGCT"), "AGCTCG")
        self.assertEqual(reverse_two_chains("GTA","CAT"),("ATG","TAC"))
        self.assertEqual(reverse_two_chains("ATA","GTA"),("ATA","GTA"))
        self.assertEqual(reverse_two_chains("GTA","CAA"),("GTA","CAA"))
        self.assertEqual(reverse_two_chains("GA","TATAT"),("AT","ATATC"))
        self.assertEqual(reverse_two_chains("GATA","TAT"),("ATAT","ATC"))

    """
    def test_transform_dna_to_embedding(self):
        s1 = "GTACACAGACT"
        s2 = "ATGACCACGAT"
        k = 4
        s = 1
        index = {"GTAC":2,"CACA":3, "AGAC":1}
        embeddings = np.array([[float(x)]*3 for x in range(6)])
        np.testing.assert_array_equal(transform_dna_to_embedding(s1, embeddings, index, k, s, "mean"),
                                      np.array([2., 2., 2.]))
        np.testing.assert_array_equal(transform_dna_to_embedding(s1, embeddings, index, k, s, "sum"),
                                      np.array([6., 6., 6.]))
        np.testing.assert_array_equal(transform_dna_to_embedding(s1, embeddings, index, k, s, "normal"),
                                      np.array([[2., 2., 2.], [3., 3., 3.], [1., 1., 1.]]))
        np.testing.assert_array_equal(transform_dna_to_embedding(s2, embeddings, index, k, s), np.array([]))
    """

    def test_kmer_to_int(self):
        kmer1 = "ATAG"
        kmer2 = "ACC"
        self.assertEqual(kmer_to_int(kmer1), 140)
        self.assertEqual(kmer_to_int(kmer2), 20)


if __name__=="__main__":
    unittest.main()
