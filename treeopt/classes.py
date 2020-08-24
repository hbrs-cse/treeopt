#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 09:35:26 2020

@author: alex
"""


class metamodell:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def do_something(self):
        print("ich mache ein metamodell")
        return self.a + self.b + self.c


class fqm:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def do_something(self):
        print("ich mache eine Fehlerquadradmethode")
        return -self.a - self.b - self.c
