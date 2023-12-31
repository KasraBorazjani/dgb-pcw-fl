#!/bin/bash

stdbuf -oL python bashtest1.py > out1.out & 
stdbuf -oL python bashtest2.py > out2.out &