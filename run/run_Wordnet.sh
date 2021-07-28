#!/bin/bash
while read -r line  || [ -n "$line" ] ; do eval $line ; done < run_Wordnet.txt