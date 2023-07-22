# GFSM

This is the repository for GPU Forest-Based Subgraph Matching.

To run an experiment use: `./GFSM "Query Graph Location" "Data Graph Location`

Note that the graph format should look like:
```
t # 0
2 2 1 1
v 0 1
v 1 1
e 0 1 1
e 1 0 1
t # -1
```
- `t # 0` and `t # -1` are the start and end of a file
  
- The first row of numbers (the header) are: **Vertex Count**, **Edge Count**, **Vertex Label Max**, **Edge Label Max**
  
- `v 1 1` signifies the `2nd`vertex (Starts at 0) has label `1`

- `e 0 1 1` signifies an edge exists between vertex 0 and 1 with label `1`

Note that the Environment.h file contain important defines (mattering on GPU target/experiments):

&nbsp;&nbsp;&nbsp;**CCSRRelSize** - Max size of a CCSR relation, the larger this is, the smaller max size of a CCSR

&nbsp;&nbsp;&nbsp;**MAXSOLNSIZE** - Max size of a solution (in mappings/FFS M-Nodes)

&nbsp;&nbsp;&nbsp;**MAXSCANSIZE** - Max size of a exclusive scan buffer/result

&nbsp;&nbsp;&nbsp;**MEMLIMIT** - Initial max amount of memory that can be used

&nbsp;&nbsp;&nbsp;**UPPERMEMLIMIT** - Max amount of memory that can be used if a problem overflows with MEMLIMIT
