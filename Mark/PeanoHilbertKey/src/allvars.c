#include "allvars.h"

int NumPart;
int MaxNodes;
int *Nextnode;
int *Father;
struct particle_data *P;
struct NODE *Nodes_base;	/* points to the actual memory allocted for the nodes */
struct NODE *Nodes;		/* this is a pointer used to access the nodes which is shifted such that Nodes[All.MaxPart] gives the first allocated node */

int BitsPerDimension;
