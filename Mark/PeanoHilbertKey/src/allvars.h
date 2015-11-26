#include <stdio.h>


extern int NumPart;
extern int  MaxNodes;
extern int *Nextnode;
extern int *Father;

extern int BitsPerDimension;
extern struct particle_data
{
  int Pos[3];			/*!< particle position at its current time */
  long long Id;                 /*!< peano hilbert key */
}
*P;                             /*!< points to particles on this processor */

extern struct NODE
{
  float len;			/*!< sidelength of treenode */
  float center[3];		/*!< geometrical center of node */
  union
  {
    int suns[8];		/*!< temporary pointers to daughter nodes */
    struct
    {
      float s[3];               /*!< center of mass of node */
      int mass;                 /*!< mass of node */
      int sibling;              /*!< this gives the next node in the walk in case the current node can be used */
      int nextnode;             /*!< this gives the next node in case the current node needs to be opened */
      int father;               /*!< this gives the parent node of each node (or -1 if we have the root node) */
    }
    d;
  }
  u;
}
*Nodes_base,                    /*!< points to the actual memory allocted for the nodes */
*Nodes;                         /*!< this is a pointer used to access the nodes which is shifted such that Nodes[NumPart] gives the first allocated node */







