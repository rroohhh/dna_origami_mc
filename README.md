# Monte carlo DNA origami self assembly
This contains some experiments investigating DNA origami self assembly using MC methods.

The basic ideas is to take a number of chains made up of points with a constant spacing, but with arbitrary angle between them and associate each point in a chain with a specific Base.
Normal DNA has four bases: 
- adenine `A`
- guanine `G`
- cytosine `C`
- thymine `T`
where only `A` and `T` and `G` and `C` can form bonds. 

The set of chains is modified to generate a new state by choosing one of the points of one of the chains, choosing some angle and a direction and then rotation all points in that direction (up or down the chain from the choosen point) around the choosen point by the choosen angle.
If points from different chains with bases that can form bonds get close together, they form a bond, which causes the other chain to also rotate, if one of the points is rotated.

In case there are two bonds from the chain that gets modified to another chain, one above and one below the point around which the first chain gets rotated, the other chain gets stretched such that the bonded points stay together.

Each state is associated a energy made up of a globally acting part, that pulls points of different chains together and a locally acting force with a strict cutoff representing the binding force of bonded points. 

The new state is accepted if the energy is lower, or with a random chance if the energy is higher, depending on the energy difference.

For more details and some basic analysis take look at the `ipython` [notebook](./Monte%20Carlo%20DNA%20Origami.ipynb) or the [pdf export](./Monte%20Carlo%20DNA%20Origami.pdf).
