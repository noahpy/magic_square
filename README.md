
## Using Discrete Particle Swarm Optimization for Magic Square Finding

DPSO is used to search the [7-dimensional space of magic squares](https://de.wikipedia.org/wiki/Magisches_Quadrat#Sonstiges).
User can specify an arbritrary magic number, a list of "wish numbers" to be found in the magic square and a cost function.
Furthermore, one can tweak on the hyperparameter of the PSO as well.

Developed for my grandpa.



#### Useage
For now configure your preferences in `run_dpso_magic_square.py`.
The output might look something like this:

```
--- Magic Square DPSO Solver ---
Magic Number Target: 90
Wish Numbers: [11, 3, 54, 19, 6, 22, 16, 5, 30, 21]
Number of Particles: 200
Maximum Iterations: 1000
------------------------------
Swarm initialized with 200 particles.
Initial Global Best Cost: 52.434165
------------------------------
Starting DPSO optimization for 1000 iterations...
Iteration 1/1000, Current Gbest Cost: 52.434165
Iteration 100/1000, Current Gbest Cost: 20.973666
Converged!
Optimization finished.

--- Optimization Results ---
Global Best Coefficient Vector Found: [30  6  8 -3  6 19 24]
Global Best Cost Found: 20.973666
Coefficient Vector is Valid: True
Magic Square Sum (if valid): 90

Resulting Magic Square:
[[24  3 49 14]
 [25 38 21  6]
 [ 5 19 12 54]
 [36 30  8 16]]

Coverage of wished numbers: 8/10
```
