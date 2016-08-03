Learning Unitary Operators with Help from u(n)
----

Code for [Learning Unitary Operators with Help From u(n)](http://arxiv.org/abs/1607.04903) by Stephanie L. Hyland (me) and Gunnar Rätsch.

## How to use

### Minimal example

(assuming iPython)

Scripts for defining experiments (e.g. loss functions, parameter shapes and types) is `experiment_setups.py`. I have some 'presets' in there for specific comparative experiments. The main one is just called `rerun` for historical reasons. Others can be found at the end of the file - of particular interest are `test_random_projections` and `basis_test`.

```
> %run experiment_setups.py
> e_list = rerun(d=3)
```
The output from this is:
```
  (experiment projection): (re)setting loss function.
  (experiment complex_RNN): (re)setting loss function.
  (experiment general_unitary): (re)setting loss function.
  (experiment general_unitary_basis5): (re)setting loss function.
```
The script for actually running experiments is `learning_unitary_matrices.py`. It will take this list of `Experiment` objects.
```
> %run learning_unitary_matrices.py
> main(d=3, experiments=e_list)
```
The start of the output looks like:
```
  Running experiments:
  projection
  complex_RNN
  general_unitary
  general_unitary_basis5
  Will be saving output to output/d3_noise0.01_bn20_nb50000
  0 : generated U using: lie_algebra
  Adding noise...
  Adding noise...
  Adding noise...
  Running projection experiment!
  Initialising 18 real parameters.
  20 		VALI: 43.4122744893
  20020 		VALI: 5.61510049812
  ...
```
and so on.

### Running a new experiment

The `Experiment` class in `experiment_setups.py` allows you to define new experiments. It has these attributes:

- `project`: (boolean) project back to unitary after gradient update?
- `random_projections`: (integer) number of random projections to use instead of the full finite differences method (if 0, uses finite differences)
- `restrict_parameters`: (boolean) if true, restrict to 7*n learnable parameters
- `learnable_parameters`: (array) if `restrict_parameters`, the indices of the 7*n learnable parameters
- `theano_reflection`: (boolean) if true, use the complex reflection code directly from the Arjovsky et al. theano version (this was for testing, I'm fairly sure that code has a bug and this value should not be True)
- `change_of_basis`: (numeric) scale of the uniform distribution to draw the change of basis matrix from (if 0, no change of basis)
- `basis_change`: (matrix) the basis change matrix
- `real`: (boolean) use real parameters?
- `loss_function`: (function) given the parameters, calculate the loss on a given batch
- `learning_rate`: (float) step size to use in stochastic gradient descent

I have made an attempt to ensure the selected options are coherent (with `check_attributes()`) but I probably missed things and you can likely break the code horribly if you ask for something weird.

The `batch_size` and number of batches (`n_batches`) are defined in the `main` function in `learning_unitary_matrices.py`. In the arguments of that function you can also specify:

- `identifier`: a string to additionally mark output files (otherwise they're identified by the value of `d`/`n`, the number of batches and the batch size)
- `n_reps`: number of reps of that given experiment to do (generates new test/train/vali data each time)
- `n_epochs`: number of times the model should go through the training data before getting the test set loss (there is no early stopping)
- `noise`: the standard deviation of the Gaussian noise added to the y's in the data
- `start_from_rep`: start numbering reps from which number? (this is useful for running 3 reps, then realising you ran out of time on the cluster, running 3 more later and combining them into 6 reps without having to modify any output files)
- `real`: generate an orthogonal (real) U?


## How to make this all faster

People have commented that this approach is computationally expensive. This is true. However, my implementation is far from optimial. 

For example, every time I calculate a gradient (in one of n*n directions) using finite differences, I call the `loss_function` of the specific model. If that model is `general_unitary`, this means calling `unitary_matrix()` on its list of parameters. This constructs the corresponding element of the Lie algebra using `lie_algebra_element()`, which in turn runs through the basis elements of the Lie algebra using `lie_algebra_basis_element()`, every time. Even though each time I'm only modifying _one_ parameter, and that one parameter corresponds to a specific Lie algebra basis element. Clearly, pre-computing the element of the Lie algebra and then modifying _that_ each time would save a lot of useless calculation. However, given how I've set up the rest of the code (everything happens through `loss_function()`, which doesn't care or know about intermediate Lie algebra elements) this is a bit tricky to achieve. 

Some (extensive) refactoring could go a long way, but this is (to me) a proof of concept, so for now it's not a high priority.

One could also try to avoid using finite differences at all, which I _am_ working on. :)
