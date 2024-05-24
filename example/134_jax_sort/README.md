# install

```
git clone https://github.com/google-research/fast-soft-sort

#Cloning into 'fast-soft-sort'...
#remote: Enumerating objects: 80, done.
#remote: Counting objects: 100% (80/80), done.
#remote: Compressing objects: 100% (50/50), done.
#remote: Total 80 (delta 47), reused 60 (delta 29), pack-reused 0

python setup.py install
(base) agum:~/src/fast-soft-sort$ python
Python 3.12.3 | packaged by conda-forge | (main, Apr 15 2024, 18:38:13) [GCC 12.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import jax.numpy as np
>>> import jax.numpy as jnp
>>> from fast_soft_sort.jax_ops import soft_sort
>>> values = jnp.array([[5.,1,2.],[2,1,5]],dtype=jnp.float64)
<stdin>:1: UserWarning: Explicitly requested dtype <class 'jax.numpy.float64'> requested in array is not available, and will be truncated to dtyes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/google/jax#current-gotc
>>> soft_sort(values,regularization_strength=1.0)
Array([[1.6666666, 2.6666667, 3.6666667],
       [1.6666666, 2.6666667, 3.6666667]], dtype=float32)

```
