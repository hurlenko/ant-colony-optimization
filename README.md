# Ant Colony Optimization

Simple implementation of Ant Colony Optimization algorithm written in python3. Ant Colony Optimization is a probabilistic technique for solving computational problems which can be reduced to finding good paths through graphs.

## Getting Started

### Prerequisites

- Install Numpy

```bash
pip install numpy
```

- Generate your matrix. See [distance_matrix.txt](distance_matrix.txt) for example. To generate random distance matrix run:

```python
from aco import gen_matrix
matrix = gen_matrix(40)
np.savetxt('distance_matrix.txt', matrix, fmt='%g', delimiter=' ')
```

- Tweak constants in `aco.py`
- Run

```python
python aco.py
```

## Authors

- **pjmattingly** - *Initial work* - [Github repo](https://github.com/pjmattingly/ant-colony-optimization)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details