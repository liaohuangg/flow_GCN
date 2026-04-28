# Flow Matching Problem Definition

This rewrite uses a minimal conditional Flow Matching formulation.

- `x_data`: normalized center coordinates for every movable/layout node,
  represented as an `N x 2` tensor in canvas-relative units.
- `x_prior`: standard Gaussian noise with the same shape as `x_data`.
- `x_t`: linear interpolation, `x_t = (1 - t) * x_prior + t * x_data`,
  where `t` is sampled uniformly in `[0, 1]` per graph.
- training target: constant velocity `u_t = x_data - x_prior`.

Coordinate convention:

- positions are node center coordinates
- sizes are node width/height
- legacy dataset placement files store lower-left coordinates; the adapter
  converts them to center coordinates as `center = lower_left + size / 2`
- metrics denormalize positions back to each sample canvas before measuring

Normalization:

- position: `(center - canvas_origin) / canvas_size`
- size: `size / canvas_size`
- node power: divided by max absolute node power within the sample
- boolean flags: cast to `0.0` or `1.0`

Fixed split rule:

- aggregate `dataset.pkl`: use provided `train`; split provided `val` into
  first half validation and second half test
- graph/output pair directories: deterministic shuffle with seed, then
  90% train, 5% validation, 5% test
