/** The figures' shared plot domain half-width: plots span [-1.2, 1.2]². */
export const BOUNDARY_DOMAIN = 1.2;

/** Sample `predict` over an n×n grid of cell centers covering the domain.
 *  Row-major from the top-left: y DECREASES with row index, matching screen
 *  coordinates, so the result can be written straight into ImageData. */
export function boundaryGrid(
  predict: (x: number, y: number) => number,
  n: number
): Float32Array {
  const D = BOUNDARY_DOMAIN;
  const out = new Float32Array(n * n);
  const cell = (2 * D) / n;
  for (let row = 0; row < n; row++) {
    const y = D - (row + 0.5) * cell;
    for (let col = 0; col < n; col++) {
      const x = -D + (col + 0.5) * cell;
      out[row * n + col] = predict(x, y);
    }
  }
  return out;
}
