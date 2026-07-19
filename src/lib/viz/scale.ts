/** Linear domain→range mapping with inversion, for SVG/canvas coordinates. */
export interface Scale {
  (value: number): number;
  invert(px: number): number;
}

export function linearScale(
  [d0, d1]: [number, number],
  [r0, r1]: [number, number]
): Scale {
  const scale = ((value: number) =>
    r0 + ((value - d0) / (d1 - d0)) * (r1 - r0)) as Scale;
  scale.invert = (px: number) => d0 + ((px - r0) / (r1 - r0)) * (d1 - d0);
  return scale;
}
