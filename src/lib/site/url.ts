/**
 * Join a site-relative path against the configured base path.
 * Defaults to Astro's BASE_URL ("/" locally, "/<repo>/" on GitHub Pages).
 */
export function url(
  path: string,
  base: string = import.meta.env.BASE_URL
): string {
  const cleanBase = base.endsWith('/') ? base : `${base}/`;
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;
  return cleanBase + cleanPath;
}
