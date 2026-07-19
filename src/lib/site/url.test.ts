import { describe, it, expect } from 'vitest';
import { url } from './url';

describe('url', () => {
  it('joins against root base', () => {
    expect(url('chapters/the-neuron/', '/')).toBe('/chapters/the-neuron/');
  });

  it('joins against a repo base without trailing slash', () => {
    expect(url('chapters/the-neuron/', '/neural_networks')).toBe(
      '/neural_networks/chapters/the-neuron/'
    );
  });

  it('joins against a repo base with trailing slash', () => {
    expect(url('chapters/the-neuron/', '/neural_networks/')).toBe(
      '/neural_networks/chapters/the-neuron/'
    );
  });

  it('strips a leading slash on the path', () => {
    expect(url('/about/', '/neural_networks')).toBe('/neural_networks/about/');
  });

  it('returns the base itself for empty path (home link)', () => {
    expect(url('', '/neural_networks')).toBe('/neural_networks/');
  });
});
