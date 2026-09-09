import { describe, expect, it } from 'vitest'
import stylesheet from './workspace.css?raw'

const tokens = new Map(
  [...stylesheet.matchAll(/--([\w-]+):\s*(#[0-9a-f]{6});/g)].map(match => [match[1], match[2]])
)
function luminance(hex: string) {
  const rgb = [1, 3, 5].map(offset => Number.parseInt(hex.slice(offset, offset + 2), 16) / 255)
  const linear = rgb.map(value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4)
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
}
function contrast(a: string, b: string) {
  const values = [luminance(a), luminance(b)].sort((left, right) => right - left)
  return (values[0] + 0.05) / (values[1] + 0.05)
}

describe('Kana Legal ocean theme', () => {
  it('uses dark native controls and replaces the former green palette', () => {
    expect(stylesheet).toContain('color-scheme: dark')
    expect(tokens.get('bg')).toBe('#101929')
    expect(tokens.get('accent')).toBe('#39c8f0')
    expect(stylesheet).not.toMatch(/#173e32|#244d41|#f5f6f3/)
  })
  it('keeps primary text, controls, and semantic notices above 4.5:1 contrast', () => {
    const pairs = [
      ['text', 'bg'], ['text', 'surface'], ['heading', 'surface'],
      ['muted', 'surface'], ['muted', 'surface-raised'], ['label', 'surface'],
      ['text', 'input'], ['accent-soft', 'surface'], ['heading', 'active'],
      ['button-ink', 'accent'], ['danger', 'danger-bg'], ['warning', 'warning-bg'],
    ]
    for (const [foreground, background] of pairs) {
      expect(contrast(tokens.get(foreground)!, tokens.get(background)!), `${foreground}/${background}`)
        .toBeGreaterThanOrEqual(4.5)
    }
    expect(contrast(tokens.get('button-ink')!, '#20acd6')).toBeGreaterThanOrEqual(4.5)
    expect(contrast('#91a3bd', tokens.get('input')!)).toBeGreaterThanOrEqual(4.5)
  })
})
