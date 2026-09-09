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
  it('uses locally bundled DM Sans for every UI font declaration', () => {
    expect(stylesheet).not.toMatch(/Manrope|fonts.googleapis.com|@import/)
    expect(stylesheet).toContain('./assets/dm-sans-latin.woff2')
    expect(stylesheet).toContain('./assets/dm-sans-latin-ext.woff2')
    expect(stylesheet).toContain('font-weight: 400 800')
    for (const declaration of stylesheet.matchAll(/font-family:\s*([^;]+);/g)) {
      expect(declaration[1]).toMatch(/DM Sans|inherit/)
    }
  })
  it('supports a compact sidebar and native scrolling with reduced-motion overrides', () => {
    expect(stylesheet).toContain('.brand-name { white-space: nowrap; }')
    expect(stylesheet).toContain('.sidebar-collapsed { --sidebar-width: 84px; }')
    expect(stylesheet).toContain('scrollbar-gutter: stable')
    expect(stylesheet).toContain('scroll-behavior: smooth')
    const reducedMotion = stylesheet.slice(stylesheet.indexOf('@media (prefers-reduced-motion: reduce)'))
    expect(reducedMotion).toContain('html { scroll-behavior: auto; }')
    expect(reducedMotion).toContain('.sidebar, .workspace-body { transition: none; }')
    const mobile = stylesheet.slice(stylesheet.indexOf('@media (max-width: 760px)'))
    expect(mobile).toContain('.sidebar-collapsed .sidebar nav, .sidebar-collapsed .sidebar-bottom { display: none; }')
  })
  it('separates the ocean-blue sidebar from the darker workspace', () => {
    expect(tokens.get('sidebar')).toBe('#19394e')
    expect(luminance(tokens.get('sidebar')!)).toBeGreaterThan(luminance(tokens.get('bg')!) * 2)
    for (const foreground of ['text', 'muted', 'accent']) {
      expect(contrast(tokens.get(foreground)!, tokens.get('sidebar')!), foreground)
        .toBeGreaterThanOrEqual(4.5)
    }
    expect(contrast(tokens.get('text')!, tokens.get('sidebar-hover')!)).toBeGreaterThanOrEqual(4.5)
    expect(contrast(tokens.get('heading')!, tokens.get('sidebar-active')!)).toBeGreaterThanOrEqual(4.5)
  })
  it('uses the bundled wave only in the login hero with a readable navy overlay', () => {
    const hero = stylesheet.match(/\.lens-login > section \{([^}]+)\}/)![1]
    expect(hero).toContain('url("./assets/kana-wave.jpg") center / cover no-repeat')
    expect(hero).toContain('linear-gradient(110deg, #0c1a2ce6, #0c1a2ccc)')
    expect(stylesheet.match(/kana-wave\.jpg/g)).toHaveLength(1)
    // Worst case: pure white foam beneath the least opaque gradient stop.
    const navy = '#0c1a2c'
    const alpha = 0xcc / 255
    const brightestBackground = '#' + [1, 3, 5].map(offset =>
      Math.round(parseInt(navy.slice(offset, offset + 2), 16) * alpha + 255 * (1 - alpha))
        .toString(16).padStart(2, '0')
    ).join('')
    for (const foreground of ['text', 'heading', 'accent', 'accent-soft']) {
      expect(contrast(tokens.get(foreground)!, brightestBackground), foreground)
        .toBeGreaterThanOrEqual(4.5)
    }
  })
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
