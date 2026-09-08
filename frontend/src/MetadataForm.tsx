import type { AcquisitionMetadata } from './types'

export function MetadataForm({ value, onChange }: {
  value: AcquisitionMetadata; onChange: (value: AcquisitionMetadata) => void
}) {
  function update<K extends keyof AcquisitionMetadata>(key: K, next: AcquisitionMetadata[K]) {
    onChange({ ...value, [key]: next })
  }
  return <fieldset className="metadata-form">
    <legend>Acquisition context</legend>
    <p>Confirm these review inputs. Agency, contract type, and commercial status affect which rules are evaluated.</p>
    <div className="metadata-grid">
      <label>Agency<input className="text-input" value={value.agency} maxLength={200} onChange={e => update('agency', e.target.value)} /></label>
      <label>Solicitation number<input className="text-input" value={value.solicitation_number} maxLength={100} onChange={e => update('solicitation_number', e.target.value)} /></label>
      <label>Contract type<select value={value.contract_type} onChange={e => update('contract_type', e.target.value)}>{['firm-fixed-price', 'cost-reimbursement', 'time-and-materials', 'indefinite-delivery', 'other'].map(v => <option key={v}>{v}</option>)}</select></label>
      <label>Acquisition stage<select value={value.acquisition_stage} onChange={e => update('acquisition_stage', e.target.value)}>{['pre-solicitation', 'solicitation', 'evaluation', 'award', 'post-award'].map(v => <option key={v}>{v}</option>)}</select></label>
      <label>Estimated value (USD)<input className="text-input" type="number" min="0" step="any" value={value.estimated_value} onChange={e => update('estimated_value', Number(e.target.value))} /></label>
      <label>Performance months<input className="text-input" type="number" min="1" max="240" value={value.performance_months} onChange={e => update('performance_months', Number(e.target.value))} /></label>
      <label>Set-aside<input className="text-input" maxLength={100} value={value.set_aside} onChange={e => update('set_aside', e.target.value)} /></label>
      <label>Place of performance<input className="text-input" maxLength={300} value={value.place_of_performance} onChange={e => update('place_of_performance', e.target.value)} /></label>
    </div>
    <label className="metadata-check"><input type="checkbox" checked={value.commercial_product} onChange={e => onChange({ ...value, commercial_product: e.target.checked, cots_only: e.target.checked && value.cots_only })} /> Commercial product/service</label>
    <label className="metadata-check"><input type="checkbox" checked={value.cots_only} disabled={!value.commercial_product} onChange={e => update('cots_only', e.target.checked)} /> Solely commercial off-the-shelf (COTS)</label>
  </fieldset>
}
