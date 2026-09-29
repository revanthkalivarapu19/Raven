/**
 * Suggestions offered under an answer.
 *
 * Only the most recent answer offers them at full strength; in an older part
 * of the conversation they recede, so the current check is always the loudest
 * thing on screen. Labels come from the service layer already written in
 * human language.
 */
export default function FollowUps({ items, onSelect, quiet = false }) {
  return (
    <div className={`followups ${quiet ? 'followups--quiet' : ''}`}>
      <span className="folio followups__label">
        {quiet ? 'Also' : 'You could'}
      </span>
      <div className="followups__row">
        {items.map((item) => (
          <button
            key={item.id}
            type="button"
            className="chip"
            onClick={() => onSelect(item)}
          >
            <span className="chip__label">{item.label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
