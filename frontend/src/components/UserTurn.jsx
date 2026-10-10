export function UserTurn({ claimNumber = '01', text = '', image = null }) {
  return (
    <div className="user-turn-wrapper">
      <div className="user-turn-bubble">
        <span className="user-turn-label">You · Claim {claimNumber}</span>
        {image && (
          <div className="user-turn-attachment">
            <img
              src={typeof image === 'string' ? image : (image.url || image.preview)}
              alt="Attached claim screenshot"
              className="user-turn-image"
            />
          </div>
        )}
        {text && <p className="user-turn-text">{text}</p>}
      </div>
    </div>
  );
}
