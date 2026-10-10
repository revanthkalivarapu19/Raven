export function FollowUpTurn({ question = '', answer = '' }) {
  return (
    <div className="followup-turn-container">
      {question && (
        <div className="user-turn-wrapper">
          <div className="user-turn-bubble">
            <span className="user-turn-label">You · Follow-up</span>
            <p className="user-turn-text">{question}</p>
          </div>
        </div>
      )}
      <div className="followup-answer-wrapper">
        <span className="followup-raven-label">RAVEN</span>
        <div className="followup-answer-text">
          <p>{answer}</p>
        </div>
      </div>
    </div>
  );
}
