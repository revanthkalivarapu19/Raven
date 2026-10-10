import { useState, useEffect, useCallback } from 'react';

const STORAGE_KEY = 'raven_conversation_turns';

export function useConversation() {
  const [turns, setTurns] = useState(() => {
    try {
      const saved = sessionStorage.getItem(STORAGE_KEY);
      if (saved) {
        return JSON.parse(saved);
      }
    } catch (e) {
      console.warn('Failed to load session conversation:', e);
    }
    return [];
  });

  const [activeTurnId, setActiveTurnId] = useState(null);

  useEffect(() => {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(turns));
    } catch (e) {
      console.warn('Failed to save session conversation:', e);
    }
  }, [turns]);

  const addClaim = useCallback((claimData) => {
    const claimCount = turns.filter((t) => t.type === 'claim').length + 1;
    const claimNumber = String(claimCount).padStart(2, '0');
    const newId = `turn-${Date.now()}-${claimCount}`;

    const newTurn = {
      id: newId,
      type: 'claim',
      claimNumber,
      text: claimData.text || '',
      image: claimData.image || null, // dataUrl or object
      preview: claimData.text ? (claimData.text.length > 36 ? claimData.text.slice(0, 36) + '…' : claimData.text) : 'Screenshot claim',
      status: 'processing', // 'processing' | 'done' | 'error'
      currentStep: 0, // 0: Reading your claim, 1: Finding sources, 2: Comparing what they say, 3: Weighing the result
      result: claimData.result || null,
      error: null,
      createdAt: Date.now(),
    };

    setTurns((prev) => [...prev, newTurn]);
    setActiveTurnId(newId);
    return newId;
  }, [turns]);

  const updateTurn = useCallback((turnId, updates) => {
    setTurns((prev) =>
      prev.map((t) => (t.id === turnId ? { ...t, ...updates } : t))
    );
  }, []);

  const addFollowUp = useCallback((question, answerText, targetTurnId) => {
    const followUpId = `followup-${Date.now()}`;
    const newTurn = {
      id: followUpId,
      type: 'followup',
      question,
      answer: answerText,
      targetTurnId,
      createdAt: Date.now(),
    };
    setTurns((prev) => [...prev, newTurn]);
    return followUpId;
  }, []);

  const clearConversation = useCallback(() => {
    setTurns([]);
    setActiveTurnId(null);
    try {
      sessionStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      console.warn('Failed to clear session storage:', e);
    }
  }, []);

  return {
    turns,
    activeTurnId,
    setActiveTurnId,
    addClaim,
    updateTurn,
    addFollowUp,
    clearConversation,
  };
}
