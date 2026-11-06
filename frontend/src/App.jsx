import { useState, useRef, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { EmptyState } from './components/EmptyState';
import { Composer } from './components/Composer';
import { UserTurn } from './components/UserTurn';
import { ProcessingBlock } from './components/ProcessingBlock';
import { ResultCard } from './components/ResultCard';
import { ErrorTurn } from './components/ErrorTurn';
import { FollowUpTurn } from './components/FollowUpTurn';
import { WhyResultDialog, AboutDialog } from './components/Dialogs';
import { Toast } from './components/Toast';
import { useConversation } from './hooks/useConversation';
import { verify, followUp, recheckClaim, resetDemoSession } from './services/ravenApi';
import { classifyInputIntent } from './services/followUpIntent';

function App() {
  const {
    turns,
    activeTurnId,
    setActiveTurnId,
    addClaim,
    updateTurn,
    addFollowUp,
    clearConversation,
  } = useConversation();

  const [composerText, setComposerText] = useState('');
  const [screenshot, setScreenshot] = useState(null);
  const [theme, setTheme] = useState(() => {
    try {
      const saved = localStorage.getItem('raven_theme');
      if (saved === 'dark' || saved === 'light') {
        return saved;
      }
    } catch {
      // ignore storage errors
    }
    if (typeof document !== 'undefined') {
      const attr = document.documentElement.getAttribute('data-theme');
      if (attr === 'dark' || attr === 'light') {
        return attr;
      }
    }
    return 'light';
  });

  const transitionTimeoutRef = useRef(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('raven_theme', theme);
    } catch {
      // ignore storage errors
    }
  }, [theme]);

  const handleToggleTheme = () => {
    if (transitionTimeoutRef.current) {
      clearTimeout(transitionTimeoutRef.current);
    }
    document.documentElement.classList.add('theme-transitioning');
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
    transitionTimeoutRef.current = setTimeout(() => {
      document.documentElement.classList.remove('theme-transitioning');
    }, 280);
  };

  // Dialog & Toast states
  const [whyDialogResult, setWhyDialogResult] = useState(null);
  const [isWhyDialogOpen, setIsWhyDialogOpen] = useState(false);
  const [isAboutDialogOpen, setIsAboutDialogOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);

  const whyDialogTriggerRef = useRef(null);
  const aboutDialogTriggerRef = useRef(null);

  const chatScrollRef = useRef(null);
  const turnsEndRef = useRef(null);

  // Scroll newest turn into view
  useEffect(() => {
    if (turns.length > 0) {
      turnsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [turns.length]);

  // Handle step progression for processing turns calling verify service
  useEffect(() => {
    const processingTurn = turns.find((t) => t.status === 'processing');
    if (!processingTurn) return;

    let isMounted = true;

    const timer = setTimeout(async () => {
      if (!isMounted) return;

      if (processingTurn.currentStep < 3) {
        updateTurn(processingTurn.id, {
          currentStep: processingTurn.currentStep + 1,
        });
      } else {
        try {
          const result = await verify({
            text: processingTurn.text,
            image: processingTurn.image,
          });
          if (!isMounted) return;
          updateTurn(processingTurn.id, {
            status: 'done',
            result,
          });
        } catch (err) {
          if (!isMounted) return;
          updateTurn(processingTurn.id, {
            status: 'error',
            error: err?.message || 'Verification failed',
          });
        }
      }
    }, 600);

    return () => {
      isMounted = false;
      clearTimeout(timer);
    };
  }, [turns, updateTurn]);

  const handleAttachScreenshot = (file) => {
    const reader = new FileReader();
    reader.onload = () => {
      setScreenshot({
        name: file.name,
        preview: reader.result,
      });
    };
    reader.readAsDataURL(file);
  };

  const handleSelectExample = (prompt) => {
    addClaim({
      text: prompt,
      image: null,
    });
  };

  const handleSendClaim = async () => {
    const textToSend = composerText.trim();
    if (!textToSend && !screenshot) return;

    // Check if this is a follow-up to an existing conversation
    const lastClaimTurn = [...turns].reverse().find((t) => t.type === 'claim' && t.status === 'done');
    const intent = classifyInputIntent(textToSend);

    if (lastClaimTurn && !screenshot && intent !== 'NEW_CLAIM') {
      const followUpId = addFollowUp(
        textToSend,
        intent === 'RECHECK_CLAIM' ? 'Rechecking the original claim against sources…' : 'Consulting the existing result…',
        lastClaimTurn.id
      );
      setComposerText('');
      setScreenshot(null);

      try {
        if (intent === 'RECHECK_CLAIM') {
          const result = await recheckClaim(lastClaimTurn);
          updateTurn(followUpId, {
            answer: 'Recheck complete for the original claim.',
            result,
            status: 'done',
          });
          return;
        }
        const answerText = await followUp({
          question: textToSend,
          result: lastClaimTurn.result,
          turns,
        });
        updateTurn(followUpId, { answer: answerText });
      } catch {
        updateTurn(followUpId, {
          answer: 'Unable to retrieve follow-up response from sources.',
        });
      }
      return;
    }

    let imageData = null;
    if (screenshot) {
      imageData = {
        name: screenshot.name,
        preview: screenshot.preview,
      };
    }

    addClaim({
      text: textToSend,
      image: imageData,
    });

    setComposerText('');
    setScreenshot(null);
  };

  const handleRetryTurn = (turnId) => {
    updateTurn(turnId, {
      status: 'processing',
      currentStep: 0,
      error: null,
    });
  };

  const handleNewCheck = () => {
    clearConversation();
    setComposerText('');
    setScreenshot(null);
    resetDemoSession();
    const textarea = document.querySelector('.composer-textarea');
    if (textarea) {
      textarea.focus();
    }
  };

  const handleSelectClaim = (claimId) => {
    setActiveTurnId(claimId);
    const element = document.getElementById(claimId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const handleOpenWhyResult = (result, event) => {
    whyDialogTriggerRef.current = event?.currentTarget || document.activeElement;
    setWhyDialogResult(result);
    setIsWhyDialogOpen(true);
  };

  const handleOpenAbout = (event) => {
    aboutDialogTriggerRef.current = event?.currentTarget || document.activeElement;
    setIsAboutDialogOpen(true);
  };

  const handleOpenAnotherSource = () => {
    setToastMessage('Ready to receive additional source or claim.');
    const textarea = document.querySelector('.composer-textarea');
    if (textarea) {
      textarea.focus();
    }
  };

  const claimEntries = turns
    .filter((t) => t.type === 'claim')
    .map((t) => ({
      id: t.id,
      preview: t.preview,
      text: t.text,
    }));

  return (
    <div className="app-container">
      <Sidebar
        claims={claimEntries}
        activeClaimId={activeTurnId}
        onSelectClaim={handleSelectClaim}
        onNewCheck={handleNewCheck}
        onOpenAbout={handleOpenAbout}
      />

      <main className="app-main">
        <TopBar theme={theme} onToggleTheme={handleToggleTheme} />

        <div className="chat-scroll-area" ref={chatScrollRef}>
          <div className="chat-content-container">
            {turns.length === 0 ? (
              <EmptyState onSelectExample={handleSelectExample} />
            ) : (
              turns.map((turn) => (
                <div key={turn.id} id={turn.id} className="turn-container">
                  {turn.type === 'claim' && (
                    <div className="turn-exchange">
                      <UserTurn
                        claimNumber={turn.claimNumber}
                        text={turn.text}
                        image={turn.image}
                      />
                      <div className="turn-response">
                        {turn.status === 'processing' && (
                          <ProcessingBlock currentStep={turn.currentStep} />
                        )}
                        {turn.status === 'error' && (
                          <ErrorTurn message={turn.error} onRetry={() => handleRetryTurn(turn.id)} />
                        )}
                        {turn.status === 'done' && turn.result && (
                          <ResultCard
                            result={turn.result}
                            onOpenWhyResult={(res, e) => handleOpenWhyResult(res, e)}
                            onOpenAnotherSource={handleOpenAnotherSource}
                          />
                        )}
                      </div>
                    </div>
                  )}

                  {turn.type === 'followup' && (
                    <FollowUpTurn
                      question={turn.question}
                      answer={turn.answer}
                      result={turn.result}
                      onOpenWhyResult={handleOpenWhyResult}
                      onOpenAnotherSource={handleOpenAnotherSource}
                    />
                  )}
                </div>
              ))
            )}
            <div ref={turnsEndRef} />
          </div>
        </div>

        <Composer
          text={composerText}
          setText={setComposerText}
          screenshot={screenshot}
          onAttachScreenshot={handleAttachScreenshot}
          onRemoveScreenshot={() => setScreenshot(null)}
          onSubmit={handleSendClaim}
        />

        {/* Dialogs */}
        <WhyResultDialog
          isOpen={isWhyDialogOpen}
          onClose={() => setIsWhyDialogOpen(false)}
          result={whyDialogResult}
          triggerRef={whyDialogTriggerRef}
        />

        <AboutDialog
          isOpen={isAboutDialogOpen}
          onClose={() => setIsAboutDialogOpen(false)}
          triggerRef={aboutDialogTriggerRef}
        />

        {/* Toast confirmation */}
        <Toast
          message={toastMessage}
          onDismiss={() => setToastMessage(null)}
        />
      </main>
    </div>
  );
}

export default App;
