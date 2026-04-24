using System;
using TalkJourney.BubbleSystem.Data;

namespace TalkJourney.BubbleSystem.Events
{
    public static class BubbleEventBus
    {
        public static event Action<StageData> StageChanged;
        public static event Action<BubbleData> BubbleHoverEntered;
        public static event Action<BubbleData> BubbleHoverExited;
        public static event Action<BubbleData> BubbleClicked;
        public static event Action<SelectionBubbleData> SelectionCorrect;
        public static event Action<SelectionBubbleData> SelectionIncorrect;
        public static event Action<StageData, SelectionBubbleData> JourneyCompleted;
        public static event Action BubbleSystemHidden;
        public static event Action BypassEnabled;
        public static event Action<string> SpeechPhraseRecognized;
        public static event Action<string> AudioPlaybackStarted;
        public static event Action<string> AudioPlaybackEnded;
        public static event Action<string> AudioPlaybackFailed;

        public static void PublishStageChanged(StageData stageData)
        {
            StageChanged?.Invoke(stageData);
        }

        public static void PublishBubbleHoverEntered(BubbleData bubbleData)
        {
            BubbleHoverEntered?.Invoke(bubbleData);
        }

        public static void PublishBubbleHoverExited(BubbleData bubbleData)
        {
            BubbleHoverExited?.Invoke(bubbleData);
        }

        public static void PublishBubbleClicked(BubbleData bubbleData)
        {
            BubbleClicked?.Invoke(bubbleData);
        }

        public static void PublishSelectionCorrect(SelectionBubbleData selectionData)
        {
            SelectionCorrect?.Invoke(selectionData);
        }

        public static void PublishSelectionIncorrect(SelectionBubbleData selectionData)
        {
            SelectionIncorrect?.Invoke(selectionData);
        }

        public static void PublishJourneyCompleted(StageData stageData, SelectionBubbleData terminalSelection)
        {
            JourneyCompleted?.Invoke(stageData, terminalSelection);
        }

        public static void PublishBubbleSystemHidden()
        {
            BubbleSystemHidden?.Invoke();
        }

        public static void PublishBypassEnabled()
        {
            BypassEnabled?.Invoke();
        }

        public static void PublishSpeechPhraseRecognized(string recognizedText)
        {
            SpeechPhraseRecognized?.Invoke(recognizedText);
        }

        public static void PublishAudioPlaybackStarted(string audioIdentifier)
        {
            AudioPlaybackStarted?.Invoke(audioIdentifier);
        }

        public static void PublishAudioPlaybackEnded(string audioIdentifier)
        {
            AudioPlaybackEnded?.Invoke(audioIdentifier);
        }

        public static void PublishAudioPlaybackFailed(string audioIdentifier)
        {
            AudioPlaybackFailed?.Invoke(audioIdentifier);
        }
    }
}
