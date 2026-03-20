namespace TalkJourney.BubbleSystem.Speech
{
    public interface ISelectionSpeechTarget
    {
        bool TryActivateFromRecognizedText(string recognizedText);
    }
}
