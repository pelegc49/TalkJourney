using System;

namespace TalkJourney.BubbleSystem.Speech
{
    public interface ISpeechRecognitionService
    {
        event Action<string> PhraseRecognized;
    }
}
