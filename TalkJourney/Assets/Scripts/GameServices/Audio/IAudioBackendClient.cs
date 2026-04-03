using System.Threading;
using System.Threading.Tasks;

namespace TalkJourney.BubbleSystem.Audio
{
    public interface IAudioBackendClient
    {
        Task<AudioRequestResult> RequestAudioFromTextAsync(string text, CancellationToken cancellationToken = default);
    }
}
