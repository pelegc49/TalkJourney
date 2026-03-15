using System.Threading;
using System.Threading.Tasks;

namespace TalkJourney.BubbleSystem.Audio
{
    public interface IAudioBackendClient
    {
        Task<AudioRequestResult> RequestAudioAsync(string audioIdentifier, CancellationToken cancellationToken = default);
    }
}
