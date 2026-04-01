using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace TalkJourney.BubbleSystem.Audio
{
    public interface IAudioPlaybackManager
    {
        Task<bool> PlayByTextAsync(string text, CancellationToken cancellationToken = default);
        Task PlaySequenceAsync(IEnumerable<string> texts, CancellationToken cancellationToken = default);
        Task<bool> PlayClipAsync(AudioClip clip, CancellationToken cancellationToken = default);
        void ClearCache();
    }
}
