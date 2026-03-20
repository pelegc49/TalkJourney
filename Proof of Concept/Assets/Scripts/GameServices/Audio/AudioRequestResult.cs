using UnityEngine;

namespace TalkJourney.BubbleSystem.Audio
{
    public readonly struct AudioRequestResult
    {
        public readonly bool IsSuccess;
        public readonly AudioClip Clip;
        public readonly string Error;

        public AudioRequestResult(bool isSuccess, AudioClip clip, string error)
        {
            IsSuccess = isSuccess;
            Clip = clip;
            Error = error;
        }

        public static AudioRequestResult Success(AudioClip clip)
        {
            return new AudioRequestResult(true, clip, string.Empty);
        }

        public static AudioRequestResult Failure(string error)
        {
            return new AudioRequestResult(false, null, error ?? "Audio request failed.");
        }
    }
}
