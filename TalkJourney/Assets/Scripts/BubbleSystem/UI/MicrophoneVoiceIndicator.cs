using UnityEngine;
using UnityEngine.UI;

namespace TalkJourney.BubbleSystem.UI
{
    [DisallowMultipleComponent]
    public class MicrophoneVoiceIndicator : MonoBehaviour
    {
        [Header("UI")]
        [Tooltip("Image used for the vertical fill inside the microphone icon.")]
        public Image fillImage;

        [Tooltip("Optional RealtimeWhisper instance. If empty, the script resolves the singleton automatically.")]
        public RealtimeWhisper voiceSource;

        [Header("Smoothing")]
        [Tooltip("Higher values make the fill animation smoother and less jumpy.")]
        public float smoothing = 10f;

        [Header("Volume Mapping")]
        [Tooltip("Volume value below this level is treated as silence.")]
        public float minVolume = 0.02f;

        [Tooltip("Volume value at or above this level corresponds to a full fill.")]
        public float maxVolume = 0.35f;

        [Range(0f, 1f)]
        [Tooltip("Current normalized fill amount shown in the indicator.")]
        public float currentFill;

        private float _targetFill;

        private void Start()
        {
            if (voiceSource == null)
            {
                voiceSource = RealtimeWhisper.Instance;
            }
        }

        private void Update()
        {
            float volume = GetVoiceVolume();
            _targetFill = Mathf.InverseLerp(minVolume, maxVolume, volume);
            currentFill = Mathf.Lerp(currentFill, _targetFill, smoothing * Time.deltaTime);

            if (fillImage != null)
            {
                fillImage.fillAmount = currentFill;
            }
        }

        private float GetVoiceVolume()
        {
            if (voiceSource == null)
            {
                return 0f;
            }

            return Mathf.Clamp01(voiceSource.CurrentVolume);
        }
    }
}
